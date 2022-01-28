import argparse
import datetime
import sys

import joblib
import logging
from torch.distributions import transform_to
import os
import pickle
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.optim as optim
import subprocess
import time
import torch

from contextlib import ExitStack
from functools import partial
from pyro.contrib.oed.eig import elbo_learn, opt_eig_ape_loss
from pyro.contrib.oed.differentiable_eig import differentiable_pce_eig
from pyro.contrib.util import iter_plates_to_shape, lexpand
from pyro.envs.adaptive_design_env import AdaptiveDesignEnv, UPPER, LOWER
from pyro.models.adaptive_experiment_model import SourceModel
from pyro.util import is_bad

# TODO read from torch float spec
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])



def make_learn_xi_model(model):
    def model_learn_xi(design_prototype):
        design = pyro.param("xi")
        design = design.expand(design_prototype.shape)
        return model(design)

    return model_learn_xi


def elboguide(design, dim=10):
    # pyro.set_rng_seed(10)
    theta_mu = pyro.param("theta_mu", torch.zeros(dim, 1, 2, 2))
    theta_sig = pyro.param("theta_sig", torch.ones(dim, 1, 2, 2),
                           constraint=torch.distributions.constraints.positive)
    batch_shape = design.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)
        theta_shape = batch_shape + theta_mu.shape[-2:]
        pyro.sample("theta", dist.Normal(
            theta_mu.expand(theta_shape),
            theta_sig.expand(theta_shape)).to_event(2)
        )


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))

    return new_loss


def main(num_steps, num_parallel, experiment_name, typs, seed, lengthscale,
         num_gradient_steps, num_samples, num_contrast_samples, num_acquisition,
         loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(loglevel))
    logging.basicConfig(level=numeric_level)

    output_dir = "run_outputs/source/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.result_stream.pickle'
    results_file = os.path.join(os.path.dirname(__file__), results_file)
    try:
        os.remove(results_file)
    except OSError:
        logging.info("File {} does not exist yet".format(results_file))
    typs = typs.split(",")

    for typ in typs:
        logging.info("Type {}".format(typ))
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)
        elbo_n_samples, elbo_n_steps, elbo_lr = 10, 1000, 0.04
        k = 2
        design_dim = 2

        env_lower = AdaptiveDesignEnv(None, torch.zeros(2),
                                      SourceModel(n_parallel=num_parallel), num_steps,
                                      int(1e6), bound_type=LOWER)
        env_upper = AdaptiveDesignEnv(None, torch.zeros(2),
                                      SourceModel(n_parallel=num_parallel), num_steps,
                                      int(1e6), bound_type=UPPER)
        env_lower.reset(num_parallel)
        env_upper.reset(num_parallel)
        spce, snmc = 0, 0
        model = SourceModel(n_parallel=num_parallel)
        true_theta = env_lower.theta0
        env_upper.theta0, env_upper.thetas = env_lower.theta0, env_lower.thetas
        d_stars = torch.tensor([])
        y_stars = torch.tensor([])

        for step in range(num_steps):
            logging.info("Step {}".format(step))
            results = {'typ': typ, 'step': step, 'lengthscale': lengthscale,
                       'git-hash': get_git_revision_hash(), 'seed': seed,
                       'num_gradient_steps': num_gradient_steps, 'num_samples': num_samples,
                       'num_contrast_samples': num_contrast_samples, 'num_acquisition': num_acquisition}

            # Design phase
            t0 = time.time()

            if typ == 'pce':
                model_learn_xi = make_learn_xi_model(model.make_model())
                grad_start_lr, grad_end_lr = 0.001, 0.001

                # Suggested num_gradient_steps = 2500
                eig_loss = lambda d, N, **kwargs: differentiable_pce_eig(
                    model=model_learn_xi, design=d, observation_labels=["y"],
                    target_labels=["theta"],
                    N=N, M=num_contrast_samples, **kwargs)
                loss = neg_loss(eig_loss)

                constraint = torch.distributions.constraints.interval(-8., 8.)
                xi_init = -8 + 16 * torch.rand((num_parallel, num_acquisition, 1, design_dim))
                logging.info('init_design {} {}'.format(xi_init.squeeze(), xi_init.shape))
                pyro.param("xi", xi_init, constraint=constraint)
                pyro.get_param_store().replace_param("xi", xi_init, pyro.param("xi"))
                design_prototype = torch.zeros(num_parallel, num_acquisition, 1,
                                               design_dim)  # this is annoying, code needs refactor

                start_lr, end_lr = grad_start_lr, grad_end_lr
                gamma = (end_lr / start_lr) ** (1 / num_gradient_steps)
                scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                                      'gamma': gamma})
                ape = opt_eig_ape_loss(design_prototype, loss, num_samples=num_samples, num_steps=num_gradient_steps,
                                       optim=scheduler, final_num_samples=500)
                min_ape, d_star_index = torch.min(ape, dim=1)
                logging.info('min loss {}'.format(min_ape))
                results['min loss'] = min_ape
                X = pyro.param("xi").detach().clone()
                d_star = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2)

            elif typ == "rand":
                d_star = -8 + 16 * torch.rand((num_parallel, num_acquisition, 1, design_dim))
                d_star = lexpand(d_star[0], num_parallel)

            else:
                sys.exit(f'''optimisation type must be in ["rand", "pce" 
                but was {typ}''')

            elapsed = time.time() - t0
            logging.info('elapsed design time {}'.format(elapsed))
            results['rng_state'] = torch.get_rng_state()
            results['design_time'] = elapsed
            results['d_star'] = d_star
            logging.info('design {} {}'.format(d_star.squeeze(), d_star.shape))

            d_stars = torch.cat([d_stars, d_star], dim=-3)
            y_star = model.run_experiment(d_star, true_theta)
            y_stars = torch.cat([y_stars, y_star], dim=-1)
            logging.info(f'y_stars {y_stars.squeeze()} {y_stars.shape}')
            results['y'] = y_star

            if typ == "pce":
                # don't bother inferring posteriors for random designs
                model.reset(num_parallel)
                prior = model.make_model()

                # pyro.set_rng_seed(10)
                loss = elbo_learn(
                    prior, d_stars, ["y"], ["theta"], elbo_n_samples, elbo_n_steps,
                    partial(elboguide, dim=num_parallel), {"y": y_stars}, optim.Adam({"lr": elbo_lr})
                )
                theta_mu = pyro.param("theta_mu").detach().data.clone()
                theta_sig = pyro.param("theta_sig").detach().data.clone()
                model.theta_mu, model.theta_sig = theta_mu, theta_sig

                logging.info(f"theta_mu {theta_mu}\ntheta_sig {theta_sig}")
                results['theta_mu'], results['theta_sig'] = theta_mu, theta_sig

            results['time'] = time.time() - t0
            # estimate EIG with sPCE
            spce += env_lower.get_reward(y_star, d_star)
            snmc += env_upper.get_reward(y_star, d_star)
            results['spce'] = spce
            logging.info(f"spce {spce} {spce.shape}")
            results['snmc'] = snmc
            logging.info(f"snmc {snmc} {snmc.shape}")
            for key, val in results.items():
                if hasattr(val, "cpu"):
                    results[key] = val.cpu()

            with open(results_file, 'ab') as f:
                pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source Location"
                                                 " iterated experiment design")
    parser.add_argument("--num-steps", nargs="?", default=30, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=10, type=int)
    parser.add_argument("--name", nargs="?", default="", type=str)
    parser.add_argument("--typs", nargs="?", default="rand", type=str)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--lengthscale", nargs="?", default=10., type=float)
    parser.add_argument("--loglevel", default="info", type=str)
    parser.add_argument("--num-gradient-steps", default=2500, type=int)
    parser.add_argument("--num-samples", default=500, type=int)
    parser.add_argument("--num-contrast-samples", default=500, type=int)
    parser.add_argument("--num-acquisition", default=1, type=int)
    parser.add_argument("--policy-src", default="", type=str)
    parser.add_argument("--estimate-eig", dest="estimate_eig",
                        action='store_true')
    parser.set_defaults(estimate_eig=False)
    args = parser.parse_args()
    main(args.num_steps, args.num_parallel, args.name, args.typs, args.seed, args.lengthscale,
         args.num_gradient_steps, args.num_samples, args.num_contrast_samples, args.num_acquisition,
         args.loglevel)
