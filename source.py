import argparse
import datetime
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
from pyro.contrib.util import iter_plates_to_shape
from pyro.util import is_bad

# TODO read from torch float spec
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def make_source_model(theta_mu, theta_sig, observation_sd, alpha=1,
                      observation_label="y", b=1e-1, m=1e-4):
    def source_model(design):
        # pyro.set_rng_seed(10)
        if is_bad(design):
            raise ArithmeticError("bad design, contains nan or inf")
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            theta_shape = batch_shape + theta_mu.shape[-2:]
            theta = pyro.sample(
                "theta",
                dist.Normal(
                    theta_mu.expand(theta_shape),
                    theta_sig.expand(theta_shape)
                ).to_event(2)
            )
            distance = torch.square(
                design.unsqueeze(-2) - theta.unsqueeze(-3)
            ).sum(dim=-1)
            ratio = alpha / (m + distance)
            mu = b + ratio.sum(dim=-1)
            emission_dist = dist.Normal(
                torch.log(mu), observation_sd
            ).to_event(1)
            y = pyro.sample(observation_label, emission_dist)
            return y

    return source_model


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
         obs_sd, loglevel, policy_src, estimate_eig):
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
    observation_sd = torch.tensor(obs_sd)

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

        theta_mu = torch.zeros(num_parallel, 1, k, design_dim)
        theta_sig = torch.ones(num_parallel, 1, k, design_dim)
        theta_0 = torch.distributions.Normal(theta_mu, theta_sig).sample()
        logging.info(f"theta_0 {theta_0}")

        true_model = pyro.condition(
            make_source_model(theta_mu, theta_sig, observation_sd),
            {"theta": theta_0})
        if estimate_eig:
            print("estimate_eig")
            data = joblib.load(policy_src)
            eval_env = data['env'].env
            eval_env.l = int(1e7)
            eval_env.reset(n_parallel=num_parallel)
            spce = 0

        d_star_designs = torch.tensor([])
        ys = torch.tensor([])

        for step in range(num_steps):
            logging.info("Step {}".format(step))
            model = make_source_model(theta_mu, theta_sig, observation_sd)
            results = {'typ': typ, 'step': step, 'lengthscale': lengthscale,
                       'git-hash': get_git_revision_hash(), 'seed': seed,
                        'observation_sd': observation_sd, 'theta_0': theta_0,
                       'num_gradient_steps': num_gradient_steps, 'num_samples': num_samples,
                       'num_contrast_samples': num_contrast_samples, 'num_acquisition': num_acquisition}

            # Design phase
            t = time.time()

            if typ == 'pce-grad':
                model_learn_xi = make_learn_xi_model(model)
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
                d_star_design = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2)

            elapsed = time.time() - t
            logging.info('elapsed design time {}'.format(elapsed))
            results['design_time'] = elapsed
            results['d_star_design'] = d_star_design
            logging.info('design {} {}'.format(d_star_design.squeeze(), d_star_design.shape))
            # update using only the result of the first experiment
            prior = make_source_model(
                torch.zeros(num_parallel, 1, k, design_dim),
                torch.ones(num_parallel, 1, k, design_dim),
                observation_sd
            )
            d_star_designs = torch.cat([d_star_designs, d_star_design], dim=-2)
            # pyro.set_rng_seed(10)
            if estimate_eig:
                y = eval_env.model.run_experiment(
                    d_star_design, eval_env.theta0)
                spce += eval_env.get_reward(y, d_star_design)
                results['spce'] = spce
                if step == 0:
                    results['theta0'] = {
                        k: v.cpu() for k, v in eval_env.theta0.items()}
                logging.info(f"spce {spce} {spce.shape}")
            else:
                y = true_model(d_star_design)
            ys = torch.cat([ys, y], dim=-1)
            logging.info('ys {} {}'.format(ys.squeeze(), ys.shape))
            results['y'] = y

            # pyro.set_rng_seed(10)
            loss = elbo_learn(
                prior, d_star_designs, ["y"], ["theta"], elbo_n_samples, elbo_n_steps,
                partial(elboguide, dim=num_parallel), {"y": ys}, optim.Adam({"lr": elbo_lr})
            )
            theta_mu = pyro.param("theta_mu").detach().data.clone()
            theta_sig = pyro.param("theta_sig").detach().data.clone()

            logging.info(f"theta_mu {theta_mu}\ntheta_sig {theta_sig}")
            results['theta_mu'], results['theta_sig'] = theta_mu, theta_sig
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
    parser.add_argument("--num-gradient-steps", default=1000, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--num-contrast-samples", default=10, type=int)
    parser.add_argument("--num-acquisition", default=8, type=int)
    parser.add_argument("--observation-sd", default=0.5, type=float)
    parser.add_argument("--policy-src", default="", type=str)
    parser.add_argument("--estimate-eig", dest="estimate_eig",
                        action='store_true')
    parser.set_defaults(estimate_eig=False)
    args = parser.parse_args()
    main(args.num_steps, args.num_parallel, args.name, args.typs, args.seed, args.lengthscale,
         args.num_gradient_steps, args.num_samples, args.num_contrast_samples, args.num_acquisition,
         args.observation_sd, args.loglevel, args.policy_src, args.estimate_eig)
