import argparse
import datetime
import logging
import os
import pickle
import pyro
import pyro.distributions as dist
import pyro.optim as optim
import subprocess
import time
import torch

from contextlib import ExitStack
from functools import partial
from pyro.contrib.oed.eig import elbo_learn
from pyro.contrib.util import iter_plates_to_shape, lexpand, rexpand
from pyro.envs.adaptive_design_env import AdaptiveDesignEnv
from pyro.models.adaptive_experiment_model import PreyModel
from torch.distributions import LogNormal


# TODO read from torch float spec
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
epsilon = torch.tensor(2 ** -22)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def elboguide(design, dim=10):
    # pyro.set_rng_seed(10)
    a_mu = pyro.param("a_mu", torch.ones(dim, 1, 1) * -1.4)
    a_sig = pyro.param("a_sig", torch.ones(dim, 1, 1) * 1.35,
                       constraint=torch.distributions.constraints.positive)
    th_mu = pyro.param("th_mu", torch.ones(dim, 1, 1) * -1.4)
    th_sig = pyro.param("th_sig", torch.ones(dim, 1, 1) * 1.35,
                        constraint=torch.distributions.constraints.positive)
    batch_shape = design.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)
        a_shape = batch_shape + a_mu.shape[-1:]
        pyro.sample("a", dist.LogNormal(a_mu.expand(a_shape),
                                        a_sig.expand(a_shape)).to_event(1))
        th_shape = batch_shape + th_mu.shape[-1:]
        pyro.sample("th", dist.LogNormal(th_mu.expand(th_shape),
                                         th_sig.expand(th_shape)).to_event(1))


def main(num_steps, num_parallel, experiment_name, typs, seed,
         n_inner, n_outer, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(loglevel))
    logging.basicConfig(level=numeric_level)

    output_dir = "run_outputs/prey/"
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
        design_dim = 1

        env = AdaptiveDesignEnv(None, torch.zeros(2),
                                PreyModel(n_parallel=num_parallel), num_steps,
                                int(1e5))
        env.reset(num_parallel)
        spce = 0

        model = PreyModel(n_parallel=num_parallel)
        init_entropy = LogNormal(model.a_mu, model.a_sig).entropy() +\
                       LogNormal(model.th_mu, model.th_sig).entropy()
        true_theta = env.theta0
        d_stars = torch.tensor([])
        y_stars = torch.tensor([])

        for step in range(num_steps):
            logging.info("Step {}".format(step))
            results = {'git-hash': get_git_revision_hash(), 'typ': typ,
                       'step': step,  'seed': seed, 'n_inner': n_inner,
                       'true_theta': true_theta}

            # Design phase
            t = time.time()
            if typ == 'pce':
                # Compute PCE for each possible design
                # do this separately for `num_parallel` experiments
                # X = lexpand(torch.arange(1, 301), n_outer, num_parallel)
                # X = rexpand(X, 1, design_dim)

                # sample `n_inner` theta_l's and `n_outer` theta_0's
                thetas = model.sample_theta(n_inner + n_outer)
                # for k, v in thetas.items():
                #     dims = list(v.shape)
                #     dims[-2] = 300
                #     thetas[k] = thetas[k].expand(dims)
                theta0 = {k: v[:n_outer] for k, v in thetas.items()}
                pces = []
                for i in range(1, 301):
                    X = torch.ones(n_outer, num_parallel, 1, 1, design_dim) * i
                    # generate `n_outer` samples from p(y, theta_0 | X)
                    y = model.run_experiment(X, theta0)
                    # each y has its own theta0, and theta_l's are shared
                    theta_dict = {
                        k:
                        torch.stack(
                            [torch.cat([v[i].unsqueeze(0), v[n_outer:]])
                                for i in range(n_outer)],
                            dim=1
                        )
                        for k, v in thetas.items()
                    }
                    log_probs = model.get_likelihoods(y, X, theta_dict)
                    # we can subtract constant log(L+1) and maintain order
                    rel_pce = log_probs[0] - torch.logsumexp(log_probs, dim=0)
                    pces.append(rel_pce.mean(dim=0).squeeze())
                pces = torch.stack(pces)

                # pick the best design for each of num_parallel experiments
                max_eig, d_star_index = pces.max(dim=0)
                max_eig += torch.tensor(n_inner + 1.).log()
                logging.info('max EIG {}'.format(max_eig))
                results['max EIG'] = max_eig
                d_star = d_star_index.reshape(-1, 1, 1, design_dim) + 1

            elif typ == 'rand':
                d_star = torch.randint(1, 301, (num_parallel, 1, 1, design_dim))

            results['rng_state'] = torch.get_rng_state()
            elapsed = time.time() - t
            logging.info('elapsed design time {}'.format(elapsed))

            results['design_time'] = elapsed
            results['d_star'] = d_star
            logging.info('design {} {}'.format(d_star.squeeze(), d_star.shape))
            d_stars = torch.cat([d_stars, d_star], dim=-2)
            y_star = model.run_experiment(d_star, true_theta)
            y_stars = torch.cat([y_stars, y_star], dim=-1)
            results['y'] = y_star
            logging.info('ys {} {}'.format(y_stars.squeeze(), y_stars.shape))

            # learn posterior with VI
            t = time.time()
            if typ == 'pce':
                model.reset(num_parallel)
                prior = model.make_model()
                loss = elbo_learn(
                    prior, d_stars, ["y"], ["a", "th"], elbo_n_samples,
                    elbo_n_steps, partial(elboguide, dim=num_parallel),
                    {"y": y_stars}, optim.Adam({"lr": elbo_lr})
                )
                a_mu = pyro.param("a_mu").detach().data.clone()
                a_sig = pyro.param("a_sig").detach().data.clone()
                th_mu = pyro.param("th_mu").detach().data.clone()
                th_sig = pyro.param("th_sig").detach().data.clone()

                logging.info("a_mu {} \n a_sig {} \n th_mu {} \n th_sig {}".format(
                    a_mu.squeeze(), a_sig.squeeze(), th_mu.squeeze(), th_sig.squeeze()))
                results['a_mu'], results['a_sig'] = a_mu, a_sig
                results['th_mu'], results['th_sig'] = th_mu, th_sig
                model.a_mu, model.a_sig = a_mu, a_sig
                model.th_mu, model.th_sig = th_mu, th_sig
                entropy = LogNormal(model.a_mu, model.a_sig).entropy() +\
                          LogNormal(model.th_mu, model.th_sig).entropy()
                logging.info(f'EIG {(init_entropy - entropy).squeeze()}')
            logging.info(f'posterior learning time {time.time() - t}')

            # estimate EIG with sPCE
            spce += env.get_reward(y_star, d_star)
            results['spce'] = spce
            logging.info(f"spce {spce} {spce.shape}")
            for k, v in results.items():
                if hasattr(v, "cpu"):
                    results[k] = v.cpu()

            with open(results_file, 'ab') as f:
                pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prey population"
                                                 " iterated experiment design")
    parser.add_argument("--num-steps", nargs="?", default=10, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=10, type=int)
    parser.add_argument("--name", nargs="?", default="", type=str)
    parser.add_argument("--typs", nargs="?", default="rand", type=str)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--loglevel", default="info", type=str)
    parser.add_argument("--n-inner", default=100, type=int)
    parser.add_argument("--n-outer", default=100, type=int)
    args = parser.parse_args()
    main(args.num_steps, args.num_parallel, args.name, args.typs, args.seed,
         args.n_inner, args.n_outer, args.loglevel)
