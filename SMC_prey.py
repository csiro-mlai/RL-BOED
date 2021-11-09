"""
Implementation of SMC baseline from `Sequential experimental design for
predator–prey functional response experiments`.
It makes calls to R code authored by Hayden Moffat
"""

import argparse
import datetime
import logging
import os
import joblib


import pyro
import subprocess
import time
import torch
import rpy2.robjects as robjects

from pyro.contrib.util import rexpand
from pyro.envs.adaptive_design_env import AdaptiveDesignEnv
from pyro.models.adaptive_experiment_model import PreyModel


# TODO read from torch float spec
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
epsilon = torch.tensor(2 ** -22)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def main(num_steps, num_reps, experiment_name, output_dir, seed, loglevel):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(loglevel))
    logging.basicConfig(level=numeric_level)

    if not output_dir:
        output_dir = "run_outputs/prey/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.results'
    results_file = os.path.join(os.path.dirname(__file__), results_file)
    try:
        os.remove(results_file)
    except OSError:
        logging.info("File {} does not exist yet".format(results_file))

    logging.info("Type {}".format("SMC"))
    pyro.clear_param_store()
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = int(torch.rand(tuple()) * 2 ** 30)
        pyro.set_rng_seed(seed)

    env = AdaptiveDesignEnv(None, torch.zeros(2),
                            PreyModel(n_parallel=1), num_steps,
                            int(1e5))
    env.reset(1)
    spce = 0
    model = PreyModel(n_parallel=1)
    true_theta = env.theta0
    d_stars = torch.tensor([])
    y_stars = torch.tensor([])
    spces = torch.tensor([])
    results = {'git-hash': get_git_revision_hash(), 'typ': "SMC", 'seed': seed,
               'true_theta': true_theta}
    # setup R environment
    r = robjects.r
    r.source('scripts/R/SMC_setup.R')
    r(f'set.seed({seed})')

    # initiate R loop
    r.source('scripts/R/SMC_init.R')

    for step in range(num_steps):
        logging.info("Step {}".format(step))

        # Compute optimal design with R code
        t = time.time()
        r(f'i <- {step + 1}')
        r.source('scripts/R/SMC_design.R')
        elapsed = time.time() - t
        logging.info('elapsed design time {}'.format(elapsed))

        # Grab design from R
        d_star = rexpand(torch.tensor(r['idx']).int(), 1, 1, 1)
        logging.info('design {} {}'.format(d_star.squeeze(), d_star.shape))
        d_stars = torch.cat([d_stars, d_star], dim=-2)
        results['d_stars'] = d_stars

        # Get experimental outcome from model
        y_star = model.run_experiment(d_star, true_theta)
        logging.info('y_star {} {}'.format(y_star.squeeze(), y_star.shape))
        y_stars = torch.cat([y_stars, y_star], dim=-1)
        results['y_stars'] = y_stars

        # Send experiment outcome to R
        r(f'data[i,2] <- {y_star.int().item()}')
        r.source("scripts/R/SMC_update.R")
        print(r["data"])

        # estimate EIG with sPCE
        spce += env.get_reward(y_star, d_star)
        spces = torch.cat([spces, spce])
        results['spces'] = spces
        logging.info(f"spce {spce} {spce.shape}")

        # save results
        for k, v in results.items():
            if hasattr(v, "cpu"):
                results[k] = v.cpu()
        with open(results_file, 'wb') as f:
            joblib.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prey population"
                                                 "SMC experiment design")
    parser.add_argument("--num-steps", nargs="?", default=10, type=int)
    parser.add_argument("--num-reps", nargs="?", default=10, type=int)
    parser.add_argument("--name", nargs="?", default="", type=str)
    parser.add_argument("--dest", nargs="?", default="", type=str)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--loglevel", default="info", type=str)
    args = parser.parse_args()
    main(args.num_steps, args.num_reps, args.name, args.dest, args.seed,
         args.loglevel)
