import argparse
import datetime
import pickle
import subprocess
import time
import warnings
from contextlib import ExitStack

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss, differentiable_pce_eig, \
    _differentiable_ace_eig_loss
from pyro.contrib.util import iter_plates_to_shape

from death_process_rb import nmc_eig


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


N = 10
prior_mean = torch.tensor(0.)
prior_sd = torch.tensor(1.0)


def make_model_learn_xi(xi_init):
    def model_learn_xi(design_prototype):
        xi = pyro.param('xi', xi_init, constraint=constraints.positive)
        batch_shape = design_prototype.shape
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            b = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))
            p1 = 1 - torch.exp(-b * xi[..., 0])
            infected1 = pyro.sample("i1", dist.Binomial(N, p1))
            p2 = 1 - torch.exp(-b * xi[..., 1])
            infected2 = pyro.sample("i2", dist.Binomial(N - infected1, p2))
            return infected1, infected2

    return model_learn_xi


def make_posterior_guide(prior_mean, prior_sd):
    def posterior_guide(y_dict, design_prototype, observation_labels, target_labels):
        batch_shape = design_prototype.shape
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            posterior_mean = pyro.param("q_mean", prior_mean.clone().expand((batch_shape[-1], 66)))
            posterior_sd = 1e-6 + pyro.param("q_sd", prior_sd.clone().expand((batch_shape[-1], 66)),
                                             constraint=constraints.positive)
            x, y = N - y_dict["i1"], y_dict["i2"]
            indices = (.5 * x * (x + 1) + y).long()
            selected_mean = posterior_mean[torch.arange(batch_shape[-1]), indices]
            selected_sd = posterior_sd[torch.arange(batch_shape[-1]), indices]
            pyro.sample("b", dist.LogNormal(selected_mean, selected_sd))

    return posterior_guide


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))

    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim, time_budget):
    if time_budget is not None:
        num_steps = 100000000000
    params = None
    est_loss_history = []
    xi_history = []
    t = time.time()
    wall_times = []
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples, evaluation=True)
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
        agg_loss.backward(retain_graph=True)
        if step % 200 == 0:
            est_loss_history.append(loss)
            wall_times.append(time.time() - t)
            xi_history.append(pyro.param('xi').detach().clone())
        optim(params)
        optim.step()
        print(pyro.param("xi"))
        if time.time() - t > time_budget:
            break

    xi_history.append(pyro.param('xi').detach().clone())
    wall_times.append(time.time() - t)

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)
    wall_times = torch.tensor(wall_times)

    return xi_history, est_loss_history, wall_times


def main(num_steps, time_budget, experiment_name, num_parallel, estimators, seed, start_lr, end_lr, num_samples):
    output_dir = "./run_outputs/gradinfo/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.pickle'
    estimators = estimators.split(",")

    for estimator in estimators:
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)

        xi_init = .01 + 4.99 * torch.rand((num_parallel, 2))
        model_learn_xi = make_model_learn_xi(xi_init)

        # Fix correct loss
        if estimator == 'posterior':
            guide = make_posterior_guide(prior_mean, prior_sd)
            loss = _differentiable_posterior_loss(model_learn_xi, guide, ["i1", "i2"], ["b"])

        elif estimator == 'pce':
            eig_loss = lambda d, N, **kwargs: differentiable_pce_eig(
                model=model_learn_xi, design=d, observation_labels=["i1", "i2"], target_labels=["b"], N=N, M=N,
                **kwargs)
            loss = neg_loss(eig_loss)

        elif estimator == 'ace':
            guide = make_posterior_guide(prior_mean, prior_sd)
            eig_loss = _differentiable_ace_eig_loss(model_learn_xi, guide, num_samples, ["i1", "i2"], ["b"])
            loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr / start_lr) ** (1 / num_steps)
        if time_budget is not None and gamma < 1:
            warnings.warn("With time_budget set, we may not end on the correct learning rate")
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(num_parallel)  # this is annoying, code needs refactor

        xi_history, est_loss_history, wall_times = opt_eig_loss_w_history(
            design_prototype, loss, num_samples=num_samples, num_steps=num_steps, optim=scheduler,
            time_budget=time_budget)

        if estimator == 'posterior':
            prior_entropy = dist.Normal(prior_mean, prior_sd).entropy()
            est_eig_history = prior_entropy - est_loss_history
        else:
            est_eig_history = -est_loss_history

        eig_history = []
        for i in range(xi_history.shape[0] - 1):
            eig_history.append(nmc_eig(xi_history[i, ...], prior_mean, prior_sd, n_samples=20000))
        eig_history.append(nmc_eig(xi_history[-1, ...], prior_mean, prior_sd, n_samples=200000))
        eig_history = torch.stack(eig_history)

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history, 'est_eig_history': est_eig_history, 'eig_history': eig_history,
                   'wall_times': wall_times}

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization for Death Process")
    parser.add_argument("--num-steps", default=2000, type=int)
    parser.add_argument("--time-budget", default=None, type=float)
    parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.001, type=float)
    parser.add_argument("--end-lr", default=0.001, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.time_budget, args.name, args.num_parallel, args.estimator, args.seed, args.start_lr,
         args.end_lr, args.num_samples)
