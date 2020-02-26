import argparse
import datetime
import math
import pickle
import subprocess
import time
import warnings
from contextlib import ExitStack
from functools import lru_cache

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.contrib.util import iter_plates_to_shape


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


N = 10
prior_mean = torch.tensor(0.)
prior_sd = torch.tensor(1.)


@lru_cache(5)
def make_y_space(n):
    space = []
    for i in range(n + 1):
        for j in range(n - i + 1):
            space.append([i, j])
    return torch.tensor(space, dtype=torch.float)


def death_process_log_likelihood(lp1m1, lp2m1):
    lp1 = (1 - lp1m1.exp()).log()
    lp2 = (1 - lp2m1.exp()).log()

    y = make_y_space(N)
    log_prob_y = torch.lgamma(torch.tensor(N + 1, dtype=torch.float)) \
        - torch.lgamma(y[:, 0] + 1) \
        - torch.lgamma(y[:, 1] + 1) \
        - torch.lgamma(N - y.sum(-1) + 1) \
        + y[:, 0] * lp1 \
        + y[:, 1] * lp2 \
        + (N - y[:, 0]) * lp1m1 \
        + (N - y[:, 0] - y[:, 1]) * lp2m1
    return log_prob_y


def nmc_eig(design, prior_mean, prior_sd, n_samples=1000):
    batch_shape = design.shape[:-1]
    with pyro.plate_stack("plate_stack", (n_samples,) + batch_shape):
        samples = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))

    lp1m1 = -(samples * design[..., 0]).unsqueeze(-1)
    lp2m1 = -(samples * design[..., 1]).unsqueeze(-1)

    likelihoods = death_process_log_likelihood(lp1m1, lp2m1)
    marginal = likelihoods.logsumexp(0, keepdim=True) - math.log(n_samples)
    kls = (likelihoods.exp() * (likelihoods - marginal)).sum(-1)
    return kls.mean(0)


def summed_pce_loss(prior_mean, prior_sd, xi_init):
    def loss(design_placeholder, num_samples=1000, control_variate=0., **kwargs):
        design = pyro.param('xi', xi_init, constraint=constraints.positive)
        batch_shape = design.shape[:-1]
        with pyro.plate_stack("plate_stack", (num_samples,) + batch_shape):
            samples = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))

        lp1m1 = -(samples * design[..., 0]).unsqueeze(-1)
        lp2m1 = -(samples * design[..., 1]).unsqueeze(-1)

        likelihoods = death_process_log_likelihood(lp1m1, lp2m1)
        marginal = likelihoods.logsumexp(0, keepdim=True) - math.log(num_samples)
        eig_estimate = (likelihoods.exp() * (likelihoods - marginal)).sum(-1).mean(0)
        surrogate_loss = eig_estimate.sum()
        return surrogate_loss, eig_estimate

    return loss


def summed_posterior_loss(prior_mean, prior_sd, xi_init):
    def loss(design_placeholder, num_samples=1000, control_variate=0., **kwargs):
        design = pyro.param('xi', xi_init, constraint=constraints.positive)
        batch_shape = design.shape[:-1]
        with pyro.plate_stack("plate_stack", (num_samples,) + batch_shape):
            samples = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))

        lp1m1 = -(samples * design[..., 0]).unsqueeze(-1)
        lp2m1 = -(samples * design[..., 1]).unsqueeze(-1)

        likelihoods = death_process_log_likelihood(lp1m1, lp2m1)
        posterior_mean = pyro.param("q_mean", prior_mean.clone().expand(batch_shape + (66,)))
        posterior_sd = pyro.param("q_sd", prior_sd.clone().expand(batch_shape + (66,)), constraint=constraints.positive)
        q_dist = dist.LogNormal(posterior_mean, posterior_sd)
        qlp = q_dist.log_prob(samples.unsqueeze(-1).expand(num_samples, *batch_shape, 66))
        eig_estimate = -(likelihoods.exp() * qlp).sum(-1).mean(0)
        surrogate_loss = -(likelihoods.exp() * (qlp - control_variate)).sum(-1).mean(0).sum()
        return surrogate_loss, eig_estimate

    return loss


def summed_ace_loss(prior_mean, prior_sd, xi_init):
    def loss(design_placeholder, num_samples=1000, control_variate=0., **kwargs):
        design = pyro.param('xi', xi_init, constraint=constraints.positive)
        batch_shape = design.shape[:-1]
        posterior_mean = pyro.param("q_mean", prior_mean.clone().expand(batch_shape + (66,)))
        posterior_sd = 1e-6 + pyro.param("q_sd", prior_sd.clone().expand(batch_shape + (66,)),
                                         constraint=constraints.positive)
        with pyro.plate_stack("plate_stack", (num_samples,) + batch_shape):
            samples = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))
            contrastive_samples = pyro.sample("b_contrast", dist.LogNormal(posterior_mean, posterior_sd).to_event(1))

        lp1m1 = -(samples * design[..., 0]).unsqueeze(-1)
        lp2m1 = -(samples * design[..., 1]).unsqueeze(-1)
        clp1m1 = -(contrastive_samples * design[..., [0]])
        clp2m1 = -(contrastive_samples * design[..., [1]])

        likelihoods = death_process_log_likelihood(lp1m1, lp2m1)
        contrastive_log_prob = death_process_log_likelihood(clp1m1, clp2m1)
        p_samples = dist.LogNormal(prior_mean, prior_sd).log_prob(samples)
        q_samples = dist.LogNormal(posterior_mean, posterior_sd).log_prob(samples.unsqueeze(-1))
        p_contrastive = dist.LogNormal(prior_mean, prior_sd).log_prob(contrastive_samples)
        q_contrastive = dist.LogNormal(posterior_mean, posterior_sd).log_prob(contrastive_samples)
        sample_terms = likelihoods + p_samples.unsqueeze(-1) - q_samples
        contrastive_terms = contrastive_log_prob + p_contrastive - q_contrastive
        vnmc = contrastive_terms.logsumexp(0, keepdim=True)
        marginals = torch.log(vnmc.exp() + sample_terms.exp()) - math.log(num_samples + 1)
        eig_estimate = (likelihoods.exp() * (likelihoods - marginals)).sum(-1).mean(0)
        surrogate_loss = eig_estimate.sum()
        return surrogate_loss, eig_estimate

    return loss


def model_learn_xi(design_prototype):
    xi = pyro.param('xi')
    batch_shape = design_prototype.shape
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)

        b = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))
        p1 = 1 - torch.exp(-b * xi[0])
        infected1 = pyro.sample("i1", dist.Binomial(N, p1))
        p2 = 1 - torch.exp(-b * xi[1])
        infected2 = pyro.sample("i2", dist.Binomial(N - infected1, p2))
        return infected1, infected2


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
        if time_budget and time.time() - t > time_budget:
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

        # Fix correct loss
        if estimator == 'posterior':
            loss = summed_posterior_loss(prior_mean, prior_sd, xi_init)

        elif estimator == 'pce':
            eig_loss = summed_pce_loss(prior_mean, prior_sd, xi_init)
            loss = neg_loss(eig_loss)

        elif estimator == 'ace':
            eig_loss = summed_ace_loss(prior_mean, prior_sd, xi_init)
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

        # Build heatmap
        grid_points = 100
        b0low = min(0.05, xi_history[:, 0].min())
        b0up = max(3, xi_history[:, 0].max()) + 0.1
        b1low = min(0.05, xi_history[:, 1].min())
        b1up = max(3, xi_history[:, 1].max()) + 0.1
        xi1 = torch.linspace(b0low, b0up, grid_points)
        xi2 = torch.linspace(b1low, b1up, grid_points)
        d1 = xi1.expand(grid_points, grid_points).unsqueeze(-1)
        d2 = xi2.unsqueeze(-1).expand(grid_points, grid_points).unsqueeze(-1)
        d = torch.cat([d1, d2], dim=-1)
        eig_heatmap = nmc_eig(d, prior_mean, prior_sd, n_samples=10000)
        extent = [b0low, b0up, b1low, b1up]
        results['eig_heatmap'] = eig_heatmap
        results['extent'] = extent

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
