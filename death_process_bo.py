import argparse
import datetime
import math
import subprocess
import pickle
from functools import lru_cache
import time

import torch
from torch.distributions import constraints
from torch.distributions import transform_to

import pyro
import pyro.distributions as dist
import pyro.contrib.gp as gp
from pyro.contrib.util import rmv
from pyro.util import is_bad

from death_process_rb import nmc_eig


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


N = 10
design_dim = 2
prior_mean = torch.tensor(0.)
prior_sd = torch.tensor(1.0)


@lru_cache(5)
def make_y_space(n):
    space = []
    for i in range(n+1):
        for j in range(n-i+1):
            space.append([i, j])
    return torch.tensor(space, dtype=torch.float)


def summed_posterior_loss(prior_mean, prior_sd, num_samples):
    def loss(design):
        batch_shape = design.shape[:-1]
        with pyro.plate_stack("plate_stack", (num_samples,) + batch_shape):
            samples = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))

        lp1m1 = -(samples * design[..., 0]).unsqueeze(-1)
        lp2m1 = -(samples * design[..., 1]).unsqueeze(-1)

        def log_prob(lp1m1, lp2m1):
            lp1 = (1 - lp1m1.exp()).log()
            lp2 = (1 - lp2m1.exp()).log()

            y = make_y_space(N)
            log_prob_y = torch.lgamma(torch.tensor(N + 1, dtype=torch.float)) - torch.lgamma(y[:, 0] + 1) - torch.lgamma(y[:, 1] + 1) \
                         - torch.lgamma(N - y.sum(-1) + 1) + y[:, 0] * lp1 + y[:, 1] * lp2 + (N - y[:, 0]) * lp1m1 \
                         + (N - y[:, 0] - y[:, 1]) * lp2m1
            return log_prob_y

        likelihoods = log_prob(lp1m1, lp2m1)
        posterior_mean = pyro.param("q_mean", prior_mean.clone().expand(batch_shape + (66,)))
        posterior_sd = pyro.param("q_sd", prior_sd.clone().expand(batch_shape + (66,)), constraint=constraints.positive)
        q_dist = dist.LogNormal(posterior_mean, posterior_sd)
        qlp = q_dist.log_prob(samples.unsqueeze(-1).expand(num_samples, *batch_shape, 66))
        eig_estimate = -(likelihoods.exp() * qlp).sum(-1).mean(0)
        return eig_estimate
    return loss


def summed_ace_loss(prior_mean, prior_sd, num_samples):
    def loss(design):
        batch_shape = design.shape[:-1]
        posterior_mean = pyro.param("q_mean", prior_mean.clone().expand(batch_shape + (66,)))
        posterior_sd = 1e-6 + pyro.param("q_sd", prior_sd.clone().expand(batch_shape + (66,)), constraint=constraints.positive)
        with pyro.plate_stack("plate_stack", (num_samples,) + batch_shape):
            samples = pyro.sample("b", dist.LogNormal(prior_mean, prior_sd))
            contrastive_samples = pyro.sample("b_contrast", dist.LogNormal(posterior_mean, posterior_sd).to_event(1))

        lp1m1 = -(samples * design[..., 0]).unsqueeze(-1)
        lp2m1 = -(samples * design[..., 1]).unsqueeze(-1)
        clp1m1 = -(contrastive_samples * design[..., [0]])
        clp2m1 = -(contrastive_samples * design[..., [1]])

        def log_prob(lp1m1, lp2m1):
            lp1 = (1 - lp1m1.exp()).log()
            lp2 = (1 - lp2m1.exp()).log()

            y = make_y_space(N)
            log_prob_y = torch.lgamma(torch.tensor(N + 1, dtype=torch.float)) - torch.lgamma(y[:, 0] + 1) - torch.lgamma(y[:, 1] + 1) \
                         - torch.lgamma(N - y.sum(-1) + 1) + y[:, 0] * lp1 + y[:, 1] * lp2 + (N - y[:, 0]) * lp1m1 \
                         + (N - y[:, 0] - y[:, 1]) * lp2m1
            return log_prob_y

        likelihoods = log_prob(lp1m1, lp2m1)
        contrastive_log_prob = log_prob(clp1m1, clp2m1)
        p_samples = dist.LogNormal(prior_mean, prior_sd).log_prob(samples)
        q_samples = dist.LogNormal(posterior_mean, posterior_sd).log_prob(samples.unsqueeze(-1))
        p_contrastive = dist.LogNormal(prior_mean, prior_sd).log_prob(contrastive_samples)
        q_contrastive = dist.LogNormal(posterior_mean, posterior_sd).log_prob(contrastive_samples)
        sample_terms = likelihoods + p_samples.unsqueeze(-1) - q_samples
        contrastive_terms = contrastive_log_prob + p_contrastive - q_contrastive
        vnmc = contrastive_terms.logsumexp(0, keepdim=True)
        marginals = torch.log(vnmc.exp() + sample_terms.exp()) - math.log(num_samples + 1)
        eig_estimate = (likelihoods.exp() * (likelihoods - marginals)).sum(-1).mean(0)
        return eig_estimate
    return loss


def gp_opt_w_history(loss_fn, num_steps, time_budget, num_parallel, num_acquisition, lengthscale):

    if time_budget is not None:
        num_steps = 100000000000

    est_loss_history = []
    xi_history = []
    t = time.time()
    wall_times = []
    run_times = []
    X = .01 + 4.99 * torch.rand((num_parallel, num_acquisition, design_dim))

    y = loss_fn(X)

    # GPBO
    y = y.detach().clone()
    kernel = gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(lengthscale),
                                 variance=torch.tensor(1.))
    constraint = torch.distributions.constraints.interval(1e-2, 5.)
    noise = torch.tensor(0.5).pow(2)

    def gp_conditional(Lff, Xnew, X, y):
        KXXnew = kernel(X, Xnew)
        LiK = torch.triangular_solve(KXXnew, Lff, upper=False)[0]
        Liy = torch.triangular_solve(y.unsqueeze(-1), Lff, upper=False)[0]
        mean = rmv(LiK.transpose(-1, -2), Liy.squeeze(-1))
        KXnewXnew = kernel(Xnew)
        var = (KXnewXnew - LiK.transpose(-1, -2).matmul(LiK)).diagonal(dim1=-2, dim2=-1)
        return mean, var

    def acquire(X, y, sigma, nacq):
        Kff = kernel(X)
        Kff += noise * torch.eye(Kff.shape[-1])
        Lff = Kff.cholesky(upper=False)
        Xinit = .01 + 4.99 * torch.rand((num_parallel, nacq, design_dim))
        unconstrained_Xnew = transform_to(constraint).inv(Xinit).detach().clone().requires_grad_(True)
        minimizer = torch.optim.LBFGS([unconstrained_Xnew], max_eval=20)

        def gp_ucb1():
            minimizer.zero_grad()
            Xnew = transform_to(constraint)(unconstrained_Xnew)
            mean, var = gp_conditional(Lff, Xnew, X, y)
            ucb = -(mean + sigma * var.clamp(min=0.).sqrt())
            ucb[is_bad(ucb)] = 0.
            loss = ucb.sum()
            torch.autograd.backward(unconstrained_Xnew,
                                    torch.autograd.grad(loss, unconstrained_Xnew))
            return loss

        minimizer.step(gp_ucb1)
        X_acquire = transform_to(constraint)(unconstrained_Xnew).detach().clone()
        y_expected, _ = gp_conditional(Lff, X_acquire, X, y)

        return X_acquire, y_expected

    def find_gp_max(X, y, n_tries=100):
        X_star = torch.zeros(num_parallel, 1, design_dim)
        y_star = torch.zeros(num_parallel, 1)
        for j in range(n_tries):  # Cannot parallelize this because sometimes LBFGS goes bad across a whole batch
            X_star_new, y_star_new = acquire(X, y, 0, 1)
            y_star_new[is_bad(y_star_new)] = 0.
            mask = y_star_new > y_star
            y_star[mask, ...] = y_star_new[mask, ...]
            X_star[mask, ...] = X_star_new[mask, ...]

        return X_star.squeeze(), y_star.squeeze()

    for i in range(num_steps):
        X_acquire, _ = acquire(X, y, 2, num_acquisition)
        y_acquire = loss_fn(X_acquire).detach().clone()
        X = torch.cat([X, X_acquire], dim=-2)
        y = torch.cat([y, y_acquire], dim=-1)
        run_times.append(time.time() - t)

        if time_budget and time.time() - t > time_budget:
            break

    final_time = time.time() - t

    for i in range(1, len(run_times)+1):

        if i % 10 == 0:
            s = num_acquisition * i
            X_star, y_star = find_gp_max(X[:, :s, :], y[:, :s])
            print(X_star)
            est_loss_history.append(y_star)
            xi_history.append(X_star.detach().clone())
            wall_times.append(run_times[i-1])

    # Record the final GP max
    X_star, y_star = find_gp_max(X, y)
    xi_history.append(X_star.detach().clone())
    wall_times.append(final_time)

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)
    wall_times = torch.tensor(wall_times)

    return xi_history, est_loss_history, wall_times


def main(experiment_name, seed, estimator, num_parallel, num_steps, time_budget, num_acquisition, num_samples):
    output_dir = "./run_outputs/gradinfo/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.pickle'

    pyro.clear_param_store()
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = int(torch.rand(tuple()) * 2 ** 30)
        pyro.set_rng_seed(seed)

    prior_entropy = dist.Normal(prior_mean, prior_sd).entropy()

    # Fix correct loss
    if estimator == 'pce':
        loss = lambda X: nmc_eig(X, prior_mean, prior_sd, n_samples=num_samples)

    elif estimator == 'posterior':

        def loss(design):
            pyro.clear_param_store()
            optim = pyro.optim.Adam({"lr": 0.01})
            loss_to_opt = summed_posterior_loss(prior_mean, prior_sd, num_samples)
            params = None
            for i in range(100):
                if params is not None:
                    pyro.infer.util.zero_grads(params)
                with pyro.poutine.trace(param_only=True) as param_capture:
                    l = loss_to_opt(design)
                params = set(site["value"].unconstrained()
                             for site in param_capture.trace.nodes.values())
                l.sum().backward()
                optim(params)
            return prior_entropy - l.detach()

    elif estimator == 'ace':

        def loss(design):
            pyro.clear_param_store()
            optim = pyro.optim.Adam({"lr": 0.0025})
            loss_to_opt = summed_ace_loss(prior_mean, prior_sd, num_samples)
            params = None
            for i in range(49):
                if params is not None:
                    pyro.infer.util.zero_grads(params)
                with pyro.poutine.trace(param_only=True) as param_capture:
                    l = loss_to_opt(design)
                params = set(site["value"].unconstrained()
                             for site in param_capture.trace.nodes.values())
                (-l.sum()).backward()
                optim(params)
            loss_to_opt = summed_ace_loss(prior_mean, prior_sd, num_samples * 10)
            l = loss_to_opt(design)
            print(l)
            l[is_bad(l)] = 0.
            return l.detach()

    xi_history, est_loss_history, wall_times = gp_opt_w_history(
        loss, num_steps, time_budget, num_parallel, num_acquisition, .25)

    eig_history = []
    for i in range(xi_history.shape[0] - 1):
        eig_history.append(nmc_eig(xi_history[i, ...], prior_mean, prior_sd, n_samples=20000))
    eig_history.append(nmc_eig(xi_history[-1, ...], prior_mean, prior_sd, n_samples=200000))
    eig_history = torch.stack(eig_history)

    results = {'estimator': 'bo-'+estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
               'xi_history': xi_history, 'est_eig_history': est_loss_history, 'eig_history': eig_history,
               'wall_times': wall_times, 'num_samples': num_samples, 'num_acquisition': num_acquisition,
               'time_budget': time_budget, 'num_steps': num_steps}

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BO design optimization for Death Process")
    parser.add_argument("--num-steps", default=200, type=int)
    parser.add_argument("--estimator", default="pce", type=str)
    parser.add_argument("--time-budget", default=None, type=float)
    parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--num-acquisition", default=1, type=int)
    parser.add_argument("--num-samples", default=2000, type=int)
    args = parser.parse_args()
    main(args.name, args.seed, args.estimator, args.num_parallel, args.num_steps, args.time_budget, args.num_acquisition, args.num_samples)
