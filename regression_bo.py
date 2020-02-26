import torch
from torch.distributions import constraints
from torch import nn
import argparse
import math
import subprocess
import datetime
import pickle
import time
from torch.distributions import transform_to

import pyro
import pyro.optim as optim
import pyro.contrib.gp as gp
from pyro.contrib.oed.eig import vnmc_eig
from pyro.contrib.util import rmv
from pyro.util import is_bad

from regression import PosteriorGuide, neg_loss, get_git_revision_hash
from regression_evaluation import make_regression_model




def gp_opt_w_history(loss_fn, num_steps, time_budget, num_parallel, num_acquisition, lengthscale, n, p, device):

    if time_budget is not None:
        num_steps = 100000000000

    est_loss_history = []
    xi_history = []
    t = time.time()
    wall_times = []
    run_times = []
    X = -1 + 2 * torch.rand((num_parallel, num_acquisition, n * p), device=device)

    y = loss_fn(X.reshape(X.shape[:-1] + (n, p)))

    # GPBO
    y = y.detach().clone()
    print('initial y', y)
    kernel = gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(lengthscale, device=device),
                                 variance=torch.tensor(10., device=device))
    constraint = torch.distributions.constraints.interval(-1., 1.)
    noise = torch.tensor(0.5, device=device).pow(2)

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
        print('Kff', Kff)
        Kff += noise * torch.eye(Kff.shape[-1], device=device)
        Lff = Kff.cholesky(upper=False)
        Xinit = -1 + 2*torch.rand((num_parallel, nacq, n * p), device=device)
        unconstrained_Xnew = transform_to(constraint).inv(Xinit).detach().clone().requires_grad_(True)
        minimizer = torch.optim.LBFGS([unconstrained_Xnew], max_eval=20)

        def gp_ucb1():
            minimizer.zero_grad()
            Xnew = transform_to(constraint)(unconstrained_Xnew)
            mean, var = gp_conditional(Lff, Xnew, X, y)
            ucb = (mean - sigma * var.clamp(min=0.).sqrt())
            ucb[is_bad(ucb)] = 0.
            loss = ucb.sum()
            torch.autograd.backward(unconstrained_Xnew,
                                    torch.autograd.grad(loss, unconstrained_Xnew, retain_graph=True))
            return loss

        minimizer.step(gp_ucb1)
        X_acquire = transform_to(constraint)(unconstrained_Xnew).detach().clone()
        y_expected, _ = gp_conditional(Lff, X_acquire, X, y)

        return X_acquire, y_expected

    for i in range(num_steps):
        pyro.clear_param_store()
        X_acquire, _ = acquire(X, y, 2, num_acquisition)
        y_acquire = loss_fn(X_acquire.reshape(X_acquire.shape[:-1] + (n, p))).detach().clone()
        print('acquired', X_acquire, y_acquire)
        X = torch.cat([X, X_acquire], dim=-2)
        y = torch.cat([y, y_acquire], dim=-1)
        run_times.append(time.time() - t)
        est_loss_history.append(y.max(-1)[0])

        if time_budget and time.time() - t > time_budget:
            break

    final_time = time.time() - t

    # Record the final GP max
    y_star, idx = torch.min(y, dim=-1)
    X_star = X[torch.arange(0, num_parallel, device=device), idx]
    xi_history.append(X_star.detach().clone())
    wall_times.append(final_time)

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)
    run_times = torch.tensor(run_times)

    return xi_history, est_loss_history, run_times


def main(num_steps, num_samples, experiment_name, seed, num_parallel, start_lr, end_lr,
         device, n, p, scale, time_budget, num_acquisition, gp_lengthscale):
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

    # Change the prior distribution here
    # prior params
    w_prior_loc = torch.zeros(p, device=device)
    w_prior_scale = scale * torch.ones(p, device=device)
    sigma_prior_scale = scale * torch.tensor(1., device=device)

    model = make_regression_model(
        w_prior_loc, w_prior_scale, sigma_prior_scale)
    guide = PosteriorGuide(n, p, (num_parallel, )).to(device)

    contrastive_samples = num_samples

    # Fix correct loss
    targets = ["w", "sigma"]
    vnmc_eval = lambda design: -vnmc_eig(model, design, "y", targets, (num_samples, contrastive_samples), num_steps,
                                         guide, optim.Adam({"lr": start_lr}), final_num_samples=(400, 20))

    xi_history, est_loss_history, wall_times = gp_opt_w_history(
        vnmc_eval, None, time_budget, num_parallel, num_acquisition, gp_lengthscale, n, p, device)

    est_eig_history = -est_loss_history

    results = {'git-hash': get_git_revision_hash(), 'seed': seed,
               'xi_history': xi_history.cpu(), 'est_eig_history': est_eig_history.cpu(),
               'wall_times': wall_times.cpu()}

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BO-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=5000, type=int)
    parser.add_argument("--time-budget", default=1200, type=int)
    parser.add_argument("--num-acquisition", default=10, type=int)
    parser.add_argument("--lengthscale", default=5., type=float)
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--num-parallel", default=1, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.001, type=float)
    parser.add_argument("--end-lr", default=0.001, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("-n", default=20, type=int)
    parser.add_argument("-p", default=20, type=int)
    parser.add_argument("--scale", default=1., type=float)
    args = parser.parse_args()
    main(args.num_steps, args.num_samples, args.name, args.seed, args.num_parallel,
         args.start_lr, args.end_lr, args.device, args.n, args.p, args.scale,
         args.time_budget, args.num_acquisition, args.lengthscale)
