import argparse
import datetime
import pickle
import time

import torch
from regression import PosteriorGuide, get_git_revision_hash
from regression_evaluation import make_regression_model

import pyro
import pyro.optim as optim
from pyro.contrib.oed.eig import vnmc_eig


def gp_opt_w_history(loss_fn, num_steps, time_budget, num_parallel, num_acquisition, n, p, device):
    if time_budget is not None:
        num_steps = 100000000000

    est_loss_history = []
    xi_history = []
    t = time.time()
    wall_times = []
    run_times = []
    X = torch.randn((num_parallel, num_acquisition, n, p), device=device)

    y = loss_fn(X)

    for i in range(num_steps):
        pyro.clear_param_store()
        X_acquire = torch.randn((num_parallel, num_acquisition, n, p), device=device)
        y_acquire = loss_fn(X_acquire).detach().clone()
        print('acquired', X_acquire, y_acquire)
        X = torch.cat([X, X_acquire], dim=-3)
        y = torch.cat([y, y_acquire], dim=-1)
        run_times.append(time.time() - t)
        est_loss_history.append(y.min(-1)[0])

        if (time_budget is not None) and (time.time() - t > time_budget):
            break

    final_time = time.time() - t

    # Record the final GP max
    y_star, idx = torch.min(y, dim=-1)
    X_star = X[torch.arange(0, num_parallel, device=device), idx]
    # X_star, y_star = find_gp_max(X, y)
    xi_history.append(X_star.detach().clone())
    wall_times.append(final_time)

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)
    run_times = torch.tensor(run_times)

    return xi_history, est_loss_history, run_times


def main(num_steps, num_samples, experiment_name, seed, num_parallel, start_lr, end_lr,
         device, n, p, scale, time_budget, num_acquisition):
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
    guide = PosteriorGuide(n, p, (num_parallel,)).to(device)

    contrastive_samples = num_samples
    num_outer_steps = 72

    # Fix correct loss
    targets = ["w", "sigma"]
    vnmc_eval = lambda design: -vnmc_eig(model, design, "y", targets, (num_samples, contrastive_samples), num_steps,
                                         guide, optim.Adam({"lr": start_lr}), final_num_samples=(400, 20))

    xi_history, est_loss_history, wall_times = gp_opt_w_history(
        vnmc_eval, num_outer_steps, time_budget, num_parallel, num_acquisition, n, p, device)

    est_eig_history = -est_loss_history

    results = {'git-hash': get_git_revision_hash(), 'seed': seed,
               'xi_history': xi_history.cpu(), 'est_eig_history': est_eig_history.cpu(),
               'wall_times': wall_times.cpu()}

    with open(results_file, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BO-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=1000, type=int)
    parser.add_argument("--time-budget", default=None, type=int)
    parser.add_argument("--num-acquisition", default=10, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--num-parallel", default=10, type=int)
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
         args.time_budget, args.num_acquisition)
