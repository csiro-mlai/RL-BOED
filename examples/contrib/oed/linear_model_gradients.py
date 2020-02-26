import argparse
import datetime
from contextlib import ExitStack
import math
import subprocess
import pickle

import torch

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.contrib.oed.eig import _eig_from_ape, pce_eig, _ace_eig_loss
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss
from pyro.contrib.oed.util import linear_model_ground_truth
from pyro.contrib.util import rmv, iter_plates_to_shape, lexpand, rvv, rexpand


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


N = 2
prior_scale_tril = torch.tensor([[10., 0.], [0., 2.]])
xi_init = (math.pi/3) * torch.ones(N,)


def model_learn_xi(design_prototype):
    thetas = pyro.param('xi', xi_init)
    xi = lexpand(torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=-1), 1)
    batch_shape = design_prototype.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)

        x = pyro.sample("x", dist.MultivariateNormal(torch.zeros(2), scale_tril=prior_scale_tril))
        prediction_mean = rmv(xi, x)
        return pyro.sample("y", dist.Normal(prediction_mean, torch.tensor(1.)).independent(1))


def model_fix_xi(design):
    batch_shape = design.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)

        x = pyro.sample("x", dist.MultivariateNormal(lexpand(torch.zeros(2), *batch_shape),
                                                     scale_tril=lexpand(prior_scale_tril, *batch_shape)))
        prediction_mean = rmv(design, x)
        return pyro.sample("y", dist.Normal(prediction_mean, torch.tensor(1.)).independent(1))


model_fix_xi.w_sds = {"x": prior_scale_tril.diagonal()}
model_fix_xi.obs_sd = torch.tensor(1.)


def make_posterior_guide(d):
    def posterior_guide(y_dict, design, observation_labels, target_labels):

        y = torch.cat(list(y_dict.values()), dim=-1)
        A = pyro.param("A", torch.zeros(d, 2, N))
        scale_tril = pyro.param("scale_tril", lexpand(prior_scale_tril, d),
                                constraint=torch.distributions.constraints.lower_cholesky)
        mu = rmv(A, y)
        pyro.sample("x", dist.MultivariateNormal(mu, scale_tril=scale_tril))
    return posterior_guide


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))
    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim):

    params = None
    est_loss_history = []
    xi_history = []
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
        est_loss_history.append(loss)
        xi_history.append(pyro.param('xi').detach().clone())
        optim(params)
        optim.step()

    xi_history.append(pyro.param('xi').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)

    return xi_history, est_loss_history


def main(num_steps, experiment_name, estimators, seed, start_lr, end_lr):
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

        # Fix correct loss
        if estimator == 'posterior':
            guide = make_posterior_guide(1)
            loss = _differentiable_posterior_loss(model_learn_xi, guide, "y", "x")

        elif estimator == 'nce':
            eig_loss = lambda d, N, **kwargs: pce_eig(model=model_learn_xi, design=d, observation_labels="y",
                                                      target_labels="x", N=N, M=10, **kwargs)
            loss = neg_loss(eig_loss)

        elif estimator == 'ace':
            guide = make_posterior_guide(1)
            eig_loss = _ace_eig_loss(model_learn_xi, guide, 10, "y", "x")
            loss = neg_loss(eig_loss)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr/start_lr)**(1/num_steps)
        # optimizer = optim.Adam({"lr": start_lr})
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(1, N, 2)  # this is annoying, code needs refactor

        xi_history, est_loss_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=10,
                                                             num_steps=num_steps, optim=scheduler)

        if estimator == 'posterior':
            est_eig_history = _eig_from_ape(model_learn_xi, design_prototype, "x", est_loss_history, True, {})
        else:
            est_eig_history = -est_loss_history
        eig_history = linear_model_ground_truth(
            model_fix_xi, torch.stack([torch.sin(xi_history), torch.cos(xi_history)], dim=-1), "y", "x")

        # Build heatmap
        grid_points = 100
        b0low = min(0, xi_history[:, 0].min()) - 0.1
        b0up = max(math.pi, xi_history[:, 0].max()) + 0.1
        b1low = min(0, xi_history[:, 1].min()) - 0.1
        b1up = max(math.pi, xi_history[:, 1].max()) + 0.1
        theta1 = torch.linspace(b0low, b0up, grid_points)
        theta2 = torch.linspace(b1low, b1up, grid_points)
        d1 = torch.stack([torch.sin(theta1), torch.cos(theta1)], dim=-1).unsqueeze(-2).unsqueeze(1).expand(
            grid_points, grid_points, 1, 2)
        d2 = lexpand(torch.stack([torch.sin(theta2), torch.cos(theta2)], dim=-1).unsqueeze(-2), grid_points)
        d = torch.cat([d1, d2], dim=-2)
        eig_heatmap = linear_model_ground_truth(model_fix_xi, d, "y", "x")
        extent = [b0low, b0up, b1low, b1up]

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'prior_scale_tril': prior_scale_tril, 'xi_history': xi_history, 'est_eig_history': est_eig_history,
                   'eig_history': eig_history, 'eig_heatmap': eig_heatmap, 'extent': extent}

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=2000, type=int)
    # parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.1, type=float)
    parser.add_argument("--end-lr", default=0.001, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.name, args.estimator, args.seed, args.start_lr, args.end_lr)
