import torch
from torch.distributions import constraints
from torch import nn
import argparse
import math
import subprocess
import datetime
import pickle
import time
from contextlib import ExitStack

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape
from pyro.contrib.oed.differentiable_eig import (
    _differentiable_posterior_loss, differentiable_pce_eig, _differentiable_ace_eig_loss,
        )
from pyro import poutine
from pyro.contrib.util import lexpand, rmv
from pyro.contrib.oed.eig import _eig_from_ape, pce_eig, _ace_eig_loss, nmc_eig, vnmc_eig
from pyro.util import is_bad
from pyro.contrib.autoguide import mean_field_entropy


# TODO read from torch float spec
epsilon = torch.tensor(2**-22)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def invsoftplus(x):
    return (x.exp() - 1).log()


def invsigmoid(y):
    y, y1m = y.clamp(1e-35, 1), (1. - y).clamp(1e-35, 1)
    return y.log() - y1m.log()


def sigmoid(x, top, bottom, ee50, slope):
    return (top - bottom) * torch.sigmoid((x - ee50) * slope) + bottom


def make_docking_model(top_c, bottom_c, ee50_mu, ee50_sigma, slope_mu, slope_sigma, observation_label="y",
                       xi_init=torch.ones(6)):
    def docking_model(design_prototype):
        design = pyro.param("xi", xi_init, constraint=constraints.interval(-75, -1e-6)).expand(design_prototype.shape)
        if is_bad(design):
            raise ArithmeticError("bad design, contains nan or inf")
        batch_shape = design.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            top = pyro.sample("top", dist.Dirichlet(top_c)).select(-1, 0).unsqueeze(-1)
            bottom = pyro.sample("bottom", dist.Dirichlet(bottom_c)).select(-1, 0).unsqueeze(-1)
            ee50 = pyro.sample("ee50", dist.Normal(ee50_mu, ee50_sigma)).unsqueeze(-1)
            slope = pyro.sample("slope", dist.Normal(slope_mu, slope_sigma)).unsqueeze(-1)
            hit_rate = sigmoid(design, top, bottom, ee50, slope)
            y = pyro.sample(observation_label, dist.Bernoulli(hit_rate).to_event(1))
            return y

    return docking_model


class TensorLinear(nn.Module):

    __constants__ = ['bias']

    def __init__(self, *shape, bias=True):
        super(TensorLinear, self).__init__()
        self.in_features = shape[-2]
        self.out_features = shape[-1]
        self.batch_dims = shape[:-2]
        self.weight = nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features, self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(*self.batch_dims, self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return rmv(self.weight, input) + self.bias


class PosteriorGuide(nn.Module):
    def __init__(self, y_dim, batching):
        super(PosteriorGuide, self).__init__()
        n_hidden = 64
        self.linear1 = TensorLinear(*batching, y_dim, n_hidden)
        self.linear2 = TensorLinear(*batching, n_hidden, n_hidden)
        self.output_layer = TensorLinear(*batching, n_hidden, 2 + 2 + 2 + 2)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def set_prior(self, rho_concentration, alpha_concentration, slope_mu, slope_sigma):
        self.prior_rho_concentration = rho_concentration
        self.prior_alpha_concentration = alpha_concentration
        self.prior_slope_mu = slope_mu
        self.prior_slope_sigma = slope_sigma

    def forward(self, y_dict, design_prototype, observation_labels, target_labels):
        y = y_dict["y"] - .5
        x = self.relu(self.linear1(y))
        x = self.relu(self.linear2(x))
        final = self.output_layer(x)

        top_c = self.softplus(final[..., 0:2])
        bottom_c = self.softplus(final[..., 2:4])
        ee50_mu = final[..., 4]
        ee50_sigma = self.softplus(final[..., 5])
        slope_mu = final[..., 6]
        slope_sigma = self.softplus(final[..., 7])

        pyro.module("posterior_guide", self)

        batch_shape = design_prototype.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            pyro.sample("top", dist.Dirichlet(top_c))
            pyro.sample("bottom", dist.Dirichlet(bottom_c))
            pyro.sample("ee50", dist.Normal(ee50_mu, ee50_sigma))
            pyro.sample("slope", dist.Normal(slope_mu, slope_sigma))


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))
    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim, lower, upper, n_high_acc, h_freq):

    params = None
    est_loss_history = []
    lower_history = []
    upper_history = []
    xi_history = []
    baseline = 0.
    t = time.time()
    wall_times = []
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples, evaluation=True, control_variate=baseline)
        baseline = -loss.detach()
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
        agg_loss.backward(retain_graph=True)
        est_loss_history.append(loss.detach())
        optim(params)
        optim.step()
        print(pyro.param("xi").squeeze())
        print('eig', baseline.squeeze())

        if step % h_freq == 0:
            low = lower(design, n_high_acc, evaluation=True)
            up = upper(design, n_high_acc, evaluation=True)
            if isinstance(low, tuple):
                low = low[1]
            if isinstance(up, tuple):
                up = up[1]
            lower_history.append(low.detach())
            upper_history.append(up.detach())
            wall_times.append(time.time() - t)

    xi_history.append(pyro.param('xi').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)
    lower_history = torch.stack(lower_history)
    upper_history = torch.stack(upper_history)
    wall_times = torch.tensor(wall_times)

    return xi_history, est_loss_history, lower_history, upper_history, wall_times


def main(num_steps, high_acc_freq, num_samples, experiment_name, estimators, seed, num_parallel, start_lr, end_lr,
         device):
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

        D = 100
        xi_init = lexpand(torch.linspace(-70., -10, D, device=device), num_parallel)
        # Change the prior distribution here
        # prior params
        top_prior_concentration = torch.tensor([25., 75.], device=device)
        bottom_prior_concentration = torch.tensor([4., 96.], device=device)
        ee50_prior_mu, ee50_prior_sd = torch.tensor(-50., device=device), torch.tensor(15., device=device)
        slope_prior_mu, slope_prior_sd = torch.tensor(-0.15, device=device), torch.tensor(0.1, device=device)

        model_learn_xi = make_docking_model(
            top_prior_concentration, bottom_prior_concentration, ee50_prior_mu, ee50_prior_sd, slope_prior_mu,
            slope_prior_sd, xi_init=xi_init)

        contrastive_samples = num_samples

        # Fix correct loss
        targets = ["top", "bottom", "ee50", "slope"]
        print("Prior entropy", mean_field_entropy(model_learn_xi, [torch.zeros(num_parallel, D, device=device)],
                                                  whitelist=targets))
        if estimator == 'posterior':
            m_final = 20
            guide = PosteriorGuide(D, (num_parallel,)).to(device)
            loss = _differentiable_posterior_loss(model_learn_xi, guide, ["y"], targets)
            high_acc = loss
            upper_loss = lambda d, N, **kwargs: vnmc_eig(model_learn_xi, d, "y", targets, (N, int(math.sqrt(N))), 0, guide, None)

        elif estimator == 'pce':
            m_final = 40
            eig_loss = lambda d, N, **kwargs: differentiable_pce_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=targets,
                N=N, M=contrastive_samples, **kwargs)
            loss = neg_loss(eig_loss)
            high_acc = lambda d, N, **kwargs: pce_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=targets,
                N=N, M=int(math.sqrt(N)), **kwargs)
            upper_loss = lambda d, N, **kwargs: nmc_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=targets,
                N=N, M=int(math.sqrt(N)), **kwargs)

        elif estimator == 'ace':
            m_final = 20
            guide = PosteriorGuide(D, (num_parallel,)).to(device)
            eig_loss = _differentiable_ace_eig_loss(model_learn_xi, guide, contrastive_samples, ["y"],
                                                    ["top", "bottom", "ee50", "slope"])
            loss = neg_loss(eig_loss)
            high_acc = _ace_eig_loss(model_learn_xi, guide, m_final, "y", targets)
            upper_loss = lambda d, N, **kwargs: vnmc_eig(model_learn_xi, d, "y", targets, (N, int(math.sqrt(N))), 0, guide, None)

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr / start_lr) ** (1 / num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(num_parallel, D, device=device)  # this is annoying, code needs refactor

        xi_history, est_loss_history, lower_history, upper_history, wall_times = opt_eig_loss_w_history(
            design_prototype, loss, num_samples=num_samples, num_steps=num_steps, optim=scheduler, lower=high_acc,
            upper=upper_loss, n_high_acc=m_final**2, h_freq=high_acc_freq)

        if estimator == 'posterior':
            est_eig_history = _eig_from_ape(model_learn_xi, design_prototype, targets, est_loss_history, True, {})
            lower_history = _eig_from_ape(model_learn_xi, design_prototype, targets, lower_history, True, {})

        elif estimator in ['pce', 'pce-proposal', 'ace']:
            est_eig_history = -est_loss_history
        else:
            est_eig_history = est_loss_history

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history.cpu(), 'est_eig_history': est_eig_history.cpu(),
                   'lower_history': lower_history.cpu(), 'upper_history': upper_history.cpu(),
                   'wall_times': wall_times.cpu()}

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=500000, type=int)
    parser.add_argument("--high-acc-freq", default=50000, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.001, type=float)
    parser.add_argument("--end-lr", default=0.001, type=float)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()
    main(args.num_steps, args.high_acc_freq, args.num_samples, args.name, args.estimator, args.seed, args.num_parallel,
         args.start_lr, args.end_lr, args.device)
