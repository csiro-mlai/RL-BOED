import torch
from torch.distributions import constraints
from torch import nn
import argparse
import math
import subprocess
import datetime
import pickle
import logging
from contextlib import ExitStack

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape, rexpand, rmv
from pyro.contrib.oed.differentiable_eig import (
    _differentiable_posterior_loss, differentiable_pce_eig, _differentiable_ace_eig_loss,
    differentiable_nce_proposal_eig, _saddle_marginal_loss
        )
from pyro import poutine
from pyro.contrib.oed.eig import _eig_from_ape
from pyro.util import is_bad


# TODO read from torch float spec
epsilon = torch.tensor(2**-22)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma, observation_sd, observation_label="y",
                   xi_init=torch.ones(6)):
    def ces_model(design_prototype):
        design = pyro.param("xi", xi_init, constraint=constraints.interval(1e-6, 100)).expand(design_prototype.shape)
        if is_bad(design):
            raise ArithmeticError("bad design, contains nan or inf")
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            rho_shape = batch_shape + (rho_concentration.shape[-1],)
            rho = 0.01 + 0.99 * pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape))).select(-1, 0)
            alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
            alpha = pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
            slope = pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape), slope_sigma.expand(batch_shape)))
            rho, slope = rexpand(rho, design.shape[-2]), rexpand(slope, design.shape[-2])
            d1, d2 = design[..., 0:3], design[..., 3:6]
            U1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
            U2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1./rho)
            mean = slope * (U1rho - U2rho)
            sd = slope * observation_sd * (1 + torch.norm(d1 - d2, dim=-1, p=2))
            emission_dist = dist.CensoredSigmoidNormal(mean, sd, 1 - epsilon, epsilon).to_event(1)
            y = pyro.sample(observation_label, emission_dist)
            return y

    return ces_model


# def proposal(design):
#
#         batch_shape = design.shape[:-2]
#         with ExitStack() as stack:
#             for plate in iter_plates_to_shape(batch_shape):
#                 stack.enter_context(plate)
#             emission_dist = dist.CensoredSigmoidNormal(torch.tensor([0.]), torch.tensor([50.]), 1 - epsilon, epsilon).to_event(1)
#             pyro.sample("y", emission_dist)


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
    def __init__(self, batch_shape):
        super(PosteriorGuide, self).__init__()
        n_hidden = 64
        self.linear1 = TensorLinear(*batch_shape, 1, n_hidden)
        self.linear2 = TensorLinear(*batch_shape, n_hidden, n_hidden)
        self.output_layer = TensorLinear(*batch_shape, n_hidden, 2 + 3 + 1 + 1)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def set_prior(self, rho_concentration, alpha_concentration, slope_mu, slope_sigma):
        self.prior_rho_concentration = rho_concentration
        self.prior_alpha_concentration = alpha_concentration
        self.prior_slope_mu = slope_mu
        self.prior_slope_sigma = slope_sigma

    def forward(self, y_dict, design_prototype, observation_labels, target_labels):
        y = y_dict["y"]
        y, y1m = y.clamp(1e-35, 1), (1. - y).clamp(1e-35, 1)
        s = y.log() - y1m.log()
        x = self.relu(self.linear1(s))
        x = self.relu(self.linear2(x))
        final = self.output_layer(x)

        rho_concentration = self.softplus(final[..., 0:2]) + self.prior_rho_concentration
        alpha_concentration = self.softplus(final[..., 2:5]) + self.prior_alpha_concentration
        slope_mu = self.prior_slope_mu + 3 * 2 * (-1 + 2 * torch.sigmoid(final[..., 5]))
        slope_sigma = self.prior_slope_sigma * (1e-6 + 2 * torch.sigmoid(final[..., 6]))

        logging.debug("rho_concentration {} {} alpha concentration {} {}".format(
            rho_concentration.min().item(), rho_concentration.max().item(),
            alpha_concentration.min().item(), alpha_concentration.max().item()))
        logging.debug("slope mu {} {} sigma {} {}".format(
            slope_mu.min().item(), slope_mu.max().item(), slope_sigma.min().item(), slope_sigma.max().item()))

        pyro.module("posterior_guide", self)

        batch_shape = design_prototype.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            rho_shape = batch_shape + (rho_concentration.shape[-1],)
            pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape)))
            alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
            pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
            pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape), slope_sigma.expand(batch_shape)))


class LinearPosteriorGuide(nn.Module):
    def __init__(self, batch_shape):
        super(LinearPosteriorGuide, self).__init__()
        pyro.param("q_param", torch.zeros(*batch_shape, 4, 2 + 3 + 1 + 1))
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def set_prior(self, rho_concentration, alpha_concentration, slope_mu, slope_sigma):
        self.prior_rho_concentration = rho_concentration
        self.prior_alpha_concentration = alpha_concentration
        self.prior_slope_mu = slope_mu
        self.prior_slope_sigma = slope_sigma
        pyro.get_param_store().replace_param("q_param", torch.zeros(pyro.param("q_param").shape), pyro.param("q_param"))

    def forward(self, y_dict, design_prototype, observation_labels, target_labels):
        param = pyro.param("q_param")
        y = y_dict["y"]
        y, y1m = y.clamp(1e-35, 1), (1. - y).clamp(1e-35, 1)
        s = y.log() - y1m.log()
        final = param[..., 0, :] + param[..., 1, :] * s + param[..., 2, :] * (1e-6 + s).abs().log() + \
                param[..., 3, :] * (s > 0.).float()
        
        rho_concentration = 1e-6 + self.relu(self.prior_rho_concentration + final[..., 0:2])
        alpha_concentration = 1e-6 + self.relu(self.prior_alpha_concentration + final[..., 2:5])
        slope_mu = self.prior_slope_mu + 3 * 2 * (-1 + 2 * torch.sigmoid(final[..., 5]))
        slope_sigma = self.prior_slope_sigma * (1e-6 + 2 * torch.sigmoid(final[..., 6]))

        logging.debug("rho_concentration {} {} alpha concentration {} {}".format(
            rho_concentration.min().item(), rho_concentration.max().item(),
            alpha_concentration.min().item(), alpha_concentration.max().item()))
        logging.debug("slope mu {} {} sigma {} {}".format(
            slope_mu.min().item(), slope_mu.max().item(), slope_sigma.min().item(), slope_sigma.max().item()))


        batch_shape = design_prototype.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            rho_shape = batch_shape + (rho_concentration.shape[-1],)
            pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape)))
            alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
            pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
            pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape), slope_sigma.expand(batch_shape)))


def marginal_guide(mu_init, log_sigma_init, shape, label):
    def guide(design, observation_labels, target_labels):
        mu = pyro.param("marginal_mu", mu_init * torch.ones(*shape))
        log_sigma = pyro.param("marginal_log_sigma", log_sigma_init * torch.ones(*shape))
        ends = pyro.param("marginal_ends", 1./3 * torch.ones(*shape, 3),
                          constraint=torch.distributions.constraints.simplex)
        response_dist = dist.CensoredSigmoidNormalEnds(
            loc=mu, scale=torch.exp(log_sigma), upper_lim=1. - epsilon, lower_lim=epsilon,
            p0=ends[..., 0], p1=ends[..., 1], p2=ends[..., 2]
        ).to_event(1)
        #print('mu', mu, 'log_sigma', log_sigma, 'ends', ends)
        pyro.sample(label, response_dist)
    return guide


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))
    return new_loss


def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim):

    params = None
    est_loss_history = []
    xi_history = []
    baseline = 0.
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples, evaluation=True, control_variate=baseline)
        baseline = loss.detach()
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
        agg_loss.backward(retain_graph=True)
        est_loss_history.append(loss)
        xi_history.append(pyro.param('xi').detach().clone())
        optim(params)
        optim.step()
        print(pyro.param("xi").squeeze())
        print('eig', baseline.squeeze())

    xi_history.append(pyro.param('xi').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)

    return xi_history, est_loss_history


def marginal_gradient_eig(model, design, observation_labels, target_labels,
                          num_samples, num_steps, guide, optim, burn_in_steps=0):

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss_fn = _saddle_marginal_loss(model, guide, observation_labels, target_labels)

    params = None
    est_loss_history = []
    xi_history = []
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            d_loss, q_loss, eig_estimate = loss_fn(design, num_samples)
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(d_loss) or torch.isnan(q_loss):
            raise ArithmeticError("Encountered NaN loss in marginal_gradient_eig")
        q_loss.backward(retain_graph=True)
        optim(params)
        if step > burn_in_steps:
            (-d_loss).backward(retain_graph=True)
            optim(params)
        print(eig_estimate)
        est_loss_history.append(eig_estimate)
        xi_history.append(pyro.param('xi').detach().clone())
        optim.step()
        print(pyro.param("xi").squeeze())

    xi_history.append(pyro.param('xi').detach().clone())

    est_loss_history = torch.stack(est_loss_history)
    xi_history = torch.stack(xi_history)

    return xi_history, est_loss_history


def main(num_steps, num_samples, experiment_name, estimators, seed, start_lr, end_lr):
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

        xi_init = .01 + 99.99 * torch.rand(6 // 2)
        xi_init = torch.cat([xi_init, xi_init], dim=-1)
        observation_sd = torch.tensor(.005)
        # Change the prior distribution here
        rho_concentration = torch.tensor([[1., 1.]])
        alpha_concentration = torch.tensor([[184., 247., 418.]])
        slope_mu = torch.tensor([2.32])
        slope_sigma = torch.tensor([.0148])
        # rho_concentration = torch.tensor([[1., 1.]])
        # alpha_concentration = torch.tensor([[1., 1., 1.]])
        # slope_mu = torch.tensor([1.])
        # slope_sigma = torch.tensor([3.])
        model_learn_xi = make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma,
                                        observation_sd, xi_init=xi_init)

        contrastive_samples = num_samples ** 2

        # Fix correct loss
        if estimator == 'posterior':
            guide = LinearPosteriorGuide(tuple())
            guide.set_prior(rho_concentration, alpha_concentration, slope_mu, slope_sigma)
            loss = _differentiable_posterior_loss(model_learn_xi, guide, ["y"], ["rho", "alpha", "slope"])

        elif estimator == 'nce':
            eig_loss = lambda d, N, **kwargs: differentiable_pce_eig(
                model=model_learn_xi, design=d, observation_labels=["y"], target_labels=["rho", "alpha", "slope"],
                N=N, M=contrastive_samples, **kwargs)
            loss = neg_loss(eig_loss)

        # elif estimator == 'nce-proposal':
        #     eig_loss = lambda d, N, **kwargs: differentiable_nce_proposal_eig(
        #             model=model_learn_xi, design=d, observation_labels=["y"], target_labels=['rho', 'alpha', 'slope'],
        #             proposal=proposal, N=N, M=contrastive_samples, **kwargs)
        #     loss = neg_loss(eig_loss)

        elif estimator == 'ace':
            guide = LinearPosteriorGuide(tuple())
            guide.set_prior(rho_concentration, alpha_concentration, slope_mu, slope_sigma)
            eig_loss = _differentiable_ace_eig_loss(model_learn_xi, guide, contrastive_samples, ["y"],
                                                    ["rho", "alpha", "slope"])
            loss = neg_loss(eig_loss)

        elif estimator == 'saddle-marginal':
            guide = marginal_guide(0., 6., (1,), "y")

        else:
            raise ValueError("Unexpected estimator")

        gamma = (end_lr / start_lr) ** (1 / num_steps)
        scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                              'gamma': gamma})

        design_prototype = torch.zeros(1, 1, 6)  # this is annoying, code needs refactor

        if estimator != 'saddle-marginal':
            xi_history, est_loss_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=num_samples,
                                                                  num_steps=num_steps, optim=scheduler)
        else:
            xi_history, est_loss_history = marginal_gradient_eig(
                model_learn_xi, design_prototype, "y", ["rho", "alpha", "slope"], num_samples=num_samples, num_steps=num_steps,
                guide=guide, optim=scheduler, burn_in_steps=num_steps // 10)

        if estimator == 'posterior':
            est_eig_history = _eig_from_ape(model_learn_xi, design_prototype, ["y"], est_loss_history, True, {})
        elif estimator in ['nce', 'nce-proposal', 'ace']:
            est_eig_history = -est_loss_history
        else:
            est_eig_history = est_loss_history
        # eig_history = semi_analytic_eig(xi_history, torch.tensor(0.), torch.tensor(0.25))

        results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
                   'xi_history': xi_history, 'est_eig_history': est_eig_history}

        with open(results_file, 'wb') as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a linear model")
    parser.add_argument("--num-steps", default=2000, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    # parser.add_argument("--num-parallel", default=10, type=int)
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--estimator", default="posterior", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--start-lr", default=0.01, type=float)
    parser.add_argument("--end-lr", default=0.0005, type=float)
    args = parser.parse_args()
    main(args.num_steps, args.num_samples, args.name, args.estimator, args.seed, args.start_lr, args.end_lr)
