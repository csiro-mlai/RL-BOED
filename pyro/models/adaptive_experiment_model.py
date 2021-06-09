from abc import ABC

from contextlib import ExitStack
from functools import partial
from pyro import poutine
from pyro.contrib.util import iter_plates_to_shape, lexpand, rexpand, rmv
from pyro.util import is_bad
from scipy.integrate import solve_ivp

import torch
import pyro
import pyro.distributions as dist

EPS = 2**-22


class ExperimentModel(ABC):
    """
    Basic interface for probabilistic models
    """

    def __init__(self):
        self.epsilon = torch.tensor(EPS)

    def sanity_check(self):
        assert self.var_dim > 0
        assert len(self.var_names) > 0

    def make_model(self):
        raise NotImplementedError

    def reset(self, n_parallel):
        raise NotImplementedError

    def run_experiment(self, design, theta):
        """
        Execute an experiment with given design.
        """
        # create model from sampled params
        cond_model = pyro.condition(self.make_model(), data=theta)

        # infer experimental outcome given design and model
        y = cond_model(design)
        y = y.detach().clone()
        return y

    def get_likelihoods(self, y, design, thetas):
        size = thetas[self.var_names[0]].shape[0]
        cond_dict = dict(thetas)
        cond_dict.update({self.obs_label: lexpand(y, size)})
        cond_model = pyro.condition(self.make_model(), data=cond_dict)
        trace = poutine.trace(cond_model).get_trace(lexpand(design, size))
        trace.compute_log_prob()
        likelihoods = trace.nodes[self.obs_label]["log_prob"]
        return likelihoods

    def sample_theta(self, num_theta):
        dummy_design = torch.zeros(
            (num_theta, self.n_parallel, 1, 1, self.var_dim))
        cur_model = self.make_model()
        trace = poutine.trace(cur_model).get_trace(dummy_design)
        thetas = dict([(l, trace.nodes[l]["value"]) for l in self.var_names])
        return thetas


class CESModel(ExperimentModel):
    def __init__(self, init_rho_model=None, init_alpha_model=None,
                 init_mu_model=None, init_sig_model=None, n_parallel=1,
                 obs_sd=0.005, obs_label="y", n_elbo_samples=100,
                 n_elbo_steps=100, elbo_lr=0.04):
        super().__init__()
        self.init_rho_model = init_rho_model if init_rho_model is not None \
            else torch.ones(n_parallel, 1, 2)
        self.init_alpha_model = init_alpha_model \
            if init_alpha_model is not None else torch.ones(n_parallel, 1, 3)
        self.init_mu_model = init_mu_model if init_mu_model is not None \
            else torch.ones(n_parallel, 1)
        self.init_sig_model = init_sig_model if init_sig_model is not None \
            else 3. * torch.ones(n_parallel, 1)
        self.rho_con_model = self.init_rho_model.detach().clone()
        self.alpha_con_model = self.init_alpha_model.detach().clone()
        self.u_mu_model = self.init_mu_model.detach().clone()
        self.u_sig_model = self.init_sig_model.detach().clone()
        self.n_parallel, self.elbo_lr = n_parallel, elbo_lr
        self.n_elbo_samples, self.n_elbo_steps = n_elbo_samples, n_elbo_steps
        self.obs_sd = obs_sd
        self.obs_label = obs_label
        self.param_names = [
            "rho_con",
            "alpha_con",
            "u_mu",
            "u_sig",
        ]
        self.var_names = ["rho", "alpha", "u"]
        self.var_dim = 6
        self.sanity_check()

    def reset(self, init_rho_model=None, init_alpha_model=None,
              init_mu_model=None, init_sig_model=None, n_parallel=None):
        if n_parallel is not None:
            self.n_parallel = n_parallel
            self.init_rho_model = init_rho_model if init_rho_model \
                else torch.ones(self.n_parallel, 1, 2)
            self.init_alpha_model = init_alpha_model if init_alpha_model \
                else torch.ones(self.n_parallel, 1, 3)
            self.init_mu_model = init_mu_model if init_mu_model \
                else torch.ones(self.n_parallel, 1)
            self.init_sig_model = init_sig_model if init_sig_model \
                else 3. * torch.ones(self.n_parallel, 1)
            self.rho_con_model = self.init_rho_model.detach().clone()
            self.alpha_con_model = self.init_alpha_model.detach().clone()
            self.u_mu_model = self.init_mu_model.detach().clone()
            self.u_sig_model = self.init_sig_model.detach().clone()

    def make_model(self):
        def model(design):
            # pyro.set_rng_seed(10)
            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            batch_shape = design.shape[:-2]
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                rho_shape = batch_shape + (self.rho_con_model.shape[-1],)
                rho = 0.01 + 0.99 * pyro.sample(
                    "rho",
                    dist.Dirichlet(self.rho_con_model.expand(rho_shape))
                ).select(-1, 0)
                alpha_shape = batch_shape + (self.alpha_con_model.shape[-1],)
                alpha = pyro.sample(
                    "alpha",
                    dist.Dirichlet(self.alpha_con_model.expand(alpha_shape))
                )
                u = pyro.sample(
                    "u",
                    dist.LogNormal(
                        self.u_mu_model.expand(batch_shape),
                        self.u_sig_model.expand(batch_shape)
                    )
                )
                rho = rexpand(rho, design.shape[-2])
                u = rexpand(u, design.shape[-2])
                d1, d2 = design[..., 0:3], design[..., 3:6]
                u1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
                u2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
                mean = u * (u1rho - u2rho)
                sd = u * self.obs_sd * (
                        1 + torch.norm(d1 - d2, dim=-1, p=2))

                emission_dist = dist.CensoredSigmoidNormal(
                    mean, sd, 1 - self.epsilon, self.epsilon
                ).to_event(1)
                y = pyro.sample(self.obs_label, emission_dist)
                return y

        return model

    def get_params(self):
        return torch.cat(
            [
                self.rho_con_model.reshape(self.n_parallel, -1),
                self.alpha_con_model.reshape(self.n_parallel, -1),
                self.u_mu_model.reshape(self.n_parallel, -1),
                self.u_sig_model.reshape(self.n_parallel, -1),
            ],
            dim=-1
        )


def holling2(a, th, t, n):
    an = a * n
    return -an / (1 + an * th)


class PreyModel(ExperimentModel):
    def __init__(self, a_mu=None, a_sig=None, th_mu=None, th_sig=None, tau=24,
                 n_parallel=1, obs_sd=0.005, obs_label="y"):
        super().__init__()
        self.a_mu = a_mu if a_mu is not None \
            else torch.ones(n_parallel, 1) * -1.4
        self.a_sig = a_sig if a_sig is not None \
            else torch.ones(n_parallel, 1) * 1.35
        self.th_mu = th_mu if th_mu is not None \
            else torch.ones(n_parallel, 1) * -1.4
        self.th_sig = th_sig if th_sig is not None \
            else torch.ones(n_parallel, 1) * 1.35
        self.tau = tau
        self.n_parallel = n_parallel
        self.obs_sd = obs_sd
        self.obs_label = obs_label
        self.var_names = ["a", "th"]
        self.var_dim = 2
        self.sanity_check()

    def make_model(self):
        def model(design):
            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            batch_shape = design.shape[:-2]
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                a = pyro.sample(
                    "a",
                    dist.LogNormal(
                        self.a_mu.expand(batch_shape),
                        self.a_sig.expand(batch_shape)
                    )
                )
                th = pyro.sample(
                    "th",
                    dist.LogNormal(
                        self.th_mu.expand(batch_shape),
                        self.th_sig.expand(batch_shape)
                    )
                )
                diff_func = partial(holling2, a.numpy(), th.numpy())
                n_t = solve_ivp(diff_func, (0, self.tau), design.numpy())
                n_t = torch.as_tensor(n_t)
                p_t = (design - n_t) / design
                emission_dist = dist.Binomial(design, p_t)
                n = pyro.sample(
                    self.obs_label, emission_dist
                )
                return n

        return model


class SourceModel(ExperimentModel):
    def __init__(self, d=2, k=2, theta_mu=None, theta_sig=None, alpha=None,
                 b=1e-1, m=1e-4, n_parallel=1, obs_sd=0.5, obs_label="y"):
        super().__init__()
        self.theta_mu = theta_mu if theta_mu is not None \
            else torch.zeros(n_parallel, 1, d, k)
        self.theta_sig = theta_sig if theta_sig is not None \
            else torch.ones(n_parallel, 1, d, k)
        self.alpha = alpha if alpha is not None \
            else torch.ones(n_parallel, 1, k)
        self.d, self.k, self.b, self.m = d, k, b, m
        self.obs_sd, self.obs_label = obs_sd, obs_label
        self.n_parallel = n_parallel
        self.var_names = ["theta"]
        self.var_dim = d
        self.sanity_check()

    def make_model(self):
        def model(design):
            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            batch_shape = design.shape[:-2]
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                theta_shape = batch_shape + self.theta_mu.shape[-2:]
                theta = pyro.sample(
                    "theta",
                    dist.Normal(
                        self.theta_mu.expand(theta_shape),
                        self.theta_sig.expand(theta_shape)
                    ).to_event(2)
                )
                distance = torch.square(theta - design).sum(dim=-2)
                ratio = self.alpha / (self.m + distance)
                mu = self.b + ratio.sum(dim=-1, keepdims=True)
                emission_dist = dist.Normal(torch.log(mu), self.obs_sd).to_event(1)
                y = pyro.sample(self.obs_label, emission_dist)
                return y

        return model

    def reset(self, n_parallel):
        pass


















