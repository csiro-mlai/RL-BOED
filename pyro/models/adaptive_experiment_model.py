from abc import ABC

from pyro.contrib.oed.eig import elbo_learn
from pyro.util import is_bad
from pyro import poutine
from contextlib import ExitStack
from pyro.contrib.util import iter_plates_to_shape, lexpand, rexpand, rmv

import torch
import pyro
import pyro.distributions as dist
import torch.distributions as torch_dist
import pyro.optim as optim

epsilon = torch.tensor(2**-22)


class ExperimentModel(ABC):
    """
    A model class for
    """

    def __init__(self):
        pass

    def make_model(self):
        raise NotImplementedError

    def run_experiment(self, design, y):
        raise NotImplementedError

    def get_likelihoods(self, y, design, thetas):
        raise NotImplementedError

    def sample_theta(self, num_theta):
        raise NotImplementedError


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
                    mean, sd, 1 - epsilon, epsilon
                ).to_event(1)
                y = pyro.sample(self.obs_label, emission_dist)
                return y

        return model

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

    def get_likelihoods(self, y, design, thetas):
        size = thetas["rho"].shape[0]
        cond_dict = dict(thetas)
        cond_dict.update({"y": lexpand(y, size)})
        cond_model = pyro.condition(self.make_model(), data=cond_dict)
        trace = poutine.trace(cond_model).get_trace(lexpand(design, size))
        trace.compute_log_prob()
        likelihoods = trace.nodes["y"]["log_prob"]
        return likelihoods

    def sample_theta(self, num_theta):
        dummy_design = torch.zeros((num_theta, self.n_parallel, 1, 1, 6))
        cur_model = self.make_model()
        trace = poutine.trace(cur_model).get_trace(dummy_design)
        thetas = dict([(l, trace.nodes[l]["value"]) for l in self.var_names])
        return thetas


class DeathModel(ExperimentModel):
    def __init__(self):
        super().__init__()

    def get_likelihoods(self, y, design, thetas):
        pass

    def run_experiment(self, design, y):
        pass

    def sample_theta(self, num_theta):
        pass

    def make_model(self):
        pass
