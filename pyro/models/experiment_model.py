from abc import ABC

from pyro.contrib.oed.eig import elbo_learn
from pyro.util import is_bad
from contextlib import ExitStack
from pyro.contrib.util import iter_plates_to_shape, rexpand, rmv

import torch
import pyro
import pyro.distributions as dist
import pyro.optim as optim

epsilon = torch.tensor(1e-9)


class ExperimentModel(ABC):
    """
    A model class for
    """

    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def make_model(self):
        raise NotImplementedError

    def guide(self, design):
        raise NotImplementedError

    def run_experiment(self, design, y):
        raise NotImplementedError

    def get_params(self):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError


class CESModel(ExperimentModel):
    def __init__(self, init_rho=None, init_alpha=None, init_mu=None,
                 init_sig=None, n_parallel=1, obs_sd=0.005, obs_label="y",
                 n_elbo_samples=1, n_elbo_steps=100, elbo_lr=0.04):
        super().__init__()
        self.init_rho = init_rho if init_rho else torch.ones(n_parallel, 1, 2)
        self.init_alpha = \
            init_alpha if init_alpha else torch.ones(n_parallel, 1, 3)
        self.init_mu = init_mu if init_mu else torch.ones(n_parallel, 1)
        self.init_sig = init_sig if init_sig else 3. * torch.ones(n_parallel, 1)
        self.rho_con = self.init_rho.detach().clone()
        self.alpha_con = self.init_alpha.detach().clone()
        self.slope_mu = self.init_mu.detach().clone()
        self.slope_sig = self.init_sig.detach().clone()
        self.n_parallel, self.elbo_lr = n_parallel, elbo_lr
        self.n_elbo_samples, self.n_elbo_steps = n_elbo_samples, n_elbo_steps
        self.obs_sd = obs_sd
        self.obs_label = obs_label
        self.param_names = [
            "rho_con",
            "alpha_con",
            "slope_mu",
            "slope_sig",
        ]

    def reset(self):
        self.rho_con = self.init_rho.detach().clone()
        self.alpha_con = self.init_alpha.detach().clone()
        self.slope_mu = self.init_mu.detach().clone()
        self.slope_sig = self.init_sig.detach().clone()
        param_store = pyro.get_param_store()
        for name in self.param_names:
            if name in param_store:
                del param_store[name]

        pyro.param("rho_con", self.init_rho.detach().clone(),
                   constraint=dist.constraints.positive)
        pyro.param("alpha_con", self.init_alpha.detach().clone(),
                   constraint=dist.constraints.positive)
        pyro.param("slope_mu", self.init_mu.detach().clone())
        pyro.param("slope_sig", self.init_sig.detach().clone(),
                   constraint=dist.constraints.positive)

    def make_model(self):
        def model(design):
            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            batch_shape = design.shape[:-2]
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                rho_shape = batch_shape + (self.rho_con.shape[-1],)
                rho = 0.01 + 0.99 * pyro.sample(
                    "rho",
                    dist.Dirichlet(self.rho_con.expand(rho_shape))
                ).select(-1, 0)
                alpha_shape = batch_shape + (self.alpha_con.shape[-1],)
                alpha = pyro.sample(
                    "alpha",
                    dist.Dirichlet(self.alpha_con.expand(alpha_shape))
                )
                slope = pyro.sample(
                    "slope",
                    dist.LogNormal(
                        self.slope_mu.expand(batch_shape),
                        self.slope_sig.expand(batch_shape)
                    )
                )
                rho = rexpand(rho, design.shape[-2])
                slope = rexpand(slope, design.shape[-2])
                d1, d2 = design[..., 0:3], design[..., 3:6]
                u1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
                u2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
                mean = slope * (u1rho - u2rho)
                sd = slope * self.obs_sd * (
                        1 + torch.norm(d1 - d2, dim=-1, p=2))

                emission_dist = dist.CensoredSigmoidNormal(
                    mean, sd, 1 - epsilon, epsilon
                ).to_event(1)
                y = pyro.sample(self.obs_label, emission_dist)
                return y

        return model

    def guide(self, design):
        rho_con = pyro.param("rho_con", self.init_rho.detach().clone(),
                             constraint=dist.constraints.positive)
        alpha_con = pyro.param("alpha_con", self.init_alpha.detach().clone(),
                               constraint=dist.constraints.positive)
        slope_mu = pyro.param("slope_mu", self.init_mu.detach().clone())
        slope_sig = pyro.param("slope_sig", self.init_sig.detach().clone(),
                               constraint=dist.constraints.positive)
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            rho_shape = batch_shape + (rho_con.shape[-1],)
            pyro.sample("rho", dist.Dirichlet(rho_con.expand(rho_shape)))
            alpha_shape = batch_shape + (alpha_con.shape[-1],)
            pyro.sample("alpha", dist.Dirichlet(alpha_con.expand(alpha_shape)))
            pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape),
                                                slope_sig.expand(batch_shape)))

    def run_experiment(self, design, y=None):
        """
        Execute an experiment with given design. if `y` is `None` then fill in
        a value predicted by the model.
        """
        # create model from up-to-date params
        cur_model = self.make_model()

        # infer experimental outcome given design and model
        if y is None:
            y = cur_model(design)
        y = y.detach().clone()

        # learn the posterior given design and outcome
        elbo_learn(
            cur_model, design, [self.obs_label], ["rho", "alpha", "slope"],
            self.n_elbo_samples, self.n_elbo_steps, self.guide,
            {self.obs_label: y}, optim.Adam({"lr": self.elbo_lr})
        )

        # update parameters
        self.rho_con = pyro.param("rho_con").detach().clone()
        self.alpha_con = pyro.param("alpha_con").detach().clone()
        self.slope_mu = pyro.param("slope_mu").detach().clone()
        self.slope_sig = pyro.param("slope_sig").detach().clone()

    def get_params(self):
        return torch.cat([
            torch.flatten(self.rho_con),
            torch.flatten(self.alpha_con),
            torch.flatten(self.slope_mu),
            torch.flatten(self.slope_sig),
        ])

    def entropy(self):
        rho_dist = dist.Dirchlet(self.rho_con)
        alpha_dist = dist.Dirchlet(self.alpha_con)
        slope_dist = dist.LogNormal(self.slope_mu, self.slope_sig)
        return rho_dist.entropy() + alpha_dist.entropy() + slope_dist.entropy()
