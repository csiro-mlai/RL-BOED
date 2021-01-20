from abc import ABC

from pyro.contrib.oed.eig import elbo_learn
from pyro.util import is_bad
from contextlib import ExitStack
from pyro.contrib.util import iter_plates_to_shape, lexpand, rexpand, rmv

import torch
import pyro
import pyro.distributions as dist
import torch.distributions as torch_dist
import pyro.optim as optim

epsilon = torch.tensor(1e-9)
eps_constraint = torch_dist.constraints.greater_than(epsilon)


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
                 n_elbo_samples=100, n_elbo_steps=100, elbo_lr=0.04):
        super().__init__()
        self.init_rho = init_rho if init_rho else torch.ones(n_parallel, 1, 2)
        self.init_alpha = \
            init_alpha if init_alpha else torch.ones(n_parallel, 1, 3)
        self.init_mu = init_mu if init_mu else torch.ones(n_parallel, 1)
        self.init_sig = init_sig if init_sig else 3. * torch.ones(n_parallel, 1)
        self.rho_con = self.init_rho.detach().clone()
        self.alpha_con = self.init_alpha.detach().clone()
        self.u_mu = self.init_mu.detach().clone()
        self.u_sig = self.init_sig.detach().clone()
        self.n_parallel, self.elbo_lr = n_parallel, elbo_lr
        self.n_elbo_samples, self.n_elbo_steps = n_elbo_samples, n_elbo_steps
        self.obs_sd = obs_sd
        self.obs_label = obs_label
        self.ys = torch.tensor([])
        self.param_names = [
            "rho_con",
            "alpha_con",
            "u_mu",
            "u_sig",
        ]

    def reset(self, n_parallel=None):
        if n_parallel is not None:
            self.n_parallel = n_parallel
        self.init_rho = lexpand(self.init_rho[0], self.n_parallel)
        self.init_alpha = lexpand(self.init_alpha[0], self.n_parallel)
        self.init_mu = lexpand(self.init_mu[0], self.n_parallel)
        self.init_sig = lexpand(self.init_sig[0], self.n_parallel)
        self.rho_con = self.init_rho.detach().clone()
        self.alpha_con = self.init_alpha.detach().clone()
        self.u_mu = self.init_mu.detach().clone()
        self.u_sig = self.init_sig.detach().clone()
        param_store = pyro.get_param_store()
        for name in self.param_names:
            if name in param_store:
                del param_store[name]

        pyro.param("rho_con", self.rho_con.detach().clone(),
                   constraint=eps_constraint)
        pyro.param("alpha_con", self.alpha_con.detach().clone(),
                   constraint=eps_constraint)
        pyro.param("u_mu", self.u_mu.detach().clone())
        pyro.param("u_sig", self.u_sig.detach().clone(),
                   constraint=eps_constraint)
        self.ys = torch.tensor([])

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
                u = pyro.sample(
                    "u",
                    dist.LogNormal(
                        self.u_mu.expand(batch_shape),
                        self.u_sig.expand(batch_shape)
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

    def guide(self, design):
        rho_con = pyro.param("rho_con", self.init_rho.detach().clone(),
                             constraint=eps_constraint)
        alpha_con = pyro.param("alpha_con", self.init_alpha.detach().clone(),
                               constraint=eps_constraint)
        u_mu = pyro.param("u_mu", self.init_mu.detach().clone())
        u_sig = pyro.param("u_sig", self.init_sig.detach().clone(),
                           constraint=eps_constraint)
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            rho_shape = batch_shape + (rho_con.shape[-1],)
            pyro.sample("rho", dist.Dirichlet(rho_con.expand(rho_shape)))
            alpha_shape = batch_shape + (alpha_con.shape[-1],)
            pyro.sample("alpha", dist.Dirichlet(alpha_con.expand(alpha_shape)))
            pyro.sample("u", dist.LogNormal(u_mu.expand(batch_shape),
                                            u_sig.expand(batch_shape)))

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
        self.ys = torch.cat([self.ys, y], dim=-1)

        # learn the posterior given design and outcome
        elbo_learn(
            cur_model, design, [self.obs_label], ["rho", "alpha", "u"],
            self.n_elbo_samples, self.n_elbo_steps, self.guide,
            {self.obs_label: y}, optim.Adam({"lr": self.elbo_lr})
        )

        # update parameters
        self.rho_con = pyro.param("rho_con").detach().clone()
        self.alpha_con = pyro.param("alpha_con").detach().clone()
        self.u_mu = pyro.param("u_mu").detach().clone()
        self.u_sig = pyro.param("u_sig").detach().clone()

    def get_params(self):
        return torch.cat(
            [
                self.rho_con.reshape(self.n_parallel, -1),
                self.alpha_con.reshape(self.n_parallel, -1),
                self.u_mu.reshape(self.n_parallel, -1),
                self.u_sig.reshape(self.n_parallel, -1),
            ],
            dim=-1
        )

    def entropy(self):
        rho_dist = torch_dist.Dirichlet(self.rho_con)
        alpha_dist = torch_dist.Dirichlet(self.alpha_con)
        u_dist = torch_dist.LogNormal(self.u_mu, self.u_sig)
        return rho_dist.entropy() + alpha_dist.entropy() + u_dist.entropy()
