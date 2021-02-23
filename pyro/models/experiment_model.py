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

epsilon = torch.tensor(2**-22)


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
    def __init__(self, init_rho_model=None, init_alpha_model=None,
                 init_mu_model=None, init_sig_model=None, init_rho_guide=None,
                 init_alpha_guide=None, init_mu_guide=None, init_sig_guide=None,
                 init_ys=None, init_ds=None, n_parallel=1, obs_sd=0.005,
                 obs_label="y", n_elbo_samples=100, n_elbo_steps=100,
                 elbo_lr=0.04, markovian=False):
        super().__init__()
        self.init_rho_model = init_rho_model if init_rho_model is not None \
            else torch.ones(n_parallel, 1, 2)
        self.init_alpha_model = init_alpha_model \
            if init_alpha_model is not None else torch.ones(n_parallel, 1, 3)
        self.init_mu_model = init_mu_model if init_mu_model is not None \
            else torch.ones(n_parallel, 1)
        self.init_sig_model = init_sig_model if init_sig_model is not None \
            else 3. * torch.ones(n_parallel, 1)
        self.init_rho_guide = init_rho_guide if init_rho_guide is not None \
            else torch.ones(n_parallel, 1, 2)
        self.init_alpha_guide = init_alpha_guide \
            if init_alpha_guide is not None else torch.ones(n_parallel, 1, 3)
        self.init_mu_guide = init_mu_guide if init_mu_guide is not None \
            else torch.ones(n_parallel, 1)
        self.init_sig_guide = init_sig_guide if init_sig_guide is not None \
            else 3. * torch.ones(n_parallel, 1)
        self.rho_con_model = self.init_rho_model.detach().clone()
        self.alpha_con_model = self.init_alpha_model.detach().clone()
        self.u_mu_model = self.init_mu_model.detach().clone()
        self.u_sig_model = self.init_sig_model.detach().clone()
        self.rho_con_guide = self.init_rho_guide.detach().clone()
        self.alpha_con_guide = self.init_alpha_guide.detach().clone()
        self.u_mu_guide = self.init_mu_guide.detach().clone()
        self.u_sig_guide = self.init_sig_guide.detach().clone()
        self.n_parallel, self.elbo_lr = n_parallel, elbo_lr
        self.n_elbo_samples, self.n_elbo_steps = n_elbo_samples, n_elbo_steps
        self.obs_sd = obs_sd
        self.obs_label = obs_label
        self.init_ys = init_ys if init_ys is not None else torch.tensor([])
        self.init_ds = init_ds if init_ds is not None else torch.tensor([])
        self.ys = self.init_ys.detach().clone()
        self.ds = self.init_ds.detach().clone()
        self.markovian = markovian
        self.param_names = [
            "rho_con",
            "alpha_con",
            "u_mu",
            "u_sig",
        ]

    def reset(self, n_parallel=None):
        if n_parallel is not None:
            self.n_parallel = n_parallel
        self.init_rho_model = lexpand(self.init_rho_model[0], self.n_parallel)
        self.init_alpha_model = lexpand(self.init_alpha_model[0],
                                        self.n_parallel)
        self.init_mu_model = lexpand(self.init_mu_model[0], self.n_parallel)
        self.init_sig_model = lexpand(self.init_sig_model[0], self.n_parallel)
        self.init_rho_guide = lexpand(self.init_rho_guide[0], self.n_parallel)
        self.init_alpha_guide = lexpand(self.init_alpha_guide[0],
                                        self.n_parallel)
        self.init_mu_guide = lexpand(self.init_mu_guide[0], self.n_parallel)
        self.init_sig_guide = lexpand(self.init_sig_guide[0], self.n_parallel)
        self.rho_con_model = self.init_rho_model.detach().clone()
        self.alpha_con_model = self.init_alpha_model.detach().clone()
        self.u_mu_model = self.init_mu_model.detach().clone()
        self.u_sig_model = self.init_sig_model.detach().clone()
        self.rho_con_guide = self.init_rho_guide.detach().clone()
        self.alpha_con_guide = self.init_alpha_guide.detach().clone()
        self.u_mu_guide = self.init_mu_guide.detach().clone()
        self.u_sig_guide = self.init_sig_guide.detach().clone()
        param_store = pyro.get_param_store()
        for name in self.param_names:
            if name in param_store:
                del param_store[name]

        pyro.param("rho_con", self.rho_con_guide.detach().clone(),
                   constraint=torch_dist.constraints.positive)
        pyro.param("alpha_con", self.alpha_con_guide.detach().clone(),
                   constraint=torch_dist.constraints.positive)
        pyro.param("u_mu", self.u_mu_guide.detach().clone())
        pyro.param("u_sig", self.u_sig_guide.detach().clone(),
                   constraint=torch_dist.constraints.positive)
        self.ys = self.init_ys.detach().clone()
        self.ds = self.init_ds.detach().clone()

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

    def guide(self, design):
        # pyro.set_rng_seed(10)
        rho_con = pyro.param("rho_con", self.init_rho_guide.detach().clone(),
                             constraint=torch_dist.constraints.positive)
        alpha_con = pyro.param("alpha_con",
                               self.init_alpha_guide.detach().clone(),
                               constraint=torch_dist.constraints.positive)
        u_mu = pyro.param("u_mu", self.init_mu_guide.detach().clone())
        u_sig = pyro.param("u_sig", self.init_sig_guide.detach().clone(),
                           constraint=torch_dist.constraints.positive)
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
        self.ds = torch.cat([self.ds, design], dim=-2)

        # learn the posterior given design and outcome
        # pyro.set_rng_seed(10)
        elbo_learn(
            cur_model, self.ds, [self.obs_label], ["rho", "alpha", "u"],
            self.n_elbo_samples, self.n_elbo_steps, self.guide,
            {self.obs_label: self.ys}, optim.Adam({"lr": self.elbo_lr})
        )

        # update parameters
        if self.markovian:
            self.rho_con_model = pyro.param("rho_con").detach().clone()
            self.alpha_con_model = pyro.param("alpha_con").detach().clone()
            self.u_mu_model = pyro.param("u_mu").detach().clone()
            self.u_sig_model = pyro.param("u_sig").detach().clone()
        self.rho_con_guide = pyro.param("rho_con").detach().clone()
        self.alpha_con_guide = pyro.param("alpha_con").detach().clone()
        self.u_mu_guide = pyro.param("u_mu").detach().clone()
        self.u_sig_guide = pyro.param("u_sig").detach().clone()

    def get_params(self):
        return torch.cat(
            [
                self.rho_con_guide.reshape(self.n_parallel, -1),
                self.alpha_con_guide.reshape(self.n_parallel, -1),
                self.u_mu_guide.reshape(self.n_parallel, -1),
                self.u_sig_guide.reshape(self.n_parallel, -1),
            ],
            dim=-1
        )

    def entropy(self):
        rho_dist = torch_dist.Dirichlet(self.rho_con_guide)
        alpha_dist = torch_dist.Dirichlet(self.alpha_con_guide)
        u_dist = torch_dist.LogNormal(self.u_mu_guide, self.u_sig_guide)
        return rho_dist.entropy() + alpha_dist.entropy() + u_dist.entropy()
