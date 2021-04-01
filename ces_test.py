"""
Code to test how big the variance is between posteriors found with VI using
only one experiment (i.e. 1 input-target pair)
"""
import torch
import argparse
import time
from functools import partial
from contextlib import ExitStack

import pyro
import pyro.optim as optim
import pyro.distributions as dist
from pyro.contrib.util import iter_plates_to_shape, lexpand, rexpand, rmv
from pyro.contrib.oed.eig import elbo_learn
from pyro.util import is_bad
import torch.distributions as torch_dist


epsilon = torch.tensor(2 ** -22)


def make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma, observation_sd,
                   observation_label="y"):
    def ces_model(design):
        # pyro.set_rng_seed(10)
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
            U1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
            U2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
            mean = slope * (U1rho - U2rho)
            sd = slope * observation_sd * (1 + torch.norm(d1 - d2, dim=-1, p=2))


            emission_dist = dist.CensoredSigmoidNormal(mean, sd, 1 - epsilon, epsilon).to_event(1)
            y = pyro.sample(observation_label, emission_dist)
            return y

    return ces_model


def elboguide(design, dim=10):
    rho_concentration = pyro.param("rho_concentration", torch.ones(dim, 1, 2),
                                   constraint=torch.distributions.constraints.positive)
    alpha_concentration = pyro.param("alpha_concentration", torch.ones(dim, 1, 3),
                                     constraint=torch.distributions.constraints.positive)
    slope_mu = pyro.param("slope_mu", torch.ones(dim, 1))
    slope_sigma = pyro.param("slope_sigma", 3. * torch.ones(dim, 1),
                             constraint=torch.distributions.constraints.positive)
    batch_shape = design.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)
        rho_shape = batch_shape + (rho_concentration.shape[-1],)
        pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape)))
        alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
        pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
        pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape),
                                            slope_sigma.expand(batch_shape)))


def main(num_parallel, seed, obs_sd, elbo_n_samples, elbo_n_steps):

    pyro.clear_param_store()
    if seed >= 0:
        pyro.set_rng_seed(seed)
    else:
        seed = int(torch.rand(tuple()) * 2 ** 30)
        pyro.set_rng_seed(seed)
    elbo_lr = 0.04
    design_dim = 6
    observation_sd = torch.tensor(obs_sd)

    rho_concentration = torch.ones(num_parallel, 1, 2)
    alpha_concentration = torch.ones(num_parallel, 1, 3)
    slope_mu, slope_sigma = torch.ones(num_parallel, 1), 3. * torch.ones(num_parallel, 1)
    prior = make_ces_model(torch.ones(num_parallel, 1, 2), torch.ones(num_parallel, 1, 3),
                                           torch.ones(num_parallel, 1), 3. * torch.ones(num_parallel, 1), observation_sd)# Design phase
    true_model = pyro.condition(make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma,
                                               observation_sd),
                                {"rho": torch.tensor([.9, .1]), "alpha": torch.tensor([.2, .3, .5]),
                                 "slope": torch.tensor(10.)})
    d_star_design = .01 + 99.99 * torch.rand((num_parallel, 1, 1, design_dim))
    d_star_designs = lexpand(d_star_design[0], num_parallel)
    d_star_designs2 = lexpand(d_star_design[1], num_parallel)
    # d_star_designs = torch.cat([d_star_designs, d_star_designs2], dim=-2)

    ys = true_model(d_star_designs)

    t = time.time()
    loss = elbo_learn(
        prior, d_star_designs, ["y"], ["rho", "alpha", "slope"], elbo_n_samples, elbo_n_steps,
        partial(elboguide, dim=num_parallel), {"y": ys}, optim.Adam({"lr": elbo_lr})
    )
    elapsed = time.time() - t
    print(f"time elapsed: {elapsed}s")
    rho_concentration = pyro.param("rho_concentration").detach().data.clone()
    alpha_concentration = pyro.param("alpha_concentration").detach().data.clone()
    slope_mu = pyro.param("slope_mu").detach().data.clone()
    slope_sigma = pyro.param("slope_sigma").detach().data.clone()
    parameters = torch.cat(
        [rho_concentration.reshape(num_parallel, -1),
         alpha_concentration.reshape(num_parallel, -1),
         slope_mu.reshape(num_parallel, -1),
         slope_sigma.reshape(num_parallel, -1),
         ],
        dim=-1
    )
    rho_dist = torch_dist.Dirichlet(rho_concentration)
    alpha_dist = torch_dist.Dirichlet(alpha_concentration)
    u_dist = torch_dist.LogNormal(slope_mu, slope_sigma)
    entropy = rho_dist.entropy() + alpha_dist.entropy() + u_dist.entropy()
    print(f"posterior mean: {parameters.mean(axis=0)}")
    print(f"posterior std: {parameters.std(axis=0)}")
    print(f"posterior entropy: {entropy.mean(), entropy.std()}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CES (Constant Elasticity of Substitution) indifference"
                                                 " iterated experiment design")
    parser.add_argument("--elbo-n-samples", nargs="?", default=10, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=100, type=int)
    parser.add_argument("--elbo-n-steps", nargs="?", default=1000, type=int)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--observation-sd", default=0.005, type=float)
    args = parser.parse_args()
    main(args.num_parallel, args.seed, args.observation_sd, args.elbo_n_samples,
         args.elbo_n_steps)
