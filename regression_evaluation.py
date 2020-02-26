import argparse
import pickle
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.oed.eig import opt_eig_ape_loss, _vnmc_eig_loss, _ace_eig_loss
from pyro.contrib.util import rmv
from pyro.util import is_bad

from regression import PosteriorGuide, neg_loss

output_dir = "./run_outputs/gradinfo/"


def make_regression_model(w_loc, w_scale, sigma_scale, observation_label="y"):
    def regression_model(design):
        design = design / design.norm(dim=-1, p=1, keepdim=True)
        if is_bad(design):
            raise ArithmeticError("bad design, contains nan or inf")
        batch_shape = design.shape[:-2]
        with pyro.plate_stack("plate_stack", batch_shape):
            # `w` is shape p, the prior on each component is independent
            w = pyro.sample("w", dist.Laplace(w_loc, w_scale).to_event(1))
            # `sigma` is scalar
            sigma = 1e-6 + pyro.sample("sigma", dist.Exponential(sigma_scale)).unsqueeze(-1)
            mean = rmv(design, w)
            sd = sigma
            y = pyro.sample(observation_label, dist.Normal(mean, sd).to_event(1))
            return y

    return regression_model


def main(name, num_inner_samples, num_outer_samples, device, n, p, scale):

    fname = output_dir + name + ".pickle"
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    xi_history = results['xi_history']
    design = xi_history[-1].to(device)
    print(design.shape, design.max(), design.min())
    num_parallel = design.shape[0]

    w_prior_loc = scale * torch.zeros(p, device=device)
    w_prior_scale = scale * torch.ones(p, device=device)
    sigma_prior_scale = torch.tensor(1., device=device)

    model = make_regression_model(
        w_prior_loc, w_prior_scale, sigma_prior_scale)

    guide = PosteriorGuide(n, p, (num_parallel,)).to(device)
    targets = ["w", "sigma"]
    eig_loss = _ace_eig_loss(model, guide, 10, ["y"], targets)
    loss = neg_loss(eig_loss)
    optimizer = pyro.optim.Adam({"lr": 0.001})

    # Train guide
    print("Training")
    opt_eig_ape_loss(design, loss, num_samples=10, num_steps=20000, optim=optimizer)

    # Evaluate
    print("Evaluation")
    lower_loss = _ace_eig_loss(model, guide, num_inner_samples, "y", targets)  # isn't that an annoying API difference?
    upper_loss = _vnmc_eig_loss(model, guide, "y", targets)
    lower, upper = 0., 0.
    max_samples = 10000
    n_per_batch = max_samples // num_inner_samples
    n_batches = num_inner_samples * num_outer_samples // max_samples
    for i in range(n_batches):
        print(i)
        lower += lower_loss(design, n_per_batch, evaluation=True)[1].detach().cpu()
        upper += upper_loss(design, (n_per_batch, num_inner_samples), evaluation=True)[1].detach().cpu()

    results['final_upper_bound'] = upper.cpu() / n_batches
    results['final_lower_bound'] = lower.cpu() / n_batches

    print(results['final_lower_bound'], results['final_upper_bound'])

    with open(fname, 'wb') as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use ACE/VNMC to evaluate docking designs")
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--num-inner-samples", default=2500, type=int)
    parser.add_argument("--num-outer-samples", default=100000, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("-n", default=20, type=int)
    parser.add_argument("-p", default=20, type=int)
    parser.add_argument("--scale", default=1., type=float)
    args = parser.parse_args()

    main(args.name, args.num_inner_samples, args.num_outer_samples, args.device, args.n, args.p, args.scale)
