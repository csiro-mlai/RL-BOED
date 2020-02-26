import argparse
import pickle
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.oed.eig import opt_eig_ape_loss, _vnmc_eig_loss, _ace_eig_loss
from pyro.contrib.oed.differentiable_eig import _differentiable_ace_eig_loss

from docking_gradients import PosteriorGuide, neg_loss, sigmoid

output_dir = "./run_outputs/gradinfo/"


def make_docking_model(top_c, bottom_c, ee50_mu, ee50_sigma, slope_mu, slope_sigma, observation_label="y"):
    def docking_model(design):
        batch_shape = design.shape[:-1]
        with pyro.plate_stack("plate_stack", batch_shape):
            top = pyro.sample("top", dist.Dirichlet(top_c)).select(-1, 0).unsqueeze(-1)
            bottom = pyro.sample("bottom", dist.Dirichlet(bottom_c)).select(-1, 0).unsqueeze(-1)
            ee50 = pyro.sample("ee50", dist.Normal(ee50_mu, ee50_sigma)).unsqueeze(-1)
            slope = pyro.sample("slope", dist.Normal(slope_mu, slope_sigma)).unsqueeze(-1)
            hit_rate = sigmoid(design, top, bottom, ee50, slope)
            y = pyro.sample(observation_label, dist.Bernoulli(hit_rate).to_event(1))
            return y

    return docking_model


def main(name, num_inner_samples, device):

    fname = output_dir + name + ".pickle"
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    xi_history = results['xi_history']
    design = xi_history[-1].to(device)
    num_parallel = design.shape[0]

    D = 100
    top_prior_concentration = torch.tensor([25., 75.], device=device)
    bottom_prior_concentration = torch.tensor([4., 96.], device=device)
    ee50_prior_mu, ee50_prior_sd = torch.tensor(-50., device=device), torch.tensor(15., device=device)
    slope_prior_mu, slope_prior_sd = torch.tensor(-0.15, device=device), torch.tensor(0.1, device=device)

    model = make_docking_model(top_prior_concentration, bottom_prior_concentration, ee50_prior_mu, ee50_prior_sd,
                               slope_prior_mu, slope_prior_sd)

    guide = PosteriorGuide(D, (num_parallel,)).to(device)
    targets = ["top", "bottom", "ee50", "slope"]
    eig_loss = _differentiable_ace_eig_loss(model, guide, 10, ["y"], targets)
    loss = neg_loss(eig_loss)
    optimizer = pyro.optim.Adam({"lr": 0.001})

    # Train guide
    print("Training")
    opt_eig_ape_loss(design, loss, num_samples=10, num_steps=25000, optim=optimizer)

    # Evaluate
    print("Evaluation")
    lower_loss = _ace_eig_loss(model, guide, num_inner_samples, "y", targets)  # isn't that an annoying API difference?
    upper_loss = _vnmc_eig_loss(model, guide, "y", targets)
    lower, upper = 0., 0.
    max_samples = 10000
    n_per_batch = max_samples // num_inner_samples
    n_batches = num_inner_samples ** 3 // (1 * max_samples)
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
    parser.add_argument("--num-inner-samples", default=500, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()

    main(args.name, args.num_inner_samples, args.device)
