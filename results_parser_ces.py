from __future__ import absolute_import, division, print_function

import argparse
import pickle
import glob
from collections import defaultdict, OrderedDict

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

output_dir = "./run_outputs/ces/"
cmap = plt.get_cmap("Paired")
COLOURS = {'ace-grad': cmap(1),
           'nce-grad': cmap(3),
           'posterior-grad': cmap(5),
           'marginal': cmap(7)}
VALUE_LABELS = {"Entropy": "Posterior entropy",
                "L2 distance": "Expected L2 distance from posterior to truth",
                "Optimized EIG": "Maximized EIG",
                "EIG gap": "Difference between maximum and mean EIG",
                "rho_rmse": "RMSE in $\\rho$ estimate",
                "alpha_rmse": "RMSE in $\\mathbf{\\alpha}$ estimate",
                "slope_rmse": 'RMSE in $u$ estimate',
                "total_rmse": 'Total RMSE',
                "Imax": "EIG lower bound"}
LABELS = {'marginal': 'BO + marginal (baseline)', 'rand': 'Random design (baseline)', 'nmc': 'BOED NMC (baseline)',
          'posterior-grad': "BA gradient", 'nce-grad': "PCE gradient", "ace-grad": "ACE gradient"}
MARKERS = {'ace-grad': 'x',
           'nce-grad': '|',
           'posterior-grad': '1',
           'marginal': '.'}

S = 3


def upper_lower(array, percentile=False):
    if percentile:
        return np.percentile(array, 75, axis=1), np.percentile(array, 50, axis=1), np.percentile(array, 25, axis=1)
    else:
        centre = array.mean(1)
        se = array.std(1)/np.sqrt(array.shape[1])
        upper, lower = centre + se, centre - se
        return lower, centre, upper


def rlogdet(M):
    old_shape = M.shape[:-2]
    tbound = M.view(-1, M.shape[-2], M.shape[-1])
    ubound = torch.unbind(tbound, dim=0)
    logdets = [torch.logdet(m) for m in ubound]
    bound = torch.stack(logdets)
    return bound.view(old_shape)


def rtrace(M):
    old_shape = M.shape[:-2]
    tbound = M.view(-1, M.shape[-2], M.shape[-1])
    ubound = torch.unbind(tbound, dim=0)
    traces = [torch.trace(m) for m in ubound]
    bound = torch.stack(traces)
    return bound.view(old_shape)


def main(fnames, findices, plot, percentile):
    fnames = fnames.split(",")
    findices = map(int, findices.split(","))

    if not all(fnames):
        results_fnames = sorted(glob.glob(output_dir+"*.result_stream.pickle"))
        fnames = [results_fnames[i] for i in findices]
    else:
        fnames = [output_dir+name+".result_stream.pickle" for name in fnames]

    if not fnames:
        raise ValueError("No matching files found")

    results_dict = defaultdict(list)
    for fname in fnames:
        with open(fname, 'rb') as results_file:
            try:
                while True:
                    results = pickle.load(results_file)
                    print(fname, results.get('num_gradient_steps', 0), results.get('num_samples', 0), results.get('num_contrast_samples', 0))
                    # Compute entropy and L2 distance to the true fixed effects
                    if 'rho0' in results:
                        rho0, rho1, alpha_concentration = results['rho0'], results['rho1'], results['alpha_concentration']
                    else:
                        rho_concentration = results['rho_concentration']
                        rho0, rho1 = rho_concentration.unbind(-1)
                        alpha_concentration = results['alpha_concentration']
                    slope_mu, slope_sigma = results['slope_mu'], results['slope_sigma']
                    rho_dist = torch.distributions.Beta(rho0, rho1)
                    alpha_dist = torch.distributions.Dirichlet(alpha_concentration)
                    slope_dist = torch.distributions.LogNormal(slope_mu, slope_sigma)
                    rho_rmse = torch.sqrt((rho_dist.mean - torch.tensor(.9))**2 + rho_dist.variance)
                    alpha_rmse = torch.sqrt((alpha_dist.mean - torch.tensor([.2, .3, .5])).pow(2).sum(-1))
                    slope_rmse = torch.sqrt((slope_dist.mean - torch.tensor(10.)).pow(2) + slope_dist.variance)
                    total_rmse = torch.sqrt(rho_rmse**2 + alpha_rmse**2 + slope_rmse**2)
                    entropy = rho_dist.entropy() + alpha_dist.entropy() + slope_dist.entropy()
                    try:
                        eig = -results['min loss']
                    except:
                        eig = results['max EIG']

                    output = {"rho_rmse": rho_rmse, "alpha_rmse": alpha_rmse, "slope_rmse": slope_rmse,
                              "Entropy": entropy, "total_rmse": total_rmse, 'Imax': eig}
                    results_dict[results['typ']].append(output)
            except EOFError:
                continue

    # Get results into better format
    # First, concat across runs
    possible_stats = list(set().union(a for v in results_dict.values() for a in v[0].keys()))
    reformed = {statistic: {
        k: torch.stack([a[statistic] for a in v]).detach().numpy()
        for k, v in results_dict.items() if statistic in v[0]}
        for statistic in possible_stats}

    if plot:
        for statistic in ["Entropy", "rho_rmse", "alpha_rmse", "slope_rmse", "Imax"]:
            plt.figure(figsize=(5, 5))
            for i, k in enumerate(reformed[statistic]):
                e = reformed[statistic][k].squeeze()[1:]
                lower, centre, upper = upper_lower(e, percentile=percentile)
                x = np.arange(2, e.shape[0]+2)
                plt.plot(x, centre, linestyle='-', markersize=12, color=COLOURS[k], marker=MARKERS[k], linewidth=1.5)
                plt.fill_between(x, upper, lower, color=COLOURS[k], alpha=0.15)
            plt.xlabel("Step", fontsize=23)
            plt.xticks([5, 10, 15, 20], fontsize=23)

            plt.yticks(fontsize=23)
            plt.ylabel(VALUE_LABELS[statistic], fontsize=23)

            if statistic not in ["Entropy", "Imax"]:
                plt.yscale('log')
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid iterated experiment design results parser")
    parser.add_argument("--fnames", nargs="?", default="", type=str)
    parser.add_argument("--findices", nargs="?", default="-1", type=str)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(plot=True)
    percentile_parser = parser.add_mutually_exclusive_group(required=False)
    percentile_parser.add_argument("--percentile", dest='percentile', action='store_true')
    percentile_parser.add_argument("--no-percentile", dest='percentile', action='store_false')
    parser.set_defaults(percentile=False)
    args = parser.parse_args()
    main(args.fnames, args.findices, args.plot, args.percentile)
