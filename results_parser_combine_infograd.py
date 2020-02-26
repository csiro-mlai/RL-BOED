import argparse
import pickle
import math
import numpy as np

import torch
import matplotlib.pyplot as plt

output_dir = "./run_outputs/gradinfo/"


COLOURS = {"ace": 1, "ace-nrb":0, "pce": 3, "pce-nrb":2, "posterior": 5, "posterior-nrb":4, "bo-pce": 7}
MARKERS = {"ace": "x", "pce": "|", "posterior": "1", "bo-pce": ".", "ace-nrb": "x", "pce-nrb": "|", "posterior-nrb": "1"}
LEGENDS = {"ace": "ACE", "pce": "PCE", "posterior": "BA", "bo-pce": "BO+NMC",
           "ace-nrb": "ACE without RB", "pce-nrb": "PCE without RB", "posterior-nrb": "BA without RB"}


def main(names, sampling_interval):

    combined = {}
    for name in names.split(","):
        fname = output_dir + name + ".pickle"
        with open(fname, 'rb') as f:
            results = pickle.load(f)
        combined[name] = results

    legend = []
    plt.figure(figsize=(5, 3.5))
    for name in names.split(","):
        print(name, combined[name]['eig_history'].shape, combined[name]['seed'], combined[name].get("num_samples", 0))
        wall_time = combined[name]['wall_times'].detach().numpy()[::sampling_interval]
        hist = combined[name]['eig_history'].detach().numpy()[::sampling_interval]
        mean, se = np.nanmean(hist, 1), np.nanstd(hist, 1)/math.sqrt(hist.shape[1])
        e = combined[name]["estimator"] + ('-nrb' if 'nrb' in name else '')
        cmap = plt.get_cmap("Paired")
        col = cmap(COLOURS[e])
        marker = MARKERS[e]
        text = LEGENDS[e]
        plt.plot(wall_time, mean, color=col, marker=marker, markersize=7)
        plt.fill_between(wall_time, mean - se, mean + se, alpha=0.15, color=col)

        legend.extend([text])
    plt.ylim([0.91, 0.99])
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("EIG", fontsize=18)
    plt.legend(legend, fontsize=14, frameon=False)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

    for name in names.split(","):
        print(name, combined[name]['eig_history'][-1,...].mean(),
              combined[name]['eig_history'][-1,...].std()/math.sqrt(combined[name]['eig_history'].shape[1]),
              combined[name]['wall_times'][-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result parser for design optimization (one shot)")
    parser.add_argument("--names", default="", type=str)
    parser.add_argument("--sampling-interval", default=1, type=int)
    args = parser.parse_args()

    main(args.names, args.sampling_interval)
