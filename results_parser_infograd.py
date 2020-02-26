import argparse
import pickle
import numpy as np
import math

import torch
import matplotlib.pyplot as plt

output_dir = "./run_outputs/gradinfo/"


def main(name, sampling_interval, summary):

    fname = output_dir + name + ".pickle"
    with open(fname, 'rb') as f:
        results = pickle.load(f)

    if summary:
        print(results['final_lower_bound'].mean().item(), "+/-", results['final_lower_bound'].std().item()/math.sqrt(10))
        print(results['final_upper_bound'].mean().item(), "+/-", results['final_upper_bound'].std().item()/math.sqrt(10))
        return
    xi_history = results['xi_history']
    design = xi_history[-1, 0, ...]
    design = design / design.norm(p=1, dim=-1, keepdim=True)
    eig_history = results.get('eig_history')
    eig_heatmap = results.get('eig_heatmap')
    heatmap_extent = results.get('extent')
    eig_lower = results.get('lower_history')
    eig_upper = results.get('upper_history')

    if xi_history.shape[-1] <= 2:
        plt.figure(figsize=(5, 3.5))
        if eig_heatmap is not None:
            plt.imshow(eig_heatmap, cmap="gray", extent=heatmap_extent, origin='lower')
        x, y = xi_history[::sampling_interval, 1, 0].detach(), xi_history[::sampling_interval, 1, 1].detach()
        plt.scatter(x, y, c=torch.arange(x.shape[0]), marker='x', cmap='cool', s=70)
        plt.xlabel("$\\xi_1$", fontsize=18)
        plt.ylabel("$\\xi_2$", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()
    elif xi_history.shape[-1] > 50:
        plt.hist(xi_history[-1, 0, ...].view(-1).numpy(), bins=20)
        plt.xlabel("Predicted binding affinity", fontsize=18)
        plt.ylabel("Number of compounds to test", fontsize=18)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.text(-74., 14.5, "{:.5} < EIG < {:.5}".format(results['final_lower_bound'][0], results['final_upper_bound'][0]), ha='center', fontsize=16)
        plt.show()
        return
    else:
        print(xi_history[-1, ...])

    if eig_upper is not None and eig_lower is not None:
        plt.plot(eig_lower.clamp(min=0, max=2).numpy())
        plt.plot(eig_upper.clamp(min=0, max=2).numpy())
        print("last upper", eig_upper[-1], "last lower", eig_lower[-1])

        plt.legend(["Lower bound", "Upper bound"])
        plt.show()

    plt.plot(est_eig_history.detach().clamp(min=0).numpy()[::sampling_interval])
    if eig_history is not None:
        plt.plot(eig_history.detach().numpy()[::sampling_interval])
        print("Final true EIG", eig_history[-1].item())
        if eig_heatmap is not None:
            print("Max EIG over surface", eig_heatmap.max().item())
            print("Discrepancy", (eig_heatmap.max() - eig_history[-1]).item())
        plt.legend(["Approximate EIG", "True EIG"])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Result parser for design optimization (one shot)")
    parser.add_argument("--name", default="", type=str)
    parser.add_argument("--sampling-interval", default=20, type=int)
    parser.add_argument("--summary", default=False, type=bool)
    args = parser.parse_args()

    main(args.name, args.sampling_interval, args.summary)
