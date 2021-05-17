import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 15
plt.rcParams['figure.titlesize'] = 36

fig_dpi = 100
width, height = (16,12)
titlesize = 26


data = np.load("results.npz")
step = 10
L = 1e5
err_bar = "std"
title_template = " 20-step CES"
redline_x = np.array([100*i for i in range(20)])

typ = "emeans" if err_bar == "std" else "emedians"
means = data[typ].astype(np.float64).reshape((-1,))
smoothed_means = np.asarray([means[i:i+step].mean() for i in range(means.size - step)])
fig = plt.figure(figsize=(width, height), dpi=fig_dpi)
fig.clear()
xlim, ylim = [-5,10000], [4,13]
ax = fig.add_subplot(1,1,1, xlim=xlim, ylim=ylim)
ax.plot(np.arange(0, smoothed_means.size), smoothed_means)
# ax.vlines(redline_x, ymin=ylim[0], ymax=ylim[1], colors=["red"], linestyles='dashed')
ax.hlines(np.log(L+1), xmin=xlim[0], xmax=xlim[1], colors=["black"], linestyles='dashed')
if err_bar == "std":
	stds = data['estds'].astype(np.float64).reshape((-1,))
	smoothed_stds = np.asarray([stds[i:i+step].mean() for i in range(stds.size - step)])
	upper_bound = smoothed_means + smoothed_stds
	lower_bound = smoothed_means - smoothed_stds
elif err_bar == "iqr":
	uqs = data['ruqs'].astype(np.float64).reshape((-1,))
	smoothed_uqs = np.asarray([uqs[i:i+step].mean() for i in range(uqs.size - step)])
	lqs = data['rlqs'].astype(np.float64).reshape((-1,))
	smoothed_lqs = np.asarray([lqs[i:i+step].mean() for i in range(lqs.size - step)])
	upper_bound, lower_bound = smoothed_uqs, smoothed_lqs
ax.fill_between(
	np.arange(0, smoothed_means.size), upper_bound, lower_bound, alpha=0.5
)
ax.set_xlabel("Iterations")
ax.set_ylabel("sPCE")
ax.set_title(f"{title_template} Evaluation ({err_bar})", fontsize=titlesize)
plt.grid(True)
fig.tight_layout()
fig.savefig("ces_evaluation.png")



typ = "rmeans" if err_bar == "std" else "rmedians"
means = data[typ].astype(np.float64).reshape((-1,))
smoothed_means = np.asarray([means[i:i+step].mean() for i in range(means.size - step)])
fig = plt.figure(figsize=(width, height), dpi=fig_dpi)
fig.clear()
# xlim, ylim = [-5,1000], [-25,10]
ax = fig.add_subplot(1,1,1, xlim=xlim, ylim=ylim)
ax.plot(np.arange(0, smoothed_means.size), smoothed_means)
# ax.vlines(redline_x, ymin=ylim[0], ymax=ylim[1], colors=["red"], linestyles='dashed')
ax.hlines(np.log(L+1), xmin=xlim[0], xmax=xlim[1], colors=["black"], linestyles='dashed')
if err_bar == "std":
	stds = data['rstds'].astype(np.float64).reshape((-1,))
	smoothed_stds = np.asarray([stds[i:i+step].mean() for i in range(stds.size - step)])
	upper_bound = smoothed_means + smoothed_stds
	lower_bound = smoothed_means - smoothed_stds
elif err_bar == "iqr":
	uqs = data['ruqs'].astype(np.float64).reshape((-1,))
	smoothed_uqs = np.asarray([uqs[i:i+step].mean() for i in range(uqs.size - step)])
	lqs = data['rlqs'].astype(np.float64).reshape((-1,))
	smoothed_lqs = np.asarray([lqs[i:i+step].mean() for i in range(lqs.size - step)])
	upper_bound, lower_bound = smoothed_uqs, smoothed_lqs
ax.fill_between(
	np.arange(0, smoothed_means.size), upper_bound, lower_bound, alpha=0.5
)
ax.set_xlabel("Iterations")
ax.set_ylabel("sPCE")
ax.set_title(f"{title_template} Training ({err_bar})", fontsize=titlesize)
plt.grid(True)
fig.tight_layout()

fig.savefig("ces_returns.png")