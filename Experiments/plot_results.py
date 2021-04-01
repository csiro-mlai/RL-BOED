import numpy as np
import matplotlib.pyplot as plt

data = np.load("results.npz")
step = 1
err_bar = "std"
redline_x = np.array([100*i for i in range(20)])

means = -data['emeans'].astype(np.float64).reshape((-1,))
smoothed_means = np.asarray([means[i:i+step].mean() for i in range(means.size - step)])
fig = plt.figure(figsize=(8, 6))
fig.clear()
xlim, ylim = [-5,2000], [-25,10]
ax = fig.add_subplot(1,1,1, xlim=xlim, ylim=ylim)
ax.plot(np.arange(0, smoothed_means.size), smoothed_means)
ax.vlines(redline_x, ymin=ylim[0], ymax=ylim[1], colors=["red"], linestyles='dashed')
if err_bar == "std":
	stds = data['estds'].astype(np.float64).reshape((-1,))
	smoothed_stds = np.asarray([stds[i:i+step].mean() for i in range(stds.size - step)])
	ax.fill_between(np.arange(0, smoothed_means.size),
		smoothed_means + smoothed_stds,
		smoothed_means - smoothed_stds, alpha=0.5
	)
ax.set_xlabel("Iterations")
ax.set_ylabel("Posterior Entropy")
ax.set_title("CES Evaluation")
plt.grid(True)
fig.savefig("ces_evaluation.png")



means = -data['rmeans'].astype(np.float64).reshape((-1,))
smoothed_means = np.asarray([means[i:i+step].mean() for i in range(means.size - step)])
fig = plt.figure(figsize=(8, 6))
fig.clear()
# xlim, ylim = [-5,1000], [-25,10]
ax = fig.add_subplot(1,1,1, xlim=xlim, ylim=ylim)
ax.plot(np.arange(0, smoothed_means.size), smoothed_means)
ax.vlines(redline_x, ymin=ylim[0], ymax=ylim[1], colors=["red"], linestyles='dashed')
if err_bar == "std":
	stds = data['rstds'].astype(np.float64).reshape((-1,))
	smoothed_stds = np.asarray([stds[i:i+step].mean() for i in range(stds.size - step)])
	ax.fill_between(np.arange(0, smoothed_means.size),
		smoothed_means + smoothed_stds,
		smoothed_means - smoothed_stds, alpha=0.5
	)
ax.set_xlabel("Iterations")
ax.set_ylabel("Posterior Entropy")
ax.set_title("CES Training")
plt.grid(True)
fig.savefig("ces_returns.png")