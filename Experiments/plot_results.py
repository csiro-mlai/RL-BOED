import numpy as np
import matplotlib.pyplot as plt

data = np.load("results.npz")
step = 1

rmeans = -data['emeans'].astype(np.float64).reshape((-1,))
rstds = data['estds'].astype(np.float64).reshape((-1,))
smoothed_rmeans = np.asarray([rmeans[i:i+step].mean() for i in range(rmeans.size - step)])
smoothed_rstds = np.asarray([rstds[i:i+step].mean() for i in range(rstds.size - step)])
fig = plt.figure(figsize=(8, 6))
fig.clear()
# xlim, ylim = [-5,600], [-3,0]
ax = fig.add_subplot(1,1,1, )#xlim=xlim, ylim=ylim)
ax.plot(np.arange(0, smoothed_rmeans.size), smoothed_rmeans)
ax.fill_between(np.arange(0, smoothed_rmeans.size), smoothed_rmeans + smoothed_rstds, smoothed_rmeans - smoothed_rstds, alpha=0.5)
ax.set_xlabel("Iterations")
ax.set_ylabel("Posterior Entropy")
ax.set_title("Constant Elasticity of Substitution")
plt.grid(True)
fig.savefig("ces_evaluation.png")



rmeans = -data['rmeans'].astype(np.float64).reshape((-1,))
rstds = data['rstds'].astype(np.float64).reshape((-1,))
smoothed_rmeans = np.asarray([rmeans[i:i+step].mean() for i in range(rmeans.size - step)])
smoothed_rstds = np.asarray([rstds[i:i+step].mean() for i in range(rstds.size - step)])
fig = plt.figure(figsize=(8, 6))
fig.clear()
# xlim, ylim = [-5,600], [-3,0]
ax = fig.add_subplot(1,1,1, )#xlim=xlim, ylim=ylim)
ax.plot(np.arange(0, smoothed_rmeans.size), smoothed_rmeans)
ax.fill_between(np.arange(0, smoothed_rmeans.size), smoothed_rmeans + smoothed_rstds, smoothed_rmeans - smoothed_rstds, alpha=0.5)
ax.set_xlabel("Iterations")
ax.set_ylabel("Posterior Entropy")
ax.set_title("Constant Elasticity of Substitution")
plt.grid(True)
fig.savefig("ces_returns.png")