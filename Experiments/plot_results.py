import numpy as np
import matplotlib.pyplot as plt

data = np.load("results.npz")
step = 10

rmedians = data['rmedians'].astype(np.float64).reshape((-1,))
smoothed_rmedians = np.asarray([rmedians[i:i+step].mean() for i in range(rmedians.size - step)])
plt.plot(smoothed_rmedians)
plt.show()