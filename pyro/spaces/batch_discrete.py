import torch
import numpy as np

from akro import Discrete


class BatchDiscrete(Discrete):
    """
    A dynamic-size batch of vectors in Z. It assumes an
    implicit batch dimension that may be 0.
    """
    def __init__(self, n, floor=0, shape=()):
        super(BatchDiscrete, self).__init__(n)
        assert floor < n
        self.floor = floor
        self.shape = shape

    def sample(self, shape=(1,)):
        return torch.randint(low=self.floor, high=self.n + 1, size=shape)

    def contains(self, x):
        if isinstance(x, list):
            x = torch.as_tensor(x)  # Promote list to array for contains check
        return torch.all(x >= self.floor) and torch.all(x <= self.n)

    def unflatten(self, x):
        return torch.as_tensor(x).reshape((-1, 1))

    def unflatten_n(self, obs):
        return torch.as_tensor(obs).reshape((len(obs), -1, 1))

    # def flatten_n(self, xs):
    #     ret = np.zeros()

    def concat(self, other):
        raise NotImplementedError

    def __repr__(self):
        return f"BatchDiscrete({self.n})"

    def __eq__(self, other):
        return isinstance(other, BatchDiscrete) and self.n == other.n
