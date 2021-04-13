import numpy as np

from akro.box import Box


class BatchBox(Box):
    """
    A dynamic-size batch of (possibly unbounded) boxes in R^n. It assumes an
    implicit batch dimension that may be 0.

    There are two common use cases:

    * Identical bound for each dimension::
        >>> BatchBox(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        BatchBox(3, 4)

    * Independent bound for each dimension::
        >>> BatchBox(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        BatchBox(2,)

    """

    def sample(self):
        sample = super(BatchBox, self).sample()
        return sample.reshape((1,) + self.shape)

    def contains(self, x):
        if isinstance(x, list):
            x = np.array(x)  # Promote list to array for contains check
        # in case batch is empty, only test that shapes match
        if len(x) == 0:
            return x.shape[1:] == self.shape

        # test shape of first entry in batch
        x = x[0]
        return x.shape == self.shape and \
               np.all(x >= self.low) and \
               np.all(x <= self.high)

    def unflatten(self, x):
        return np.asarray(x).reshape((-1,) + self.shape)

    def unflatten_n(self, obs):
        return np.asarray(obs).reshape((len(obs), -1) + self.shape)

    def concat(self, other):
        assert isinstance(other, BatchBox)

        first_lb, first_ub = self.bounds
        second_lb, second_ub = other.bounds
        first_lb, first_ub = first_lb.flatten(), first_ub.flatten()
        second_lb, second_ub = second_lb.flatten(), second_ub.flatten()
        return BatchBox(np.concatenate([first_lb, second_lb]),
                        np.concatenate([first_ub, second_ub]))

    def __repr__(self):
        return "BatchBox" + str(self.shape)

    def __eq__(self, other):
        return isinstance(other, BatchBox) and \
               (self.shape == other.shape) and \
               np.allclose(self.low, other.low) and \
               np.allclose(self.high, other.high)
