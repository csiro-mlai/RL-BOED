import torch

from garage.torch.distributions import TanhNormal


class OffsetTanhNormal(TanhNormal):
    """Like TanhNormal but with an offset on the input so that the mean of a
    randomly initialised policy can be non-zero
    :math:`Y \sim \mathcal{N}(\mu, \sigma)`
    :math:`X = tanh(Y + C)`

    Args:
        loc (torch.Tensor): The mean of the normal.
        scale (torch.Tensor): the stdev of the normal.
        offset (Float): the offset C.
    """

    def __init__(self, loc, scale, offset=0):
        super().__init__(loc, scale)
        self.offset = offset

    def log_prob(self, value, pre_tanh_value=None, epsilon=1e-6):
        """The log likelihood of a sample on the this Tanh Distribution.

        Args:
            value (torch.Tensor): The sample whose loglikelihood is being
                computed.
            pre_tanh_value (torch.Tensor): The value prior to having the tanh
                and offset applied to it but after it has been sampled from the
                normal distribution.
            epsilon (float): Regularization constant. Making this value larger
                makes the computation more stable but less precise.

        Note:
              when pre_tanh_value is None, an estimate is made of what the
              value is. This leads to a worse estimation of the log_prob.
              If the value being used is collected from functions like
              `sample` and `rsample`, one can instead use functions like
              `sample_return_pre_tanh_value` or
              `rsample_return_pre_tanh_value`


        Returns:
            torch.Tensor: The log likelihood of value on the distribution.

        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1 + epsilon + value) / (1 + epsilon - value)) / 2 - self.offset
        norm_lp = self._normal.log_prob(pre_tanh_value)
        ret = (norm_lp - torch.sum(
            torch.log(self._clip_but_pass_gradient((1. - value ** 2)) + epsilon),
            axis=-1))
        return ret

    def rsample(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this OffsetTanhNormal Distribution.

        Args:
            sample_shape (list): Shape of the returned value.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Sample from this OffsetTanhNormal distribution.

        """
        z = self._normal.rsample(sample_shape)
        return torch.tanh(z + self.offset)

    def rsample_with_pre_tanh_value(self, sample_shape=torch.Size()):
        """Return a sample, sampled from this OffsetTanhNormal distribution.

        Returns the sampled value before the transform is applied and the
        sampled value with the transform applied to it.

        Args:
            sample_shape (list): shape of the return.

        Note:
            Gradients pass through this operation.

        Returns:
            torch.Tensor: Samples from this distribution.
            torch.Tensor: Samples from the underlying
                :obj:`torch.distributions.Normal` distribution, prior to being
                transformed.

        """
        z = self._normal.rsample(sample_shape)
        return z, torch.tanh(z + self.offset)

    @classmethod
    def _from_distribution(cls, new_normal, offset=0):
        """Construct a new TanhNormal distribution from a normal distribution.

        Args:
            new_normal (Independent(Normal)): underlying normal dist for
                the new OffsetTanhNormal distribution.
            offset (Float): constant offset for the new OffsetTanhNormal

        Returns:
            OffsetTanhNormal: A new distribution whose underlying normal dist
                is new_normal.

        """
        # pylint: disable=protected-access
        new = cls(torch.zeros(1), torch.zeros(1), offset)
        new._normal = new_normal
        return new

    def expand(self, batch_shape, _instance=None):
        """Returns a new OffsetTanhNormal distribution.

        (or populates an existing instance provided by a derived class) with
        batch dimensions expanded to `batch_shape`. This method calls
        :class:`~torch.Tensor.expand` on the distribution's parameters. As
        such, this does not allocate new memory for the expanded distribution
        instance. Additionally, this does not repeat any args checking or
        parameter broadcasting in `__init__.py`, when an instance is first
        created.

        Args:
            batch_shape (torch.Size): the desired expanded size.
            _instance(instance): new instance provided by subclasses that
                need to override `.expand`.

        Returns:
            Instance: New distribution instance with batch dimensions expanded
            to `batch_size`.

        """
        new_normal = self._normal.expand(batch_shape, _instance)
        new = self._from_distribution(new_normal, self.offset)
        return new

    @property
    def mean(self):
        """torch.Tensor: mean of the distribution."""
        return torch.tanh(self._normal.mean + self.offset)
