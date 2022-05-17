"""AdaptiveTanhGaussianPolicy."""
import akro
import numpy as np
import torch

from garage.torch import global_device
from garage.torch.distributions import TanhNormal
from torch.distributions import Normal
from pyro.modules import GaussianMLPTwoHeadedModule, MLPModule
from pyro.policies import AdaptiveTanhGaussianPolicy
from torch import nn


class AdaptiveToyPolicy(AdaptiveTanhGaussianPolicy):
    def forward(self, observations, mask=None):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.
            mask (torch.Tensor): a mask to account for 0-padded inputs

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        encoding = self._encoder(observations)
        if mask is not None:
            encoding = encoding * mask
        pooled_encoding = encoding.sum(dim=-2)
        dist = self._emitter(pooled_encoding)
        ret_mean = dist.mean.clone()
        ret_log_std = (dist.variance.sqrt()).log().clone()
        if not mask.any():
            ret_mean = 0 * ret_mean - 0.1
            ret_log_std = ret_log_std * 0 - 1e9
            dist = TanhNormal(ret_mean, ret_log_std.exp())
        return dist, dict(mean=ret_mean, log_std=ret_log_std)
