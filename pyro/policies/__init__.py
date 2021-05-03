"""Garage wrappers for gym environments."""

from pyro.policies.adaptive_tanh_gaussian_policy import AdaptiveTanhGaussianPolicy
from pyro.policies.reproducing_policy import ReproducingPolicy

__all__ = [
    'AdaptiveTanhGaussianPolicy',
    'ReproducingPolicy',
]
