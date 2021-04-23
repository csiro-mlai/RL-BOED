"""This modules creates a continuous Q-function network."""

import torch

from garage.torch.modules import MLPModule
from torch import nn


class AdaptiveMLPQFunction(nn.Module):
    """
    Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the history of experiments.
    It uses a PyTorch neural network module to fit the function of Q(s, a).
    Inputs to the encoder should be of the shape
    (batch_dim, history_length, obs_dim)

    Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            encoder_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for encoder. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        encoder_nonlinearity (callable): Activation function for intermediate
            dense layer(s) of encoder. It should return a torch.Tensor. Set it
            to None to maintain a linear activation.
        encoder_output_nonlinearity (callable): Activation function for encoder
            output dense layer. It should return a torch.Tensor. Set it to None
            to maintain a linear activation.
        encoding_dim (int): Output dimension of output dense layer for encoder.
        emitter_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for emitter.
        emitter_nonlinearity (callable): Activation function for intermediate
            dense layer(s) of emitter.
        emitter_output_nonlinearity (callable): Activation function for emitter
            output dense layer.
    """

    def __init__(self,
                 env_spec,
                 encoder_sizes=(32, 32),
                 encoder_nonlinearity=nn.ReLU,
                 encoder_output_nonlinearity=None,
                 encoding_dim=16,
                 emitter_sizes=(32, 32),
                 emitter_nonlinearity=nn.ReLU,
                 emitter_output_nonlinearity=None,
                 **kwargs):
        super().__init__()
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self._encoder = MLPModule(
            input_dim=self._obs_dim,
            output_dim=encoding_dim,
            hidden_sizes=encoder_sizes,
            hidden_nonlinearity=encoder_nonlinearity,
            output_nonlinearity=encoder_output_nonlinearity,
            **kwargs
        )

        self._emitter = MLPModule(
            input_dim=encoding_dim + self._action_dim,
            output_dim=1,
            hidden_sizes=emitter_sizes,
            hidden_nonlinearity=emitter_nonlinearity,
            output_nonlinearity=emitter_output_nonlinearity,
            **kwargs
        )

    def forward(self, observations, actions, mask=None):
        """Return Q-value(s)."""
        encoding = self._encoder.forward(observations)
        if mask is not None:
            encoding = encoding * mask
        pooled_encoding = encoding.sum(dim=-2)
        return self._emitter.forward(torch.cat([pooled_encoding, actions], 1))
