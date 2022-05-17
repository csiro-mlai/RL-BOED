"""Discrete Dueling CNN Q Function."""
import torch
from torch import nn

from garage import InOutSpec
from garage.torch.modules import CNNModule, MLPModule


# pytorch v1.6 issue, see https://github.com/pytorch/pytorch/issues/42305
# pylint: disable=abstract-method
class AdaptiveDuelingQFunction(nn.Module):
    """Discrete Dueling CNN Q Function.

    A dueling Q network that estimates Q values of all possible discrete
    actions. It is constructed using an encoder architecture followed by
    fully-connected modules for both value and advantage estimation.

    Args:
        env_spec (EnvSpec): Environment specification.
        encoder_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for encoder. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        encoder_nonlinearity (callable): Activation function for
            intermediate dense layer(s) in the encoder. It should return
            a torch.Tensor. Set it to None to maintain a linear activation.
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
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
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
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False):
        super().__init__()

        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        self._encoder = MLPModule(
            input_dim=self._obs_dim,
            output_dim=encoding_dim,
            hidden_sizes=encoder_sizes,
            hidden_nonlinearity=encoder_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=encoder_output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

        self._val = MLPModule(
            input_dim=encoding_dim,
            output_dim=self._action_dim,
            hidden_sizes=emitter_sizes,
            hidden_nonlinearity=emitter_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=emitter_output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)
        self._act = MLPModule(
            input_dim=encoding_dim,
            output_dim=self._action_dim,
            hidden_sizes=emitter_sizes,
            hidden_nonlinearity=emitter_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=emitter_output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

    # pylint: disable=arguments-differ
    def forward(self, observations, mask=None):
        """Return Q-value(s).

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.
            mask (torch.Tensor): a mask to account for 0-padded inputs

        Returns:
            torch.Tensor: Output value
        """
        encoding = self._encoder(observations)
        if mask is not None:
            encoding = encoding * mask
        encoding = encoding.sum(dim=-2)
        val = self._val(encoding)
        act = self._act(encoding)
        act = act - act.mean(1).unsqueeze(1)
        return val + act
