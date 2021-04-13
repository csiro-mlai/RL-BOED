"""AdaptiveTanhGaussianPolicy."""
import numpy as np
from torch import nn

from garage.torch.distributions import TanhNormal
from garage.torch.modules import GaussianMLPTwoHeadedModule, MLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class AdaptiveTanhGaussianPolicy(StochasticPolicy):
    """Multiheaded MLP with an encoder and pooling operation.

    A policy that takes as input entire histories and maps them to a Gaussian
    distribution with a tanh transformation. Inputs to the network should be of
    the shape (batch_dim, history_length, obs_dim)

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
        init_std (float): Initial value for std.
            (plain value - not log or exponentiated).
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues (plain value - not log or exponentiated).
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
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
                 init_std=1.0,
                 min_std=np.exp(-20.),
                 max_std=np.exp(2.),
                 std_parameterization='exp',
                 layer_normalization=False):
        super().__init__(env_spec, name='AdaptiveTanhGaussianPolicy')

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

        self._emitter = GaussianMLPTwoHeadedModule(
            input_dim=encoding_dim,
            output_dim=self._action_dim,
            hidden_sizes=emitter_sizes,
            hidden_nonlinearity=emitter_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=emitter_output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            init_std=init_std,
            min_std=min_std,
            max_std=max_std,
            std_parameterization=std_parameterization,
            layer_normalization=layer_normalization,
            normal_distribution_cls=TanhNormal)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        encoding = self._encoder(observations)
        pooled_encoding = encoding.sum(dim=-2)
        dist = self._emitter(pooled_encoding)
        ret_mean = dist.mean.cpu()
        ret_log_std = (dist.variance.sqrt()).log().cpu()
        return dist, dict(mean=ret_mean, log_std=ret_log_std)
