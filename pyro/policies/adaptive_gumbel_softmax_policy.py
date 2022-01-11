"""AdaptiveGumbelSoftmaxPolicy."""
import akro
import numpy as np
import torch

from garage.torch import global_device
from pyro.policies.gumbelsoftmax_mlp_module import \
    GumbelSoftmaxMLPTwoHeadedModule
from garage.torch.modules import MLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy
from torch import nn


class AdaptiveGumbelSoftmaxPolicy(StochasticPolicy):
    """Multiheaded MLP with an encoder and pooling operation.

    A policy that takes as input entire histories and maps them to a
    GumbelSoftmax distribution. Inputs to the network should be of
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
        init_temp (float): Initial value for temp.
            (plain value - not log or exponentiated).
        min_temp (float): If not None, the temp is at least the value of min_temp,
            to avoid numerical issues (plain value - not log or exponentiated).
        max_temp (float): If not None, the temp is at most the value of max_temp,
            to avoid numerical issues (plain value - not log or exponentiated).
        temp_parameterization (str): How the temp should be parametrized. There
            are two options:
            - exp: the logarithm of the temp will be stored, and applied a
               exponential transformation
            - softplus: the temp will be computed as log(1+exp(x))
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
                 init_temp=1.0,
                 learn_temp=True,
                 min_temp=0.01,
                 max_temp=10.,
                 temp_parameterization='exp',
                 layer_normalization=False):
        super().__init__(env_spec, name='AdaptiveGumbelSoftmaxPolicy')

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

        self._emitter = GumbelSoftmaxMLPTwoHeadedModule(
            input_dim=encoding_dim,
            output_dim=self._action_dim,
            hidden_sizes=emitter_sizes,
            hidden_nonlinearity=emitter_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=emitter_output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            learn_temp=learn_temp,
            init_temp=init_temp,
            min_temp=min_temp,
            max_temp=max_temp,
            temp_parameterization=temp_parameterization,
            layer_normalization=layer_normalization)

    def get_actions(self, observations, mask=None):
        r"""Get actions given observations.

        Args:
            observations (np.ndarray): Observations from the environment.
                Shape is :math:`batch_dim \bullet env_spec.observation_space`.

        Returns:
            tuple:
                * np.ndarray: Predicted actions.
                    :math:`batch_dim \bullet env_spec.action_space`.
                * dict:
                    * np.ndarray[float]: Logits of the distribution.
                    * np.ndarray[float]: Logarithm temperature of the
                        distribution.

        """
        if not isinstance(observations[0], np.ndarray) and not isinstance(
                observations[0], torch.Tensor):
            observations = self._env_spec.observation_space.flatten_n(
                observations)

        # frequently users like to pass lists of torch tensors or lists of
        # numpy arrays. This handles those conversions.
        if isinstance(observations, list):
            if isinstance(observations[0], np.ndarray):
                observations = np.stack(observations)
            elif isinstance(observations[0], torch.Tensor):
                observations = torch.stack(observations)

        if isinstance(self._env_spec.observation_space, akro.Image) and \
                len(observations.shape) < \
                len(self._env_spec.observation_space.shape):
            observations = self._env_spec.observation_space.unflatten_n(
                observations)
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(
                    global_device())
                if mask is not None:
                    mask = torch.as_tensor(mask).float().to(
                        global_device())
            dist, info = self.forward(observations, mask)
            return dist.sample().detach().unsqueeze(dim=-1), {
                k: v.detach()
                for (k, v) in info.items()
            }

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
        ret_logits = dist.logits.clone()
        ret_log_temp = dist.temperature.log().clone()
        return dist, dict(logits=ret_logits, log_temp=ret_log_temp)
