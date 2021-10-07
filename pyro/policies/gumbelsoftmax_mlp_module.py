"""GumbelSoftmaxMLPModule."""
import abc

import torch
from torch import nn

from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule

from pyro.policies.GumbelSoftmax import GumbelSoftmax


class GumbelSoftmaxMLPBaseModule(nn.Module):
    """Base of GumbelSoftmaxMLPModule.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for logits. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_temp (bool): Is temp trainable.
        init_temp (float): Initial value for temp.
            (plain value - not log or exponentiated).
        temp_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for temp. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        min_temp (float): If not None, the temp is at least the value of
            min_temp, to avoid numerical issues.
        max_temp (float): If not None, the temp is at most the value of
            max_temp, to avoid numerical issues.
        temp_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the temp network.
        temp_hidden_w_init (callable):  Initializer function for the weight
            of hidden layer (s).
        temp_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s).
        temp_output_nonlinearity (callable): Activation function for output
            dense layer in the temp network. It should return a torch.Tensor.
            Set it to None to maintain a linear activation.
        temp_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the temp network.
        temp_parameterization (str): How the temp should be parametrized. There
            are two options:
            - exp: the logarithm of the temp will be stored, and applied a
               exponential transformation.
            - softplus: the temp will be computed as log(1+exp(x)).
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_temp=True,
                 init_temp=1.0,
                 min_temp=1e-6,
                 max_temp=None,
                 temp_hidden_sizes=(32, 32),
                 temp_hidden_nonlinearity=torch.tanh,
                 temp_hidden_w_init=nn.init.xavier_uniform_,
                 temp_hidden_b_init=nn.init.zeros_,
                 temp_output_nonlinearity=None,
                 temp_output_w_init=nn.init.xavier_uniform_,
                 temp_parameterization='exp',
                 layer_normalization=False):
        super().__init__()

        self._input_dim = input_dim
        self._hidden_sizes = hidden_sizes
        self._action_dim = output_dim
        self._learn_temp = learn_temp
        self._temp_hidden_sizes = temp_hidden_sizes
        self._min_temp = min_temp
        self._max_temp = max_temp
        self._temp_hidden_nonlinearity = temp_hidden_nonlinearity
        self._temp_hidden_w_init = temp_hidden_w_init
        self._temp_hidden_b_init = temp_hidden_b_init
        self._temp_output_nonlinearity = temp_output_nonlinearity
        self._temp_output_w_init = temp_output_w_init
        self._temp_parameterization = temp_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

        if self._temp_parameterization not in ('exp', 'softplus'):
            raise NotImplementedError

        init_temp_param = torch.Tensor([init_temp]).log()
        if self._learn_temp:
            self._init_temp = torch.nn.Parameter(init_temp_param)
        else:
            self._init_temp = init_temp_param
            self.register_buffer('init_temp', self._init_temp)

        self._min_temp_param = self._max_temp_param = None
        if min_temp is not None:
            self._min_temp_param = torch.Tensor([min_temp]).log()
            self.register_buffer('min_temp_param', self._min_temp_param)
        if max_temp is not None:
            self._max_temp_param = torch.Tensor([max_temp]).log()
            self.register_buffer('max_temp_param', self._max_temp_param)

    def to(self, *args, **kwargs):
        """Move the module to the specified device.

        Args:
            *args: args to pytorch to function.
            **kwargs: keyword args to pytorch to function.

        """
        super().to(*args, **kwargs)
        buffers = dict(self.named_buffers())
        if not isinstance(self._init_temp, torch.nn.Parameter):
            self._init_temp = buffers['init_temp']
        self._min_temp_param = buffers['min_temp_param']
        self._max_temp_param = buffers['max_temp_param']

    @abc.abstractmethod
    def _get_temp_and_logits(self, *inputs):
        pass

    def forward(self, *inputs):
        """Forward method.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.distributions.independent.Independent: Independent
                distribution.

        """
        log_temp, logits = self._get_temp_and_logits(*inputs)

        if self._min_temp_param or self._max_temp_param:
            log_temp = log_temp.clamp(
                min=(None if self._min_temp_param is None else
                     self._min_temp_param.item()),
                max=(None if self._max_temp_param is None else
                     self._max_temp_param.item()))

        if self._temp_parameterization == 'exp':
            temp = log_temp.exp()
        else:
            temp = log_temp.exp().exp().add(1.).log()
        dist = GumbelSoftmax(temperature=temp, logits=logits)

        return dist


class GumbelSoftmaxMLPModule(GumbelSoftmaxMLPBaseModule):
    """GumbelSoftmaxMLPModule where temp and logits share the same network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for logits. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_temp (bool): Is temp trainable.
        init_temp (float): Initial value for temp.
            (plain value - not log or exponentiated).
        min_temp (float): If not None, the temp is at least the value of
            min_temp, to avoid numerical issues.
        max_temp (float): If not None, the temp is at most the value of
            max_temp, to avoid numerical issues.
        temp_parameterization (str): How the temp should be parametrized. There
            are two options:
            - exp: the logarithm of the temp will be stored, and applied a
               exponential transformation
            - softplus: the temp will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_temp=True,
                 init_temp=1.0,
                 min_temp=1e-6,
                 max_temp=None,
                 temp_parameterization='exp',
                 layer_normalization=False):
        super(GumbelSoftmaxMLPModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_temp=learn_temp,
                             init_temp=init_temp,
                             min_temp=min_temp,
                             max_temp=max_temp,
                             temp_parameterization=temp_parameterization,
                             layer_normalization=layer_normalization)

        self._logits_module = MLPModule(
            input_dim=self._input_dim,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearity=self._output_nonlinearity,
            output_w_init=self._output_w_init,
            output_b_init=self._output_b_init,
            layer_normalization=self._layer_normalization)

    def _get_temp_and_logits(self, *inputs):
        """Get temp and logits of GumbelSoftmax distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The logits of GumbelSoftmax distribution.
            torch.Tensor: The temp of GumbelSoftmax distribution.

        """
        assert len(inputs) == 1
        logits = self._logits_module(*inputs)

        broadcast_shape = list(inputs[0].shape[:-1]) + [self._action_dim]
        log_temp = torch.zeros(*broadcast_shape) + self._init_temp

        return logits, log_temp


class GumbelSoftmaxMLPTwoHeadedModule(GumbelSoftmaxMLPBaseModule):
    """GumbelSoftmaxMLPModule which has only one logits network.

    Args:
        input_dim (int): Input dimension of the model.
        output_dim (int): Output dimension of the model.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for logits. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a torch.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        learn_temp (bool): Is temp trainable.
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
                 input_dim,
                 output_dim,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 learn_temp=True,
                 init_temp=1.0,
                 min_temp=1e-6,
                 max_temp=None,
                 temp_parameterization='exp',
                 layer_normalization=False):
        super(GumbelSoftmaxMLPTwoHeadedModule,
              self).__init__(input_dim=input_dim,
                             output_dim=output_dim,
                             hidden_sizes=hidden_sizes,
                             hidden_nonlinearity=hidden_nonlinearity,
                             hidden_w_init=hidden_w_init,
                             hidden_b_init=hidden_b_init,
                             output_nonlinearity=output_nonlinearity,
                             output_w_init=output_w_init,
                             output_b_init=output_b_init,
                             learn_temp=learn_temp,
                             init_temp=init_temp,
                             min_temp=min_temp,
                             max_temp=max_temp,
                             temp_parameterization=temp_parameterization,
                             layer_normalization=layer_normalization)

        self._shared_temp_logit_network = MultiHeadedMLPModule(
            n_heads=2,
            input_dim=self._input_dim,
            output_dims=[1, self._action_dim],
            hidden_sizes=self._hidden_sizes,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            output_nonlinearities=self._output_nonlinearity,
            output_w_inits=self._output_w_init,
            output_b_inits=[
                nn.init.zeros_,
                lambda x: nn.init.constant_(x, self._init_temp.item())
            ],
            layer_normalization=self._layer_normalization)

    def _get_temp_and_logits(self, *inputs):
        """Get temp and logits of GumbelSoftmax distribution given inputs.

        Args:
            *inputs: Input to the module.

        Returns:
            torch.Tensor: The logits of GumbelSoftmax distribution.
            torch.Tensor: The temp of GumbelSoftmax distribution.

        """
        return self._shared_temp_logit_network(*inputs)
