"""Adaptive versions of Garage q-functions"""

from pyro.q_functions.adaptive_discrete_q_function import \
    AdaptiveDiscreteQFunction
from pyro.q_functions.adaptive_dueling_q_function import \
    AdaptiveDuelingQFunction
from pyro.q_functions.adaptive_mlp_q_function import AdaptiveMLPQFunction

__all__ = [
    'AdaptiveDiscreteQFunction',
    'AdaptiveDuelingQFunction',
    'AdaptiveMLPQFunction'
]
