"""A Discrete QFunction-derived policy.

This policy chooses the action that yields to the largest Q-value.
"""
import random

import torch

from garage.torch.policies.policy import Policy


class AdaptiveArgmaxPolicy(Policy):
    """Policy that derives its actions from a learned Q function.

    The action returned is the one that yields the highest Q value for
    a given state, as determined by the supplied Q function.

    Args:
        qf (object): Q network.
        env_spec (EnvSpec): Environment specification.
        name (str): Name of this policy.
    """

    def __init__(self, qfs, env_spec, name='AdaptiveArgmaxPolicy',
                 deep_exp=False):
        super().__init__(env_spec, name)
        self._qfs = qfs
        if deep_exp:
            assert isinstance(self._qfs, list), "attempted deep exploration " \
                                                "without ensemble"
            self._weights = self.stochastic_vector()
        self._deep_exp = deep_exp

    # pylint: disable=arguments-differ
    def forward(self, observations, masks=None, deep_exp=False):
        """Get actions corresponding to a batch of observations.

        Args:
            observations(torch.Tensor): Batch of observations of shape
                :math:`(N, O)`. Observations should be flattened even
                if they are images as the underlying Q network handles
                unflattening.

        Returns:
            torch.Tensor: Batch of actions of shape :math:`(N, A)`
        """
        if deep_exp:
            qs = torch.sum(
                self._weights * torch.stack(
                    [q(observations, masks) for q in self._qfs], dim=-1),
                dim=-1)
        elif isinstance(self._qfs, list):
            qs = torch.mean(
                torch.stack([q(observations, masks) for q in self._qfs]), dim=0)
        else:
            qs = self._qfs(observations, masks)
        return torch.argmax(qs, dim=1, keepdim=True)

    def get_action(self, observation, mask=None):
        """Get a single action given an observation.

        Args:
            observation (torch.tensor): Observation with shape :math:`(O, )`.

        Returns:
            torch.Tensor: Predicted action with shape :math:`(A, )`.
            dict: Empty since this policy does not produce a distribution.
        """
        act, info = self.get_actions(torch.unsqueeze(observation, dim=0),
                                     torch.unsqueeze(mask, dim=0))
        return act[0], info

    def get_actions(self, observations, masks=None):
        """Get actions given observations.

        Args:
            observations (torch.tensor): Batch of observations, should
                have shape :math:`(N, O)`.

        Returns:
            torch.Tensor: Predicted actions. Tensor has shape :math:`(N, A)`.
            dict: Empty since this policy does not produce a distribution.
        """
        with torch.no_grad():
            return self(observations, masks, self._deep_exp), dict()

    def reset(self, do_resets=None):
        if self._deep_exp:
            self._weights = self.stochastic_vector()

    def stochastic_vector(self):
        vec = torch.rand(len(self._qfs)) + 1e-9
        return vec / vec.sum()
