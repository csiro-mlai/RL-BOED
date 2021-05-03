from garage.torch.policies.stochastic_policy import StochasticPolicy


class ReproducingPolicy(StochasticPolicy):
    """
    A policy for reproducing previously recorded actions

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        actions (list[torch.Tensor]): list of actions recorded from the policy
            that we wish to reproduce
        agent_infos (list[dict]) list of agent_info dictionaries that corrpond
            to the policy actions
    """
    def __init__(self,
                 env_spec,
                 actions,
                 agent_infos):
        super().__init__(env_spec, name='ReproducingPolicy')
        self._idx = 0
        self._actions = actions
        self._agent_infos = agent_infos

    def get_actions(self, observation):
        actions = self._actions[:,self._idx]
        info = {k: v[:, self._idx] for k, v in self._agent_infos.items()}
        self._idx += 1
        return actions, info

    def forward(self, observations):
        pass

    def reset(self, dones=None):
        self._idx = 0