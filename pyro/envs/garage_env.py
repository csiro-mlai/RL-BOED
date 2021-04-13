"""
This is a light wrapper around the standard GarageEnv that allows us
to pass akro spaces
"""
import akro
import gym

from garage import envs
from garage.envs.env_spec import EnvSpec


class GarageEnv(envs.GarageEnv):
    def __init__(self, env=None, env_name='', is_image=False):
        # Needed for deserialization
        self._env_name = env_name
        self._env = env

        if env_name:
            super().__init__(gym.make(env_name))
        else:
            super().__init__(env)

        if isinstance(self.env.action_space, akro.Space):
            self.action_space = self.env.action_space
        else:
            self.action_space = akro.from_gym(self.env.action_space)
        if isinstance(self.env.observation_space, akro.Space):
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = akro.from_gym(self.env.observation_space,
                                                   is_image=is_image)
        self.__spec = EnvSpec(action_space=self.action_space,
                              observation_space=self.observation_space)
