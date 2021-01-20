import torch
import numpy as np

from gym import Env


class DesignEnv(Env):
    def __init__(self, design_space, model_space, model, budget):
        """
        A generic class for building a SED MDP

        args:
            design_space (gym.Space): the space of experiment designs
            model_space (gym.Space): the space of model parameterisations
            outcome_space (gym.Space): the space of experiment outcomes
            model (models.ExperimentModel): a model of experiment outcomes
        """
        self.action_space = design_space
        self.observation_space = model_space
        self.model = model
        self.n_parallel = model.n_parallel
        self.budget = budget
        self.exp_idx = 0

    def reset(self, n_parallel=1):
        self.n_parallel = n_parallel
        self.model.reset(n_parallel)
        self.exp_idx = 0
        return self._get_obs()

    def step(self, action):
        design = torch.tensor(action)
        self.model.run_experiment(design)
        self.exp_idx += 1
        obs = self._get_obs()
        reward = torch.zeros((self.n_parallel,))
        done = self.terminal()
        if done:
            reward = -self.model.entropy().reshape((self.n_parallel,))
        done = done * torch.ones_like(reward, dtype=torch.bool)
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        return np.array(self.model.get_params())

    def terminal(self):
        return self.exp_idx >= self.budget

    def render(self, mode='human'):
        pass
