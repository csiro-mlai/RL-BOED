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
        self.budget = budget
        self.exp_idx = 0
        self.reset()

    def reset(self):
        self.model.reset()
        self.exp_idx = 0
        return self._get_obs()

    def step(self, action):
        design = torch.tensor(action)
        self.model.run_experiment(design)
        self.exp_idx += 1
        obs = self._get_obs()
        reward = 0
        done = self.terminal()
        if done:
            reward = -self.model.entropy()
        info = {}
        return obs, reward, done, info

    def _get_obs(self):
        return np.asarray(self.model.get_params())

    def terminal(self):
        return self.exp_idx >= self.budget

    def render(self, mode='human'):
        pass
