import pyro
import torch
import numpy as np

from gym import Env


class DesignEnv(Env):
    def __init__(self, design_space, model_space, model, budget,
                 true_model=None):
        """
        A generic class for building a SED MDP

        args:
            design_space (gym.Space): the space of experiment designs
            model_space (gym.Space): the space of model parameterisations
            outcome_space (gym.Space): the space of experiment outcomes
            model (models.ExperimentModel): a model of experiment outcomes
            true_model (models.ExperimentModel): a ground-truth model
        """
        self.action_space = design_space
        self.observation_space = model_space
        self.model = model
        self.n_parallel = model.n_parallel
        self.budget = budget
        self.true_model = true_model
        if true_model is None:
            self.true_model = lambda d: None
        self.exp_idx = 0

    def reset(self, n_parallel=1):
        self.n_parallel = n_parallel
        self.model.reset(n_parallel)
        self.exp_idx = 0
        return self.get_obs()

    def step(self, action):
        design = torch.tensor(action)
        # design = design*0 + torch.tensor([96.1118, 23.8202,  2.2753, 96.3520, 24.9322,  2.0734])
        # pyro.set_rng_seed(10)
        y = self.true_model(design)
        # if y is not None:
        #     design_matrix = torch.tensor(
        #         [[50.3792, 80.3535, 55.9687, 50.8422, 80.4572, 54.5796],
        #          [77.7523, 56.7398, 48.2423, 76.8291, 57.0578, 48.1290],
        #          [78.0204, 97.1950, 44.2077, 78.5460, 96.7123, 44.8592],
        #          [93.9356, 86.3262, 3.1899, 93.5782, 87.1479, 3.0300],
        #          [63.7868, 99.8714, 79.6939, 64.1879, 99.8821, 80.3965],
        #          [51.5320, 86.5144, 2.4482, 54.2495, 86.9252, 2.3194],
        #          [10.1960, 69.3457, 3.2294, 10.8484, 68.9667, 3.0892],
        #          [29.9592, 30.7825, 0.5911, 31.1638, 30.0594, 0.5490],
        #          [19.9938, 73.0488, 24.0562, 19.9052, 73.5574, 24.9325],
        #          [19.9507, 35.9193, 63.9539, 18.0343, 37.3669, 64.3128]]
        #     )
        #     ents, ys, obses = [], [], []
        #     for d in design_matrix:
        #         self.model.reset(self.n_parallel)
        #         design = design * 0 + d
        #         y = self.true_model(design)
        #         ys.append(np.array(y.squeeze(-1)))
        #         self.model.run_experiment(design, y)
        #         obses.append(np.array(self._get_obs().transpose()))
        #         ents.append(np.array(self.model.entropy().reshape(1, -1)))
        #     np.savez_compressed(
        #         "/home/bla363/boed/Experiments/bench.npz",
        #         ys=ys, obses=obses, ents=ents
        #     )
        self.model.run_experiment(design, y)
        self.exp_idx += 1
        obs = self.get_obs()
        reward = torch.zeros((self.n_parallel,))
        done = self.terminal()
        if done:
            reward = -self.model.entropy().reshape((self.n_parallel,))
        done = done * torch.ones_like(reward, dtype=torch.bool)
        y = y if y is not None else torch.zeros_like(reward)
        info = {'y': y}
        return obs, reward, done, info

    def get_obs(self):
        return np.array(self.model.get_params())

    def terminal(self):
        return self.exp_idx >= self.budget

    def render(self, mode='human'):
        pass
