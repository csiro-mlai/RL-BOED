import torch

from gym import Env


class AdaptiveDesignEnv(Env):
    def __init__(self, design_space, history_space, model, budget, l,
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
        self.observation_space = history_space
        self.obs_dims = history_space.low.ndim
        self.model = model
        self.n_parallel = model.n_parallel
        self.budget = budget
        self.l = l
        self.log_products = None
        self.last_logsumprod = None
        self.history = []
        self.true_model = true_model
        if true_model is None:
            self.true_model = lambda d: None
        self.thetas = None

    def reset(self, n_parallel=1):
        self.history = []
        self.log_products = torch.zeros((self.l + 1, self.n_parallel))
        self.last_logsumprod = torch.logsumexp(self.log_products, dim=0)
        self.thetas = self.model.sample_theta(self.l + 1)
        return self.get_obs()

    def step(self, action):
        design = torch.as_tensor(action)
        # y = self.true_model(design)
        # index theta correctly because it is a dict
        theta0 = {k: v[0] for k, v in self.thetas.items()}
        y = self.model.run_experiment(design, theta0)
        self.history.append(
            torch.cat([design.squeeze(), y.squeeze(dim=-1)], dim=-1))
        obs = self.get_obs()
        reward = self.get_reward(y, design)
        done = self.terminal()
        done = done * torch.ones_like(reward, dtype=torch.bool)
        info = {'y': y.squeeze()}
        return obs, reward, done, info

    def get_obs(self):
        if self.history:
            return torch.stack(self.history, dim=-1-self.obs_dims)
        else:
            return torch.zeros(
                (self.n_parallel, 0, self.observation_space.shape[-1]),
            )

    def terminal(self):
        return len(self.history) >= self.budget
        # return False

    def get_reward(self, y, design):
        log_probs = self.model.get_likelihoods(y, design, self.thetas).squeeze()
        log_prob0 = log_probs[0]
        self.log_products += log_probs
        logsumprod = torch.logsumexp(self.log_products, dim=0)
        reward = log_prob0 + self.last_logsumprod - logsumprod
        self.last_logsumprod = logsumprod
        return reward


    def render(self, mode='human'):
        pass
