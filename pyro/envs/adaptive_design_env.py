import torch

from gym import Env

LOWER = 0
UPPER = 1
TERMINAL = 2


class AdaptiveDesignEnv(Env):
    def __init__(self, design_space, history_space, model, budget, l,
                 true_model=None, bound_type=LOWER):
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
        self.model = model
        self.n_parallel = model.n_parallel
        self.budget = budget
        self.l = l
        self.bound_type = bound_type
        self.log_products = None
        self.last_logsumprod = None
        self.history = []
        self.true_model = true_model
        if true_model is None:
            self.true_model = lambda d: None
        self.thetas = None
        self.theta0 = None

    def reset(self, n_parallel=1):
        self.model.reset(n_parallel=n_parallel)
        self.n_parallel = n_parallel
        self.history = []
        self.log_products = torch.zeros((
            self.l + 1 if self.bound_type in [LOWER, TERMINAL] else self.l,
            self.n_parallel
        ))
        self.last_logsumprod = torch.logsumexp(self.log_products, dim=0)
        self.thetas = self.model.sample_theta(self.l + 1)
        # index theta correctly because it is a dict
        self.theta0 = {k: v[0] for k, v in self.thetas.items()}
        return self.get_obs()

    def step(self, action):
        design = torch.as_tensor(action)
        # y = self.true_model(design)
        y = self.model.run_experiment(design, self.theta0)
        self.history.append(
            torch.cat(
                [design.squeeze(dim=-2).squeeze(dim=-2), y.squeeze(dim=-2)],
                dim=-1
            )
        )
        obs = self.get_obs()
        # TODO: make sure we get correct rewards
        reward = self.get_reward(y, design)
        done = self.terminal()
        done = done * torch.ones_like(reward, dtype=torch.bool)
        info = {'y': y.squeeze()}
        return obs, reward, done, info

    def get_obs(self):
        if self.history:
            return torch.stack(self.history, dim=-2)
        else:
            return torch.zeros(
                (self.n_parallel, 0, self.observation_space.shape[-1]),
            )

    def terminal(self):
        return len(self.history) >= self.budget
        # return False

    def get_reward(self, y, design):
        with torch.no_grad():
            log_probs = self.model.get_likelihoods(
                y, design, self.thetas).squeeze(dim=-1)
        log_prob0 = log_probs[0]
        if self.bound_type in [LOWER, TERMINAL]:
            # maximise lower bound
            self.log_products += log_probs
        elif self.bound_type == UPPER:
            # maximise upper bound
            self.log_products += log_probs[1:]

        logsumprod = torch.logsumexp(self.log_products, dim=0)
        if self.bound_type in [LOWER, UPPER]:
            reward = log_prob0 + self.last_logsumprod - logsumprod
        elif self.bound_type == TERMINAL:
            if self.terminal():
                reward = self.log_products[0] - logsumprod + \
                         torch.log(torch.as_tensor(self.l + 1.))
            else:
                reward = torch.zeros(self.n_parallel)
        self.last_logsumprod = logsumprod
        return reward

    def render(self, mode='human'):
        pass
