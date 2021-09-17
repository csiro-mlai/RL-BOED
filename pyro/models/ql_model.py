from contextlib import ExitStack

import numpy as np
from torch.nn.functional import one_hot
from pyro import poutine
from pyro.contrib.util import iter_plates_to_shape, lexpand, rexpand, rmv
from pyro.models.adaptive_experiment_model import ExperimentModel
from pyro.util import is_bad
import torch
import pyro
import pyro.distributions as dist

EPS = 2**-22

class QLModel(ExperimentModel):
    def __init__(self, n_parallel=1, T = 100, obs_label="y"):
        super().__init__()
        self.n_parallel = n_parallel
        self.var_names = ["lr", "gamma"]
        self.var_dim = 2
        self.T = T
        self.obs_label = obs_label
        self.sanity_check()


    def make_model(self):
        def model(design):

            if is_bad(design):
                raise ArithmeticError("bad design, contains nan or inf")
            batch_shape = design.shape[:-2]
            with ExitStack() as stack:
                for plate in iter_plates_to_shape(batch_shape):
                    stack.enter_context(plate)
                batch_shape = design.shape[:-2]
                theta_shape = batch_shape + (1, 1,)
                theta_lr = pyro.sample(
                    "lr",
                    dist.Uniform(
                        torch.tensor([0.0]).expand(theta_shape),
                        torch.tensor([1.0]).expand(theta_shape)
                    ).to_event(2)
                )

                theta_gamma = pyro.sample(
                    "gamma",
                    dist.Uniform(
                        torch.tensor([0.0]).expand(theta_shape),
                        torch.tensor([10.0]).expand(theta_shape)
                    ).to_event(2)
                )

                Q = torch.rand(design.shape)
                y = torch.zeros(batch_shape + (1, self.T, 2))
                for i in range(self.T):
                    action = pyro.sample(f"a_{i}", dist.Categorical(logits=Q * theta_gamma).to_event(1))
                    p_reward0 = torch.gather(design, dim=-1, index=action[..., np.newaxis])
                    Q_a = torch.gather(Q, dim=-1, index=action[..., np.newaxis])
                    p_reward = torch.cat([p_reward0, 1-p_reward0], dim=-1)
                    reward = pyro.sample(f"r_{i}", dist.Categorical(probs=p_reward).to_event(1))
                    Q = Q + theta_lr * one_hot(action, num_classes=Q.shape[-1]) * (reward[..., np.newaxis] - Q_a)
                    y[..., i, 0] = action
                    y[..., i, 1] = reward
                return torch.squeeze(torch.flatten(y, start_dim=-2, end_dim=-1), dim=-2)

        return model

    def get_likelihoods(self, y, design, thetas):
        size = thetas[self.var_names[0]].shape[0]
        cond_dict = dict(thetas)
        y = y.unflatten(-1, (self.T, 2))
        for t in range(0, self.T):
            cond_dict.update({f"a_{t}": lexpand(y[..., t, np.newaxis, 0].type(torch.int64), size)})
            cond_dict.update({f"r_{t}": lexpand(y[..., t, np.newaxis, 1], size)})

        cond_model = pyro.condition(self.make_model(), data=cond_dict)
        trace = poutine.trace(cond_model).get_trace(lexpand(design, size))
        trace.compute_log_prob()
        likelihoods = trace.nodes[f"a_{0}"]["log_prob"] + trace.nodes[f"r_{1}"]["log_prob"]
        for t in range(1, self.T):
            likelihoods += trace.nodes[f"a_{t}"]["log_prob"] + trace.nodes[f"r_{t}"]["log_prob"]
        return likelihoods


    def reset(self, n_parallel):
        self.n_parallel = n_parallel


if __name__ == "__main__":
    pass
    # T = 100
    # # A x B x 1 x 1 x Design params (reward probs)
    # design = torch.rand([7, 5, 3])
    #
    # batch_shape = design.shape[:-1]
    # theta_shape = batch_shape + (1,)
    # theta_lr = pyro.sample(
    #     "lr",
    #     dist.Uniform(
    #         torch.tensor([0.0]).expand(theta_shape),
    #         torch.tensor([1.0]).expand(theta_shape)
    #     ).to_event(2)
    # )
    #
    # theta_gamma = pyro.sample(
    #     "gamma",
    #     dist.Uniform(
    #         torch.tensor([0.0]).expand(theta_shape),
    #         torch.tensor([10.0]).expand(theta_shape)
    #     ).to_event(2)
    # )
    #
    # Q = torch.rand(design.shape)
    # y = torch.zeros(batch_shape + (T, 2))
    # for i in range(T):
    #     action = pyro.sample("gamma", dist.Categorical(logits=Q * theta_gamma).to_event(2))
    #     p_reward0 = torch.gather(design, dim=2, index=action[:, :, np.newaxis])
    #     Q_a = torch.squeeze(torch.gather(Q, dim=2, index=action[:, :, np.newaxis]))
    #     p_reward = torch.cat([p_reward0, 1-p_reward0], dim=2)
    #     reward = pyro.sample("reward", dist.Categorical(probs=p_reward).to_event(2))
    #     Q = Q + theta_lr * one_hot(action, num_classes=Q.shape[2]) * (reward - Q_a)[:, :, np.newaxis]
    #     y[:, :, i, 0] = action
    #     y[:, :, i, 1] = reward
    #
    #
    # print(y)
