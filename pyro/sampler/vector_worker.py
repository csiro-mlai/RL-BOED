"""A worker class for environments that receive a vector of actions and return a
 vector of states. Not to be confused with vec_worker, which handles a vector of
 environments."""
from collections import defaultdict

from pyro import TrajectoryBatch
from garage.sampler.default_worker import DefaultWorker

import torch


class VectorWorker(DefaultWorker):
    def __init__(
            self,
            *,  # Require passing by keyword, since everything's an int.
            seed,
            max_path_length,
            worker_number):
        super().__init__(seed=seed,
                         max_path_length=max_path_length,
                         worker_number=worker_number)
        self._n_parallel = None
        self._prev_mask = None
        self._masks = []
        self._last_masks = []

    def update_env(self, env_update):
        super().update_env(env_update)
        self._n_parallel = self.env.n_parallel

    def pad_observation(self, obs):
        pad_shape = list(obs.shape)
        pad_shape[1] = self._max_path_length - pad_shape[1]
        pad = torch.zeros(pad_shape)
        padded_obs = torch.cat([obs, pad], dim=1)
        mask = torch.cat([torch.ones_like(obs), pad], dim=1)[..., :1]
        return padded_obs, mask


    def start_rollout(self):
        """Begin a new rollout."""
        self._path_length = 0
        self._prev_obs = self.env.reset(n_parallel=self._n_parallel)
        self._prev_obs, self._prev_mask = self.pad_observation(self._prev_obs)
        self.agent.reset()

    def step_rollout(self, deterministic):
        """Take a vector of time-steps in the current rollout

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination or due to reaching `max_path_length`.
        """
        if self._path_length < self._max_path_length:
            a, agent_info = self.agent.get_actions(
                self._prev_obs, self._prev_mask)
            if deterministic and 'mean' in agent_info:
                a = agent_info['mean']
            a_shape = (self._n_parallel,) + self.env.action_space.shape[1:]
            next_o, r, d, env_info = self.env.step(a.reshape(a_shape))
            self._observations.append(self._prev_obs)
            self._rewards.append(r)
            self._actions.append(a)
            for k, v in agent_info.items():
                self._agent_infos[k].append(v)
            for k, v in env_info.items():
                self._env_infos[k].append(v)
            self._masks.append(self._prev_mask)
            self._path_length += 1
            self._terminals.append(d.float())
            if not d.all():
                next_o, next_mask = self.pad_observation(next_o)
                self._prev_obs = next_o
                self._prev_mask = next_mask
                return False
        self._lengths = self._path_length * torch.ones(self._n_parallel,
                                                       dtype=torch.int)
        self._last_observations.append(self._prev_obs)
        self._last_masks.append(self._prev_mask)
        return True

    def collect_rollout(self):
        """Collect the current rollout of vectors, convert it to a vector of
        rollouts, and clear the internal buffer

        Returns:
            garage.TrajectoryBatch: A batch of the trajectories completed since
                the last call to collect_rollout().
        """
        observations = torch.cat(
            torch.split(torch.stack(self._observations, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._observations = []
        last_observations = torch.cat(self._last_observations)
        self._last_observations = []
        masks = torch.cat(
            torch.split(torch.stack(self._masks, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._masks = []
        last_masks = torch.cat(self._last_masks)
        self._last_masks = []
        actions = torch.cat(
            torch.split(torch.stack(self._actions, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._actions = []
        rewards = torch.cat(
            torch.split(torch.stack(self._rewards, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._rewards = []
        terminals = torch.cat(
            torch.split(torch.stack(self._terminals, dim=1), 1),
            dim=1
        ).squeeze(0)
        self._terminals = []
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = torch.cat(
                torch.split(torch.stack(v, dim=1), 1),
                dim=1
            ).squeeze(0)
        zs = torch.zeros((self._n_parallel, self._lengths[0]))
        for k, v in env_infos.items():
            if torch.is_tensor(v[0]):
                env_infos[k] = torch.cat(
                    torch.split(torch.stack(v, dim=1), 1),
                    dim=1
                ).squeeze(0)
            else:
                env_infos[k] = torch.cat(
                    torch.split(torch.as_tensor(v).float() + zs, 1),
                    dim=1
                ).squeeze(0)
        lengths = self._lengths
        self._lengths = []
        return TrajectoryBatch(self.env.spec, observations, last_observations,
                               masks, last_masks, actions, rewards, terminals,
                               dict(env_infos), dict(agent_infos), lengths)

    def rollout(self, deterministic=False):
        """Sample a single vectorised rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.start_rollout()
        while not self.step_rollout(deterministic):
            pass
        return self.collect_rollout()
