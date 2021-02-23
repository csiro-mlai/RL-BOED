"""A worker class for environments that receive a vector of actions and return a
 vector of states. Not to be confused with vec_worker, which handles a vector of
 environments."""
from collections import defaultdict

from garage import TrajectoryBatch
from garage.sampler.default_worker import DefaultWorker

import numpy as np


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

    def update_env(self, env_update):
        super().update_env(env_update)
        self._n_parallel = self.env.n_parallel

    def start_rollout(self):
        """Begin a new rollout."""
        self._path_length = 0
        self._prev_obs = self.env.reset(n_parallel=self._n_parallel)
        self.agent.reset()

    def step_rollout(self, deterministic):
        """Take a vector of single time-steps in the current rollout

        Returns:
            bool: True iff the path is done, either due to the environment
            indicating termination or due to reaching `max_path_length`.
        """
        if self._path_length < self._max_path_length:
            a, agent_info = self.agent.get_actions(self._prev_obs)
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
            self._path_length += 1
            self._terminals.append(d)
            if not d.all():
                self._prev_obs = next_o
                return False
        self._lengths.append(self._path_length * np.ones(self._n_parallel,))
        self._last_observations.append(self._prev_obs)
        return True

    def collect_rollout(self):
        """Collect the current rollout of vectors, convert it to a vector of
        rollouts, and clear the internal buffer

        Returns:
            garage.TrajectoryBatch: A batch of the trajectories completed since
                the last call to collect_rollout().
        """
        observations = self._observations
        self._observations = []
        last_observations = self._last_observations
        self._last_observations = []
        actions = self._actions
        self._actions = []
        rewards = self._rewards
        self._rewards = []
        terminals = self._terminals
        self._terminals = []
        env_infos = self._env_infos
        self._env_infos = defaultdict(list)
        agent_infos = self._agent_infos
        self._agent_infos = defaultdict(list)
        for k, v in agent_infos.items():
            agent_infos[k] = np.concatenate(v)
        zs = np.zeros((self._n_parallel,))
        for k, v in env_infos.items():
            env_infos[k] = np.stack(v, axis=-1) + zs
        lengths = self._lengths
        self._lengths = []
        return TrajectoryBatch(self.env.spec, np.concatenate(observations),
                               np.concatenate(last_observations),
                               np.concatenate(actions), np.concatenate(rewards),
                               np.concatenate(terminals), dict(env_infos),
                               dict(agent_infos),
                               np.concatenate(lengths).astype('i'))

    def rollout(self, deterministic=False):
        """Sample a single rollout of the agent in the environment.

        Returns:
            garage.TrajectoryBatch: The collected trajectory.

        """
        self.start_rollout()
        while not self.step_rollout(deterministic):
            pass
        return self.collect_rollout()
