"""A path buffer specifically tailored for efficiently estimating nested
expectations. cf. On Nesting Monte Carlo Estimators, ICML 2018"""

import numpy as np
import torch

from pyro._dtypes import TimeStepBatch


class NMCBuffer:
    """A replay buffer that stores and can sample whole episodes.	
    This buffer stores transitions in a fixed ratio of outer samples to inner
    samples of a nested expectation. Samples from the buffer also maintain this
    ratio.
    Args:	
        capacity_in_transitions (int): Total memory allocated for the buffer.	
        env_spec (EnvSpec): Environment specification.
        M (int): number of trajectories per sample of theta.
        N (int): number of samples of theta.
        path_len (int): expected length of paths stored in the buffer
    """
    def __init__(self, capacity_in_transitions, M, N, path_len, env_spec=None):
        # Ensure that we don't have to split samples of theta
        assert capacity_in_transitions % (M * path_len) == 0
        self._capacity = capacity_in_transitions
        self._env_spec = env_spec
        self._transitions_stored = 0
        self._chunks_stored = 0
        self._next_idx = 0
        self._chunk_size = M * path_len
        self.M = M
        self.N = N
        self.path_len = path_len
        self._buffer = {}

    def add_episode_batch(self, episodes):
        """Add a EpisodeBatch to the buffer.	
        Args:	
            episodes (EpisodeBatch): Episodes to add.	
        """
        if self._env_spec is None:
            self._env_spec = episodes.env_spec
        for eps in episodes.split():
            path = dict(
                observation=eps.observations,
                mask=eps.masks,
                action=eps.actions,
                reward=eps.rewards.reshape(-1, 1),
                next_observation=eps.next_observations,
                next_mask=torch.cat(
                    [eps.masks[1:], eps.last_masks.unsqueeze(dim=0)]),
                terminal=eps.step_types.reshape(-1, 1),)
            self.add_path(path)

    def add_path(self, path):
        """Add a path to the buffer.

        Args:
            path (dict): A dict of array of shape (path_len, flat_dim).

        Raises:
            ValueError: If a key is missing from path or path has wrong shape.

        """
        for key, buf_arr in self._buffer.items():
            path_array = path.get(key, None)
            if path_array is None:
                raise ValueError('Key {} missing from path.'.format(key))
            if (len(path_array.shape) < 2
                    or path_array.shape[1:] != buf_arr.shape[1:]):
                raise ValueError('Array {} has wrong shape.'.format(key))
        assert self._get_path_length(path) == self.path_len
        start_idx = self._next_idx
        end_idx = start_idx + self.path_len
        for key, array in path.items():
            buf_arr = self._get_or_allocate_key(key, array)
            # numpy doesn't special case range indexing, so it's very slow.
            # Slice manually instead, which is faster than any other method.
            # pylint: disable=invalid-slice-index
            buf_arr[start_idx:end_idx] = array[:self.path_len]
        self._next_idx = self._next_idx + self.path_len
        if self._next_idx == self._capacity:
            self._next_idx = 0
        assert self._next_idx < self._capacity
        self._transitions_stored = min(self._capacity,
                                       self._transitions_stored + self.path_len)
        self._chunks_stored = self._transitions_stored / self._chunk_size

    def sample_transitions(self, batch_size):
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            dict: A dict of arrays of shape (batch_size, flat_dim).

        """
        assert batch_size == self.M * self.N
        bases = np.random.randint(self._chunks_stored, size=(self.N, 1))
        offsets = np.random.randint(self._chunk_size, size=(self.N, self.M))
        idx = (bases * self._chunk_size + offsets).flatten()
        return {key: buf_arr[idx] for key, buf_arr in self._buffer.items()}

    def sample_timesteps(self, batch_size):
        """Sample a batch of timesteps from the buffer.

        Args:
            batch_size (int): Number of timesteps to sample.

        Returns:
            TimeStepBatch: The batch of timesteps.

        """
        samples = self.sample_transitions(batch_size)
        return TimeStepBatch(env_spec=self._env_spec,
                             episode_infos={},
                             observations=samples['observation'],
                             masks=samples['mask'],
                             actions=samples['action'],
                             rewards=samples['reward'].flatten(),
                             next_observations=samples['next_observation'],
                             next_masks=samples['next_mask'],
                             step_types=samples['terminal'].flatten(),
                             env_infos={},
                             agent_infos={})

    def _get_or_allocate_key(self, key, array):
        """Get or allocate key in the buffer.

        Args:
            key (str): Key in buffer.
            array (numpy.ndarray): Array corresponding to key.

        Returns:
            numpy.ndarray: A NumPy array corresponding to key in the buffer.

        """
        buf_arr = self._buffer.get(key, None)
        if buf_arr is None:
            buf_arr = torch.zeros(
                (self._capacity,) + array.shape[1:],
                dtype=array.dtype
            )
            self._buffer[key] = buf_arr
        return buf_arr

    def clear(self):
        """Clear buffer."""
        self._transitions_stored = 0
        self._chunks_stored = 0
        self._next_idx = 0
        self._buffer.clear()

    @staticmethod
    def _get_path_length(path):
        """Get path length.

        Args:
            path (dict): Path.

        Returns:
            length: Path length.

        Raises:
            ValueError: If path is empty or has inconsistent lengths.

        """
        length_key = None
        length = None
        for key, value in path.items():
            if length is None:
                length = len(value)
                length_key = key
            elif len(value) != length:
                raise ValueError('path has inconsistent lengths between '
                                 '{!r} and {!r}.'.format(length_key, key))
        if not length:
            raise ValueError('Nothing in path')
        return length

    @property
    def n_transitions_stored(self):
        """Return the size of the replay buffer.

        Returns:
            int: Size of the current replay buffer.

        """
        return int(self._transitions_stored)
