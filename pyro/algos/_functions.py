"""Utility functions for NumPy-based Reinforcement learning algorithms."""
import numpy as np
import torch

from pyro.sampler.utils import rollout
from pyro._dtypes import TrajectoryBatch


def obtain_evaluation_samples(policy, env, max_path_length=1000,
                              num_trajs=100, n_parallel=1):
    """Sample the policy for num_trajs trajectories and return average values.

    Args:
        policy (garage.Policy): Policy to use as the actor when
            gathering samples.
        env (garage.envs.GarageEnv): The environement used to obtain
            trajectories.
        max_path_length (int): Maximum path length. The episode will
            terminate when length of trajectory reaches max_path_length.
        num_trajs (int): Number of trajectories.

    Returns:
        TrajectoryBatch: Evaluation trajectories, representing the best
            current performance of the algorithm.

    """
    paths = []
    # Use a finite length rollout for evaluation.

    path = rollout(env,
                   policy,
                   max_path_length=max_path_length,
                   deterministic=True,
                   n_parallel=n_parallel)
    lengths = torch.full((n_parallel,), max_path_length)
    last_observations = path["observations"][max_path_length-1::max_path_length]
    return TrajectoryBatch(env_spec=env.spec,
                           observations=path["observations"],
                           last_observations=last_observations,
                           masks=None,
                           last_masks=None,
                           actions=path["actions"],
                           rewards=path["rewards"],
                           terminals=path["dones"],
                           env_infos=path["env_infos"],
                           agent_infos=path["agent_infos"],
                           lengths=lengths)
