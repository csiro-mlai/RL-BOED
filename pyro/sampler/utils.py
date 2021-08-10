"""Utility functions related to sampling."""

import time

import numpy as np
import torch

from collections import defaultdict
from garage.misc import tensor_utils
from torch.nn.functional import pad


def transpose_tensor(arr_list):
    arr = np.stack(arr_list)
    arr_s = list(range(len(arr.shape)))
    arr_s[:2] = [1, 0]
    return arr.transpose(*arr_s)


def transpose_list(lst):
    return np.stack(lst).transpose()


def rollout(env,
            agent,
            *,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            deterministic=False,
            n_parallel=1):
    """Sample a single rollout of the agent in the environment.

    Args:
        agent(Policy): Agent used to select actions.
        env(gym.Env): Environment to perform actions in.
        max_path_length(int): If the rollout reaches this many timesteps, it is
            terminated.
        animated(bool): If true, render the environment after each step.
        speedup(float): Factor by which to decrease the wait time between
            rendered steps. Only relevant, if animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.

    Returns:
        dict[str, torch.Tensor or dict]: Dictionary, with keys:
            * observations(torch.Tensor): Flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape (T + 1, S^*) (the
                unflattened state space of the current environment).
            * actions(torch.Tensor): Non-flattened array of actions. Should have
                shape (T, S^*) (the unflattened action space of the current
                environment).
            * rewards(torch.Tensor): Array of rewards of shape (T,) (1D array of
                length timesteps).
            * agent_infos(Dict[str, torch.Tensor]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, torch.Tensor]): Dictionary of stacked,
                non-flattened `env_info` arrays.
            * dones(torch.Tensor): Array of termination signals.

    """
    observations = []
    actions = []
    rewards = []
    agent_infos = defaultdict(list)
    env_infos = defaultdict(list)
    dones = []
    o = env.reset(n_parallel=n_parallel)
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < (max_path_length or np.inf):
        a, agent_info = agent.get_actions(o)
        if deterministic and 'mean' in agent_info:
            a = agent_info['mean']
        a_shape = (n_parallel,) + env.action_space.shape[1:]
        next_o, r, d, env_info = env.step(a.reshape(a_shape))
        o = pad(o, (0, 0, 0, max_path_length - path_length, 0, 0))
        observations.append(o)
        rewards.append(r)
        actions.append(a)
        for k, v in agent_info.items():
            agent_infos[k].append(v)
        for k, v in env_info.items():
            if hasattr(v, 'shape'):
                env_infos[k].append(v.squeeze())
            else:
                env_infos[k].append([v] * n_parallel)
        dones.append(d)
        path_length += 1
        if d.any():
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

    for k, v in agent_infos.items():
        agent_infos[k] = torch.cat(torch.split(torch.stack(v, dim=1), 1), dim=1
                                   ).squeeze(0)
    for k, v in env_infos.items():
        if torch.is_tensor(v[0]):
            env_infos[k] = torch.cat(
                torch.split(torch.stack(v, dim=1), 1), dim=1).squeeze(0)
        else:
            env_infos[k] = torch.cat(
                torch.split(torch.as_tensor(v), 1), dim=1).squeeze(0)
    return dict(
        observations=torch.cat(
            torch.split(torch.stack(observations, dim=1), 1), dim=1).squeeze(0),
        actions=torch.cat(
            torch.split(torch.stack(actions, dim=1), 1), dim=1).squeeze(0),
        rewards=torch.cat(
            torch.split(torch.stack(rewards, dim=1), 1), dim=1).squeeze(0),
        agent_infos=agent_infos,
        env_infos=env_infos,
        dones=torch.cat(
            torch.split(torch.stack(dones, dim=1), 1), dim=1).squeeze(0),
    )


def truncate_paths(paths, max_samples):
    """Truncate the paths so that the total number of samples is max_samples.

    This is done by removing extra paths at the end of
    the list, and make the last path shorter if necessary

    Args:
        paths (list[dict[str, np.ndarray]]): Samples, items with keys:
            * observations (np.ndarray): Enviroment observations
            * actions (np.ndarray): Agent actions
            * rewards (np.ndarray): Environment rewards
            * env_infos (dict): Environment state information
            * agent_infos (dict): Agent state information
        max_samples(int) : Maximum number of samples allowed.

    Returns:
        list[dict[str, np.ndarray]]: A list of paths, truncated so that the
            number of samples adds up to max-samples

    Raises:
        ValueError: If key a other than 'observations', 'actions', 'rewards',
            'env_infos' and 'agent_infos' is found.

    """
    # chop samples collected by extra paths
    # make a copy
    valid_keys = {
        'observations', 'actions', 'rewards', 'env_infos', 'agent_infos'
    }
    paths = list(paths)
    total_n_samples = sum(len(path['rewards']) for path in paths)
    while paths and total_n_samples - len(paths[-1]['rewards']) >= max_samples:
        total_n_samples -= len(paths.pop(-1)['rewards'])
    if paths:
        last_path = paths.pop(-1)
        truncated_last_path = dict()
        truncated_len = len(
            last_path['rewards']) - (total_n_samples - max_samples)
        for k, v in last_path.items():
            if k in ['observations', 'actions', 'rewards']:
                truncated_last_path[k] = v[:truncated_len]
            elif k in ['env_infos', 'agent_infos']:
                truncated_last_path[k] = tensor_utils.truncate_tensor_dict(
                    v, truncated_len)
            else:
                raise ValueError(
                    'Unexpected key {} found in path. Valid keys: {}'.format(
                        k, valid_keys))
        paths.append(truncated_last_path)
    return paths
