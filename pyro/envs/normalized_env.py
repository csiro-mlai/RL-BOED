"""An environment wrapper that normalizes action, observation and reward."""
import gym
import gym.spaces
import gym.spaces.utils
import pyro.spaces
import torch

from pyro.util import clip


class NormalizedEnv(gym.Wrapper):
    """An environment wrapper for normalization.

    This wrapper normalizes action, and optionally observation and reward.

    Args:
        env (garage.envs.GarageEnv): An environment instance.
        scale_reward (float): Scale of environment reward.
        normalize_obs (bool): If True, normalize observation.
        normalize_reward (bool): If True, normalize reward. scale_reward is
            applied after normalization.
        expected_action_scale (float): Assuming action falls in the range of
            [-expected_action_scale, expected_action_scale] when normalize it.
        flatten_obs (bool): Flatten observation if True.
        obs_alpha (float): Update rate of moving average when estimating the
            mean and variance of observations.
        reward_alpha (float): Update rate of moving average when estimating the
            mean and variance of rewards.

    """

    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            expected_action_scale=1.,
            flatten_obs=True,
            obs_alpha=0.001,
            reward_alpha=0.001,
    ):
        super().__init__(env)

        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._expected_action_scale = expected_action_scale
        self._flatten_obs = flatten_obs

        self._obs_alpha = obs_alpha
        flat_obs_dim = gym.spaces.utils.flatdim(env.observation_space)
        self._obs_mean = torch.zeros(flat_obs_dim)
        self._obs_var = torch.ones(flat_obs_dim)

        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.

    def _update_obs_estimate(self, obs):
        flat_obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
        self._obs_mean = (
                                 1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (
                                1 - self._obs_alpha) * self._obs_var + self._obs_alpha * \
                        torch.square(flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * \
                            self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
                                   1 - self._reward_alpha
                           ) * self._reward_var + self._reward_alpha * torch.square(
            reward - self._reward_mean)

    # def _apply_normalize_obs(self, obs):
    #     """Compute normalized observation.
    #
    #     Args:
    #         obs (torch.Tensor): Observation.
    #
    #     Returns:
    #         torch.Tensor: Normalized observation.
    #
    #     """
    #     self._update_obs_estimate(obs)
    #     flat_obs = gym.spaces.utils.flatten(self.env.observation_space, obs)
    #     normalized_obs = (flat_obs -
    #                       self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
    #     if not self._flatten_obs:
    #         normalized_obs = gym.spaces.utils.unflatten(
    #             self.env.observation_space, normalized_obs)
    #     return normalized_obs
    def _apply_normalize_obs(self, obs):
        """rescale observatios to range [0,1]

        Args:
            obs (torch.Tensor): Observation.

        Returns:
            torch.Tensor: Normalized observation.
        """
        lb, ub = self.observation_space.low, self.observation_space.high
        norm_obs = (obs - lb) / (ub - lb)
        return norm_obs

    def _apply_denormalize_obs(self, obs):
        """rescale observations from [0,1] to range [lb, ub]"""
        lb, ub = self.observation_space.low, self.observation_space.high
        denorm_obs = obs * (ub - lb) + lb
        return denorm_obs

    def _scale_action(self, action):
        """rescale action from [-1,1] to [lb, ub]"""
        lb, ub = self.action_space.low, self.action_space.high
        if torch.isfinite(lb).all() and torch.isfinite(ub).all():
            scaled_action = lb + (action + self._expected_action_scale) * (
                    0.5 * (ub - lb) / self._expected_action_scale)
            return clip(scaled_action, lb, ub)
        else:
            return action

    def _unscale_action(self, action):
        """rescale action from [lb, ub] tp [-1,1]"""
        lb, ub = self.action_space.low, self.action_space.high
        scaling_factor = 0.5 * (ub - lb) / self._expected_action_scale
        return (action - lb) / scaling_factor - self._expected_action_scale

    def _apply_normalize_reward(self, reward):
        """Compute normalized reward.

        Args:
            reward (float): Reward.

        Returns:
            float: Normalized reward.

        """
        self._update_reward_estimate(reward)
        return reward / (torch.sqrt(self._reward_var) + 1e-8)

    def reset(self, **kwargs):
        """Reset environment.

        Args:
            **kwargs: Additional parameters for reset.

        Returns:
            tuple:
                * observation (torch.Tensor): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        ret = self.env.reset(**kwargs)
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        else:
            return ret

    def step(self, action):
        """Feed environment with one step of action and get result.

        Args:
            action (torch.Tensor): An action fed to the environment.

        Returns:
            tuple:
                * observation (torch.Tensor): The observation of the environment.
                * reward (float): The reward acquired at this time step.
                * done (boolean): Whether the environment was completed at this
                    time step.
                * infos (dict): Environment-dependent additional information.

        """
        if isinstance(self.action_space, gym.spaces.Box):
            # rescale the action when the bounds are not inf
            scaled_action = self._scale_action(action)
        elif isinstance(self.action_space, pyro.spaces.BatchDiscrete):
            scaled_action = action + 1
        else:
            scaled_action = action

        next_obs, reward, done, info = self.env.step(scaled_action)

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        return next_obs, reward * self._scale_reward, done, info


normalize = NormalizedEnv
