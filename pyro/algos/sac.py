"""This modules creates a sac model in PyTorch."""
# yapf: disable
from collections import deque
import copy

from pyro.dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F
import time

from pyro import log_performance
from pyro.algos._functions import obtain_evaluation_episodes
from garage.np.algos import RLAlgorithm
from garage.torch import as_torch_dict, global_device

# yapf: enable


class SAC(RLAlgorithm):
    """A SAC Model in Torch.

    Based on Soft Actor-Critic and Applications:
        https://arxiv.org/abs/1812.05905

    Soft Actor-Critic (SAC) is an algorithm which optimizes a stochastic
    policy in an off-policy way, forming a bridge between stochastic policy
    optimization and DDPG-style approaches.
    A central feature of SAC is entropy regularization. The policy is trained
    to maximize a trade-off between expected return and entropy, a measure of
    randomness in the policy. This has a close connection to the
    exploration-exploitation trade-off: increasing entropy results in more
    exploration, which can accelerate learning later on. It can also prevent
    the policy from prematurely converging to a bad local optimum.

    Args:
        policy (garage.torch.policy.Policy): Policy/Actor/Agent that is being
            optimized by SAC.
        qfs [(garage.torch.q_function.ContinuousMLPQFunction)]: list of
            QFunctions used for actor/policy optimization.
        replay_buffer (ReplayBuffer): Stores transitions that are previously
            collected by the sampler.
        sampler (garage.sampler.Sampler): Sampler.
        env_spec (EnvSpec): The env_spec attribute of the environment that the
            agent is being trained in.
        max_episode_length_eval (int or None): Maximum length of episodes used
            for off-policy evaluation. If None, defaults to
            `env_spec.max_episode_length`.
        gradient_steps_per_itr(int): Number of optimization steps that should
            occur before the training step is over and a new batch of
            transitions is collected by the sampler.
        fixed_alpha (float): The entropy/temperature to be used if temperature
            is not supposed to be learned.
        target_entropy (float): target entropy to be used during
            entropy/temperature optimization. If None, the default heuristic
            from Soft Actor-Critic Algorithms and Applications is used.
        initial_log_entropy (float): initial entropy/temperature coefficient
            to be used if a fixed_alpha is not being used (fixed_alpha=None),
            and the entropy/temperature coefficient is being learned.
        discount (float): Discount factor to be used during sampling and
            critic/q_function optimization.
        discount_delta (float): change in discount factor once-per-epoch
        buffer_batch_size (int): The number of transitions sampled from the
            replay buffer that are used during a single optimization step.
        min_buffer_size (int): The minimum number of transitions that need to
            be in the replay buffer before training can begin.
        target_update_tau (float): coefficient that controls the rate at which
            the target q_functions update over optimization iterations.
        policy_lr (float): learning rate for policy optimizers.
        qf_lr (float): learning rate for q_function optimizers.
        reward_scale (float): reward scale. Changing this hyperparameter
            changes the effect that the reward from a transition will have
            during optimization.
        optimizer (torch.optim.Optimizer): optimizer to be used for
            policy/actor, q_functions/critics, and temperature/entropy
            optimizations.
        steps_per_epoch (int): Number of train_once calls per epoch.
        num_evaluation_episodes (int): The number of evaluation episodes used
            for computing eval stats at the end of every epoch.
        eval_env (Environment): environment used for collecting evaluation
            episodes. If None, a copy of the train env is used.
        use_deterministic_evaluation (bool): True if the trained policy
            should be evaluated deterministically.
        M (int): in-target minimization parameter
        ent_anneal_rate (float): the rate at which to anneal the target entropy
            in each iteration of the algorithm.

    """

    def __init__(
            self,
            env_spec,
            policy,
            qfs,
            replay_buffer,
            sampler,
            *,  # Everything after this is numbers.
            max_episode_length_eval=None,
            gradient_steps_per_itr,
            fixed_alpha=None,
            target_entropy=None,
            initial_log_entropy=0.,
            discount=0.99,
            discount_delta=0.,
            buffer_batch_size=64,
            min_buffer_size=int(1e4),
            target_update_tau=5e-3,
            policy_lr=3e-4,
            qf_lr=3e-4,
            reward_scale=1.0,
            optimizer=torch.optim.Adam,
            steps_per_epoch=1,
            num_evaluation_episodes=10,
            eval_env=None,
            use_deterministic_evaluation=True,
            M=2,
            ent_anneal_rate=0.):

        self._qfs = qfs
        self.replay_buffer = replay_buffer
        self._tau = target_update_tau
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._initial_log_entropy = initial_log_entropy
        self._gradient_steps = gradient_steps_per_itr
        self._optimizer = optimizer
        self._num_evaluation_episodes = num_evaluation_episodes
        self._eval_env = eval_env
        self._M = M

        self._min_buffer_size = min_buffer_size
        self._steps_per_epoch = steps_per_epoch
        self._buffer_batch_size = buffer_batch_size
        self._discount = discount
        self._discount_delta = discount_delta
        self._reward_scale = reward_scale
        self.max_episode_length = env_spec.max_episode_length
        self._max_episode_length_eval = env_spec.max_episode_length

        if max_episode_length_eval is not None:
            self._max_episode_length_eval = max_episode_length_eval
        self._use_deterministic_evaluation = use_deterministic_evaluation

        self.policy = policy
        self.env_spec = env_spec
        self.replay_buffer = replay_buffer

        self._sampler = sampler

        self._reward_scale = reward_scale
        # use ensemble of target q networks
        self._target_qfs = [copy.deepcopy(q) for q in self._qfs]
        self._policy_optimizer = self._optimizer(self.policy.parameters(),
                                                 lr=self._policy_lr)
        self._qf_optimizers = [
            self._optimizer(q.parameters(), lr=self._qf_lr) for q in self._qfs]
        # automatic entropy coefficient tuning
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(
                    self.env_spec.action_space.shape).item()
            self._log_alpha = torch.Tensor([self._initial_log_entropy
                                            ]).requires_grad_()
            self._alpha_optimizer = optimizer([self._log_alpha],
                                              lr=self._policy_lr)
        else:
            self._log_alpha = torch.Tensor([self._fixed_alpha]).log()
        self.episode_rewards = deque(maxlen=30)
        self._ent_anneal_rate = ent_anneal_rate

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None
        for _ in trainer.step_epochs():
            for _ in range(self._steps_per_epoch):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                trainer.step_episode = trainer.obtain_samples(
                    trainer.step_itr, batch_size)
                path_returns = []
                for path in trainer.step_episode:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=path['step_types'].reshape(-1, 1),
                             mask=path['masks'],
                             next_mask=path['next_masks'],))
                    path_returns.append(sum(path['rewards']))
                assert len(path_returns) == len(trainer.step_episode)
                allrets = torch.tensor(
                    [path["rewards"].sum() for path in trainer.step_episode]
                ).cpu().numpy()
                allact = torch.cat(
                    [path["actions"] for path in trainer.step_episode]
                ).cpu().numpy()
                self.episode_rewards.append(
                    torch.stack(path_returns).mean().cpu().numpy())
                for _ in range(self._gradient_steps):
                    policy_loss, qf_losses, entropy = self.train_once()
            last_return = allrets
            if self._eval_env is not None:
                last_return = self._evaluate_policy(trainer.step_itr)
            self._log_statistics(policy_loss, qf_losses, entropy)
            self._discount = np.clip(self._discount + self._discount_delta,
                                     a_min=0., a_max=1.)
            tabular.record('TotalEnvSteps', trainer.total_env_steps)
            tabular.record('Return/MedianReturn', np.median(allrets))
            tabular.record('Return/LowerQuartileReturn',
                           np.percentile(allrets, 25))
            tabular.record('Return/UpperQuartileReturn',
                           np.percentile(allrets, 75))
            tabular.record('Return/MeanReturn', np.mean(allrets))
            tabular.record('Return/StdReturn', np.std(allrets))
            tabular.record("Return/MaxReturn", allrets.max())
            tabular.record("Return/MinReturn", allrets.min())
            tabular.record("Policy/Discount", self._discount)
            if "log_std" in trainer.step_episode[0]["agent_infos"]:
                log_stds = torch.stack(
                    [p["agent_infos"]["log_std"] for p in trainer.step_episode])
                mean_std = log_stds.exp().mean(dim=(0, 1)).cpu().numpy()
                tabular.record("Policy/MeanStd", mean_std)
                mean_ent = log_stds.mean().cpu().numpy() + \
                    0.5 + 0.5 * np.log(2 * np.pi)
            if "mean" in trainer.step_episode[0]["agent_infos"]:
                mean_mean = torch.stack(
                    [p["agent_infos"]["mean"] for p in trainer.step_episode]
                ).mean(dim=(0, 1)).cpu().numpy()
                tabular.record("Policy/Mean", mean_mean)
            if "logits" in trainer.step_episode[0]["agent_infos"]:
                mean_temp = torch.stack(
                    [p["agent_infos"]["log_temp"] for p in trainer.step_episode]
                ).exp().mean(dim=(0, 1)).cpu().numpy()
                tabular.record("Policy/MeanTemp", mean_temp)
                logits = torch.stack(
                    [p["agent_infos"]["logits"] for p in trainer.step_episode])
                lps = F.log_softmax(logits, dim=-1)
                mean_ent = (-lps * lps.exp()).sum(dim=-1).mean().cpu().numpy()
                if not self._use_automatic_entropy_tuning:
                    self._log_alpha -= 1e-4
            if self._use_automatic_entropy_tuning:
                self._target_entropy -= self._ent_anneal_rate
            tabular.record("Policy/MeanEntropy", mean_ent)
            tabular.record("Action/MeanAction", allact.mean(axis=0))
            tabular.record("Action/StdAction", allact.std(axis=0))
            trainer.step_itr += 1

        return np.mean(last_return)

    def train_once(self, itr=None, paths=None):
        """Complete 1 training iteration of SAC.

        Args:
            itr (int): Iteration number. This argument is deprecated.
            paths (list[dict]): A list of collected paths.
                This argument is deprecated.

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        del itr
        del paths
        if self.replay_buffer.n_transitions_stored >= self._min_buffer_size:
            samples = self.replay_buffer.sample_transitions(
                self._buffer_batch_size)
            # samples = as_torch_dict(samples)
            policy_loss, qf_losses, entropy = self.optimize_policy(samples)
            self._update_targets()

        return policy_loss, qf_losses, entropy

    def _get_log_alpha(self, samples_data):
        """Return the value of log_alpha.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        This function exists in case there are versions of sac that need
        access to a modified log_alpha, such as multi_task sac.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: log_alpha

        """
        del samples_data
        log_alpha = self._log_alpha
        return log_alpha

    def _temperature_objective(self, log_pi, samples_data):
        """Compute the temperature/alpha coefficient loss.

        Args:
            log_pi(torch.Tensor): log probability of actions that are sampled
                from the replay buffer. Shape is (1, buffer_batch_size).
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: the temperature/alpha coefficient loss.

        """
        alpha_loss = 0
        if self._use_automatic_entropy_tuning:
            log_pi = log_pi.detach()
            alpha = self._get_log_alpha(samples_data).exp()
            alpha_loss = -alpha * (log_pi + self._target_entropy)
            if self.env_spec.action_space.is_discrete:
                alpha_loss = (alpha_loss * log_pi.exp()).sum(axis=-1)
            alpha_loss = alpha_loss.mean()
        return alpha_loss

    def _actor_objective(self, samples_data, new_actions, log_pi_new_actions):
        """Compute the Policy/Actor loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.
            new_actions (torch.Tensor): Actions resampled from the policy based
                based on the Observations, obs, which were sampled from the
                replay buffer. Shape is (action_dim, buffer_batch_size).
            log_pi_new_actions (torch.Tensor): Log probability of the new
                actions on the TanhNormal distributions that they were sampled
                from. Shape is (1, buffer_batch_size).

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from the Policy/Actor.

        """
        obs = samples_data['observation']
        mask = samples_data['mask']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        if self.env_spec.action_space.is_discrete:
            min_q_new_actions = torch.min(
                torch.stack([q(obs, mask=mask) for q in self._qfs]),
                dim=0).values
            pi_new_actions = log_pi_new_actions.exp()
            policy_objective = ((alpha * log_pi_new_actions) -
                                min_q_new_actions) * pi_new_actions
            policy_objective = policy_objective.sum(axis=1).mean()
        else:
            min_q_new_actions = torch.min(
                torch.stack([q(obs, new_actions, mask) for q in self._qfs]),
                dim=0).values
            policy_objective = ((alpha * log_pi_new_actions) -
                                min_q_new_actions.flatten()).mean()
        return policy_objective

    def _critic_objective(self, samples_data):
        """Compute the Q-function/critic loss.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        actions = samples_data['action']
        rewards = samples_data['reward'].flatten()
        terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']
        mask = samples_data['mask']
        next_mask = samples_data['next_mask']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()

        next_action_dist = self.policy(next_obs, mask)[0]
        if hasattr(next_action_dist, 'rsample_with_pre_tanh_value'):
            next_actions_pre_tanh, next_actions = (
                next_action_dist.rsample_with_pre_tanh_value())
            new_log_pi = next_action_dist.log_prob(
                value=next_actions, pre_tanh_value=next_actions_pre_tanh)
        else:
            new_log_pi = next_action_dist.logits

        # use random ensemble of q functions
        with torch.no_grad():
            m_idx = np.random.choice(len(self._qfs), self._M, replace=False)
            in_target_qfs = [self._target_qfs[i] for i in m_idx]
            # get exact expectation for discrete action spaces
            if self.env_spec.action_space.is_discrete:
                target_q_values = torch.min(
                    torch.stack([
                        q(next_obs, mask=next_mask) for q in in_target_qfs
                    ]),
                    dim=0
                ).values - alpha * new_log_pi
                new_pi = next_action_dist.probs
                target_q_values = (target_q_values * new_pi).sum(axis=1)
            else:
                target_q_values = torch.min(
                    torch.stack([
                        q(next_obs, next_actions, next_mask) for q in in_target_qfs
                    ]),
                    dim=0
                ).values.flatten() - (alpha * new_log_pi)
            q_target = rewards * self._reward_scale + (
                    1. - terminals) * self._discount * target_q_values
        q_preds = [q(obs, actions, mask) for q in self._qfs]
        qf_losses = [F.mse_loss(pred.flatten(), q_target) for pred in q_preds]
        return qf_losses

    def _update_targets(self):
        """Update parameters in the target q-functions."""
        for target_qf, qf in zip(self._target_qfs, self._qfs):
            for t_param, param in zip(target_qf.parameters(), qf.parameters()):
                t_param.data.copy_(t_param.data * (1.0 - self._tau) +
                                   param.data * self._tau)

    def optimize_policy(self, samples_data):
        """Optimize the policy q_functions, and temperature coefficient.

        Args:
            samples_data (dict): Transitions(S,A,R,S') that are sampled from
                the replay buffer. It should have the keys 'observation',
                'action', 'reward', 'terminal', and 'next_observations'.

        Note:
            samples_data's entries should be torch.Tensor's with the following
            shapes:
                observation: :math:`(N, O^*)`
                action: :math:`(N, A^*)`
                reward: :math:`(N, 1)`
                terminal: :math:`(N, 1)`
                next_observation: :math:`(N, O^*)`

        Returns:
            torch.Tensor: loss from actor/policy network after optimization.
            torch.Tensor: loss from 1st q-function after optimization.
            torch.Tensor: loss from 2nd q-function after optimization.

        """
        obs = samples_data['observation']
        mask = samples_data['mask']
        # train critic
        qf_losses = self._critic_objective(samples_data)
        for i in range(len(qf_losses)):
            self._qf_optimizers[i].zero_grad()
            qf_losses[i].backward()
            self._qf_optimizers[i].step()

        # train actor
        action_dists = self.policy(obs, mask)[0]
        if hasattr(action_dists, 'rsample_with_pre_tanh_value'):
            new_actions_pre_tanh, new_actions = (
                action_dists.rsample_with_pre_tanh_value())
            log_pi_new_actions = action_dists.log_prob(
                value=new_actions, pre_tanh_value=new_actions_pre_tanh)
        else:
            new_actions = None
            log_pi_new_actions = action_dists.logits
        policy_loss = self._actor_objective(samples_data, new_actions,
                                            log_pi_new_actions)
        self._policy_optimizer.zero_grad()
        policy_loss.backward()

        self._policy_optimizer.step()

        # train temperature
        entropy = -log_pi_new_actions.mean()
        if self.env_spec.action_space.is_discrete:
            entropy = (log_pi_new_actions.exp() * -log_pi_new_actions
                       ).sum(axis=-1).mean()
        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(log_pi_new_actions,
                                                     samples_data)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        return policy_loss, qf_losses, entropy

    def _evaluate_policy(self, epoch):
        """Evaluate the performance of the policy via deterministic sampling.

            Statistics such as (average) discounted return and success rate are
            recorded.

        Args:
            epoch (int): The current training epoch.

        Returns:
            float: The average return across self._num_evaluation_episodes
                episodes

        """
        eval_episodes = obtain_evaluation_episodes(
            self.policy,
            self._eval_env,
            self._max_episode_length_eval,
            num_eps=self._num_evaluation_episodes,
            n_parallel=self._eval_env.n_parallel)
        last_return = log_performance(epoch,
                                      eval_episodes,
                                      discount=self._discount)
        return last_return

    def _log_statistics(self, policy_loss, qf_losses, entropy):
        """Record training statistics to dowel such as losses and returns.

        Args:
            policy_loss (torch.Tensor): loss from actor/policy network.
            qf_losses [(torch.Tensor)]: loss from qf/critic networks.

        """
        with torch.no_grad():
            tabular.record('AlphaTemperature/mean',
                           self._log_alpha.exp().mean().item())
        tabular.record('Policy/Loss', policy_loss.item())
        tabular.record('Policy/Entropy', entropy.item())
        tabular.record('QF/{}'.format('QfLoss'),
                       np.mean([loss.item() for loss in qf_losses]))
        tabular.record('ReplayBuffer/buffer_size',
                       self.replay_buffer.n_transitions_stored)
        tabular.record('Average/TrainAverageReturn',
                       np.mean(self.episode_rewards))

    @property
    def networks(self):
        """Return all the networks within the model.

        Returns:
            list: A list of networks.

        """
        return [
            self.policy, *self._qfs, *self._target_qfs
        ]

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)
        if not self._use_automatic_entropy_tuning:
            self._log_alpha = torch.Tensor([self._fixed_alpha
                                            ]).log().to(device)
        else:
            self._log_alpha = torch.Tensor([self._initial_log_entropy
                                            ]).to(device).requires_grad_()
            self._alpha_optimizer = self._optimizer([self._log_alpha],
                                                    lr=self._policy_lr)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_sampler']
        del state['replay_buffer']
        return state
