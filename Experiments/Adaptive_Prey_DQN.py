"""
Use DDQN to learn an agent that adaptively designs prey population experiments
"""
import argparse

import joblib
import torch
import pyro

from dowel import logger
from garage.experiment import deterministic
from garage.torch import set_gpu_mode
from pyro import wrap_experiment
from pyro.algos import DQN
from pyro.envs import AdaptiveDesignEnv, GymEnv, normalize
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from pyro.experiment import Trainer
from pyro.models.adaptive_experiment_model import PreyModel
from pyro.policies import AdaptiveArgmaxPolicy, EpsilonGreedyPolicy
from pyro.q_functions import AdaptiveDiscreteQFunction
from pyro.replay_buffer import PathBuffer
from pyro.sampler.local_sampler import LocalSampler
from pyro.sampler.vector_worker import VectorWorker
from pyro.spaces.batch_box import BatchBox
from pyro.spaces.batch_discrete import BatchDiscrete
from torch import nn

seeds = [373693, 943929, 675273, 79387, 508137, 557390, 756177, 155183, 262598,
         572185]


def main(n_parallel=1, budget=1, n_rl_itr=1, n_cont_samples=10, seed=0,
         log_dir=None, snapshot_mode='gap', snapshot_gap=500, bound_type=LOWER,
         src_filepath=None, discount=1., buffer_capacity=int(1e6), qf_lr=1e-3,
         update_freq=5, tau=None):
    @wrap_experiment(log_dir=log_dir, snapshot_mode=snapshot_mode,
                     snapshot_gap=snapshot_gap)
    def dqn_prey(ctxt=None, n_parallel=1, budget=1, n_rl_itr=1,
                 n_cont_samples=10, seed=0, src_filepath=None,
                 discount=1., buffer_capacity=int(1e6), qf_lr=1e-3,
                 update_freq=5, tau=None,):
        if log_info:
            logger.log(str(log_info))
        if torch.cuda.is_available():
            set_gpu_mode(True)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            print("\nGPU available\n")
        else:
            set_gpu_mode(False)
            print("\nno GPU detected\n")
        deterministic.set_seed(seed)
        pyro.set_rng_seed(seed)
        if src_filepath:
            print(f"loading data from {src_filepath}")
            data = joblib.load(src_filepath)
            env = data['env']
            dqn = data['algo']
        else:
            print("creating new policy")
            layer_size = 128
            design_space = BatchDiscrete(floor=0, n=300, shape=(1,) * 4)
            obs_space = BatchBox(low=0., high=300., shape=(2,))

            model = PreyModel(n_parallel=n_parallel)

            def make_env(design_space, obs_space, model, budget, n_cont_samples,
                         bound_type, true_model=None):
                env = GymEnv(
                    normalize(
                        AdaptiveDesignEnv(
                            design_space, obs_space, model, budget,
                            n_cont_samples, true_model=true_model,
                            bound_type=bound_type),
                        normalize_obs=True
                    )
                )
                return env

            def make_policy(qf):
                return AdaptiveArgmaxPolicy(
                    env_spec=env.spec,
                    qf=qf
                )

            def make_q_func():
                return AdaptiveDiscreteQFunction(
                    env_spec=env.spec,
                    encoder_sizes=[layer_size, layer_size],
                    encoder_nonlinearity=nn.ReLU,
                    encoder_output_nonlinearity=None,
                    emitter_sizes=[2*layer_size, 2*layer_size],
                    emitter_nonlinearity=nn.ReLU,
                    emitter_output_nonlinearity=None,
                    encoding_dim=layer_size
                )

            env = make_env(design_space, obs_space, model, budget,
                           n_cont_samples, bound_type)
            qf = make_q_func()
            policy = make_policy(qf)
            exploration_policy = EpsilonGreedyPolicy(
                env_spec=env.spec,
                policy=policy,
                total_timesteps=n_rl_itr * n_parallel * budget,
                max_epsilon=1.0,
                min_epsilon=0.01,
                decay_ratio=0.1)

            replay_buffer = PathBuffer(capacity_in_transitions=buffer_capacity)
            sampler = LocalSampler(agents=exploration_policy, envs=env,
                                   max_episode_length=budget,
                                   worker_class=VectorWorker)

            dqn = DQN(env_spec=env.spec,
                      policy=policy,
                      qf=qf,
                      double_q=True,
                      replay_buffer=replay_buffer,
                      sampler=sampler,
                      exploration_policy=exploration_policy,
                      discount=discount,
                      buffer_batch_size=4096,
                      n_train_steps=64,
                      steps_per_epoch=1,
                      min_buffer_size=int(1e5),
                      max_episode_length_eval=budget,
                      target_update_freq=update_freq,
                      qf_lr=qf_lr,
                      tau=tau)

        dqn.to()
        trainer = Trainer(snapshot_config=ctxt)
        trainer.setup(algo=dqn, env=env)
        trainer.train(n_epochs=n_rl_itr, batch_size=n_parallel * budget)

    dqn_prey(n_parallel=n_parallel, budget=budget, n_rl_itr=n_rl_itr,
             n_cont_samples=n_cont_samples, seed=seed, qf_lr=qf_lr,
             src_filepath=src_filepath, discount=discount, tau=tau,
             buffer_capacity=buffer_capacity, update_freq=update_freq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="1", type=int)
    parser.add_argument("--n-parallel", default="100", type=int)
    parser.add_argument("--budget", default="25", type=int)
    parser.add_argument("--n-rl-itr", default="1000", type=int)
    parser.add_argument("--n-contr-samples", default="10", type=int)
    parser.add_argument("--log-dir", default=None, type=str)
    parser.add_argument("--src-filepath", default=None, type=str)
    parser.add_argument("--snapshot-mode", default="gap", type=str)
    parser.add_argument("--snapshot-gap", default=500, type=int)
    parser.add_argument("--bound-type", default="terminal", type=str.lower,
                        choices=["lower", "upper", "terminal"])
    parser.add_argument("--discount", default="0.95", type=float)
    parser.add_argument("--buffer-capacity", default="1e6", type=float)
    parser.add_argument("--qf-lr", default="1e-3", type=float)
    parser.add_argument("--update-freq", default="5", type=int)
    parser.add_argument("--tau", default="-1", type=float)
    args = parser.parse_args()
    bound_type_dict = {"lower": LOWER, "upper": UPPER, "terminal": TERMINAL}
    bound_type = bound_type_dict[args.bound_type]
    exp_id = args.id
    buff_cap = int(args.buffer_capacity)
    tau = args.tau if args.tau >= 0 else None
    log_info = f"input params: {vars(args)}"
    main(n_parallel=args.n_parallel, budget=args.budget, n_rl_itr=args.n_rl_itr,
         n_cont_samples=args.n_contr_samples, seed=seeds[exp_id - 1],
         log_dir=args.log_dir, snapshot_mode=args.snapshot_mode,
         snapshot_gap=args.snapshot_gap, bound_type=bound_type,
         src_filepath=args.src_filepath, discount=args.discount,
         buffer_capacity=buff_cap, qf_lr=args.qf_lr,
         update_freq=args.update_freq, tau=tau)


