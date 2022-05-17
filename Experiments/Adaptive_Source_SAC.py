"""
Use SAC to learn an agent that adaptively designs source location experiments
"""
import argparse

import joblib
import torch
import numpy as np

from garage.experiment import deterministic
from garage.torch import set_gpu_mode
from os import environ
from pyro import wrap_experiment, set_rng_seed
from pyro.algos import SAC
from pyro.envs import AdaptiveDesignEnv, GymEnv, normalize
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from pyro.experiment import Trainer
from pyro.models.adaptive_experiment_model import SourceModel
from pyro.policies import AdaptiveTanhGaussianPolicy, AdaptiveToyPolicy
from pyro.q_functions.adaptive_mlp_q_function import AdaptiveMLPQFunction
from pyro.replay_buffer import PathBuffer, NMCBuffer
from pyro.sampler.local_sampler import LocalSampler
from pyro.sampler.vector_worker import VectorWorker
from pyro.spaces.batch_box import BatchBox
from torch import nn
from dowel import logger

seeds = [373693, 943929, 675273, 79387, 508137, 557390, 756177, 155183, 262598,
         572185]


def main(n_parallel=1, budget=1, n_rl_itr=1, n_cont_samples=10, seed=0,
         log_dir=None, snapshot_mode='gap', snapshot_gap=500, bound_type=LOWER,
         src_filepath=None, discount=1., alpha=None, k=2, d=2, log_info=None,
         tau=5e-3, pi_lr=3e-4, qf_lr=3e-4, buffer_capacity=int(1e6), ens_size=2,
         M=2, minibatch_size=4096):
    if log_info is None:
        log_info = []

    @wrap_experiment(log_dir=log_dir, snapshot_mode=snapshot_mode,
                     snapshot_gap=snapshot_gap)
    def sac_source(ctxt=None, n_parallel=1, budget=1, n_rl_itr=1,
                   n_cont_samples=10, seed=0, src_filepath=None, discount=1.,
                   alpha=None, k=2, d=2, tau=5e-3, pi_lr=3e-4, qf_lr=3e-4,
                   buffer_capacity=int(1e6), ens_size=2, M=2,
                   minibatch_size=4096):
        if log_info:
            logger.log(str(log_info))
        if torch.cuda.is_available():
            set_gpu_mode(True, int(environ['CUDA_VISIBLE_DEVICES']))
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            logger.log("GPU available")
        else:
            set_gpu_mode(False)
            logger.log("no GPU detected")
        deterministic.set_seed(seed)
        set_rng_seed(seed)
        # if there is a saved agent to load
        if src_filepath:
            logger.log(f"loading data from {src_filepath}")
            data = joblib.load(src_filepath)
            env = data['env']
            sac = data['algo']
            if not hasattr(sac, '_sampler'):
                sac._sampler = LocalSampler(agents=sac.policy, envs=env,
                                            max_episode_length=budget,
                                            worker_class=VectorWorker)
            if not hasattr(sac, 'replay_buffer'):
                sac.replay_buffer = PathBuffer(
                    capacity_in_transitions=buffer_capacity)
            if alpha is not None:
                sac._use_automatic_entropy_tuning = False
                sac._fixed_alpha = alpha
        else:
            logger.log("creating new policy")
            layer_size = 128
            design_space = BatchBox(low=-4., high=4., shape=(1, 1, 1, d))
            obs_space = BatchBox(low=torch.as_tensor([-4.] * d + [-3.]),
                                 high=torch.as_tensor([4.] * d + [10.])
                                 )
            # is_cube = round(minibatch_size ** (1/3)) ** 3 == minibatch_size
            is_cube = False
            if is_cube:
                n_in_samples = round(minibatch_size ** (1/3))
                n_out_samples = n_in_samples ** 2
                ratio = int(n_parallel / n_in_samples)
                n_parallel = ratio * n_in_samples
                logger.log(f"changing n_parallel to {n_parallel}")
                capacity_factor = int(buffer_capacity / (n_in_samples * budget))
                buffer_capacity = capacity_factor * n_in_samples * budget
                logger.log(f"changing buffer_capacity to {buffer_capacity}")
                replay_buffer = NMCBuffer(buffer_capacity, n_in_samples,
                                          n_out_samples, budget)
            else:
                n_in_samples = n_out_samples = ratio = 1
                replay_buffer = PathBuffer(capacity_in_transitions=buffer_capacity)
            model = SourceModel(n_parallel=n_parallel, d=d, k=k)

            def make_env(design_space, obs_space, model, budget, n_cont_samples,
                         bound_type, true_model=None):
                env = GymEnv(
                    normalize(
                        AdaptiveDesignEnv(
                            design_space, obs_space, model, budget,
                            n_cont_samples, true_model=true_model,
                            bound_type=bound_type, M=n_in_samples,
                            N=ratio),
                        normalize_obs=True
                    )
                )
                return env

            def make_policy():
                return AdaptiveToyPolicy(
                    env_spec=env.spec,
                    encoder_sizes=[layer_size, layer_size],
                    encoder_nonlinearity=nn.ReLU,
                    encoder_output_nonlinearity=None,
                    emitter_sizes=[layer_size, layer_size],
                    emitter_nonlinearity=nn.ReLU,
                    emitter_output_nonlinearity=None,
                    encoding_dim=layer_size//2,
                    init_mean=0.4,
                    init_std=np.sqrt(1 / 3),
                    min_std=np.exp(-20.),
                    max_std=np.exp(0.),
                )

            def make_q_func():
                return AdaptiveMLPQFunction(
                    env_spec=env.spec,
                    encoder_sizes=[layer_size, layer_size],
                    encoder_nonlinearity=nn.ReLU,
                    encoder_output_nonlinearity=None,
                    emitter_sizes=[layer_size, layer_size],
                    emitter_nonlinearity=nn.ReLU,
                    emitter_output_nonlinearity=None,
                    encoding_dim=layer_size//2
                )

            env = make_env(design_space, obs_space, model, budget,
                           n_cont_samples, bound_type)
            if is_cube:
                assert env.M == replay_buffer.M
                assert env.budget == replay_buffer.path_len
            policy = make_policy()
            qfs = [make_q_func() for _ in range(ens_size)]
            sampler = LocalSampler(agents=policy, envs=env,
                                   max_episode_length=budget,
                                   worker_class=VectorWorker)

            sac = SAC(env_spec=env.spec,
                      policy=policy,
                      qfs=qfs,
                      replay_buffer=replay_buffer,
                      sampler=sampler,
                      max_episode_length_eval=budget,
                      gradient_steps_per_itr=64,
                      min_buffer_size=int(1e5),
                      target_update_tau=tau,
                      policy_lr=pi_lr,
                      qf_lr=qf_lr,
                      discount=discount,
                      discount_delta=0.,
                      fixed_alpha=alpha,
                      buffer_batch_size=minibatch_size,
                      reward_scale=1.,
                      M=M)

        sac.to()
        trainer = Trainer(snapshot_config=ctxt)
        trainer.setup(algo=sac, env=env)
        trainer.train(n_epochs=n_rl_itr, batch_size=n_parallel * budget)

    sac_source(n_parallel=n_parallel, budget=budget, n_rl_itr=n_rl_itr,
               n_cont_samples=n_cont_samples, seed=seed,
               src_filepath=src_filepath, discount=discount, alpha=alpha, k=k,
               d=d, tau=tau, pi_lr=pi_lr, qf_lr=qf_lr,
               buffer_capacity=buffer_capacity, ens_size=ens_size, M=M,
               minibatch_size=minibatch_size)

    logger.dump_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="1", type=int)
    parser.add_argument("--n-parallel", default="100", type=int)
    parser.add_argument("--budget", default="30", type=int)
    parser.add_argument("--n-rl-itr", default="50", type=int)
    parser.add_argument("--n-contr-samples", default="10", type=int)
    parser.add_argument("--log-dir", default=None, type=str)
    parser.add_argument("--src-filepath", default=None, type=str)
    parser.add_argument("--snapshot-mode", default="gap", type=str)
    parser.add_argument("--snapshot-gap", default=500, type=int)
    parser.add_argument("--bound-type", default="terminal", type=str.lower,
                        choices=["lower", "upper", "terminal"])
    parser.add_argument("--discount", default="1", type=float)
    parser.add_argument("--alpha", default="-1", type=float)
    parser.add_argument("--d", default="2", type=int)
    parser.add_argument("--k", default="2", type=int)
    parser.add_argument("--tau", default="5e-3", type=float)
    parser.add_argument("--pi-lr", default="3e-4", type=float)
    parser.add_argument("--qf-lr", default="3e-4", type=float)
    parser.add_argument("--buffer-capacity", default="1e6", type=float)
    parser.add_argument("--ens-size", default="2", type=int)
    parser.add_argument("--M", default="2", type=int)
    parser.add_argument("--minibatch-size", default="4096", type=int)
    args = parser.parse_args()
    bound_type_dict = {"lower": LOWER, "upper": UPPER, "terminal": TERMINAL}
    bound_type = bound_type_dict[args.bound_type]
    exp_id = args.id
    alpha = args.alpha if args.alpha >= 0 else None
    buff_cap = int(args.buffer_capacity)
    log_info = f"input params: {vars(args)}"
    main(n_parallel=args.n_parallel, budget=args.budget, n_rl_itr=args.n_rl_itr,
         n_cont_samples=args.n_contr_samples, seed=seeds[exp_id - 1],
         log_dir=args.log_dir, snapshot_mode=args.snapshot_mode,
         snapshot_gap=args.snapshot_gap, bound_type=bound_type,
         src_filepath=args.src_filepath, discount=args.discount, alpha=alpha,
         k=args.k, d=args.d, log_info=log_info, tau=args.tau, pi_lr=args.pi_lr,
         qf_lr=args.qf_lr, buffer_capacity=buff_cap, ens_size=args.ens_size,
         M=args.M, minibatch_size=args.minibatch_size)
