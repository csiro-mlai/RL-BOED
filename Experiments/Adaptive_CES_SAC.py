import argparse
import torch
import pyro
import numpy as np

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from gym import spaces
from pyro.algos import SAC
from pyro.envs import AdaptiveDesignEnv, GarageEnv
from pyro.models.adaptive_experiment_model import CESModel
from pyro.policies.adaptive_tanh_gaussian_policy import \
    AdaptiveTanhGaussianPolicy
from pyro.q_functions.adaptive_mlp_q_function import AdaptiveMLPQFunction
from pyro.replay_buffer import ListBuffer
from pyro.sampler.vector_worker import VectorWorker
from pyro.spaces.batch_box import BatchBox
from torch import nn

seeds = [126127, 911353, 783935, 631280, 100573, 677846, 692965, 516184, 165479,
         643024]


@wrap_experiment(snapshot_mode='none')
def sac_ces(ctxt=None, n_parallel=1, seq_length=1, n_rl_itr=1,
            n_cont_samples=10, seed=0):
    deterministic.set_seed(seed)
    pyro.set_rng_seed(seed)
    layer_size = 128
    design_space = spaces.Box(low=0.01, high=100, shape=(1, 1, 1, 6))
    obs_space = BatchBox(low=np.array([-1.] * 6 + [0.]), high=np.ones((7,)))

    model = CESModel(n_parallel=n_parallel, n_elbo_steps=1000,
                     n_elbo_samples=10)

    def make_env(design_space, obs_space, model, seq_length, n_cont_samples,
                 true_model=None):
        env = GarageEnv(
            normalize(
                AdaptiveDesignEnv(design_space, obs_space, model, seq_length,
                                  n_cont_samples, true_model=true_model),
                normalize_obs=False
            )
        )
        return env

    def make_policy():
        return AdaptiveTanhGaussianPolicy(
            env_spec=env.spec,
            encoder_sizes=[layer_size, layer_size],
            encoder_nonlinearity=nn.ReLU,
            encoder_output_nonlinearity=None,
            emitter_sizes=[layer_size, layer_size],
            emitter_nonlinearity=nn.ReLU,
            emitter_output_nonlinearity=None,
            encoding_dim=layer_size // 2,
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
            encoding_dim=layer_size // 2
        )

    env = make_env(design_space, obs_space, model, seq_length, n_cont_samples)
    policy = make_policy()
    qf1 = make_q_func()
    qf2 = make_q_func()
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)

    runner = LocalRunner(snapshot_config=ctxt)
    true_model = pyro.condition(
        model.make_model(),
        {
            "rho": torch.tensor([.9, .1]),
            "alpha": torch.tensor([.2, .3, .5]),
            "u": torch.tensor(10.)
        },
    )
    eval_env = make_env(design_space, obs_space, model, seq_length,
                        n_cont_samples, true_model=true_model)
    replay_buffer = ListBuffer(capacity_in_transitions=int(1e6))

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=1000,
              max_path_length=seq_length,
              replay_buffer=replay_buffer,
              min_buffer_size=int(n_parallel),
              target_update_tau=5e-3,
              discount=1.0,
              buffer_batch_size=256,
              reward_scale=1.,
              steps_per_epoch=1,
              num_evaluation_trajectories=n_parallel,
              eval_env=eval_env)

    sac.to()
    runner.setup(algo=sac, env=env, sampler_cls=LocalSampler,
                 worker_class=VectorWorker)
    runner.train(n_epochs=n_rl_itr, batch_size=n_parallel * seq_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="1", type=int)
    parser.add_argument("--n-parallel", default="100", type=int)
    parser.add_argument("--seq-length", default="20", type=int)
    parser.add_argument("--n-rl-itr", default="50", type=int)
    parser.add_argument("--n-contr-samples", default="10", type=int)
    args = parser.parse_args()
    exp_id = args.id
    sac_ces(n_parallel=args.n_parallel, seq_length=args.seq_length,
            n_rl_itr=args.n_rl_itr, n_cont_samples=args.n_contr_samples,
            seed=seeds[exp_id - 1])
