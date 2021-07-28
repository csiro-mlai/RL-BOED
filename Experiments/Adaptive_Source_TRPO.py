import argparse

import joblib
import torch
import pyro
import numpy as np

from garage.experiment import deterministic
from garage.torch import set_gpu_mode
from pyro import wrap_experiment
from pyro.algos import TRPO
from pyro.envs import AdaptiveDesignEnv, GarageEnv, normalize
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from pyro.experiment import LocalRunner
from pyro.models.adaptive_experiment_model import SourceModel
from pyro.policies import AdaptiveGaussianMLPPolicy
from pyro.value_functions import AdaptiveMLPValueFunction
from pyro.replay_buffer import PathBuffer
from pyro.sampler.local_sampler import LocalSampler
from pyro.sampler.vector_worker import VectorWorker
from pyro.spaces.batch_box import BatchBox
from torch import nn

seeds = [373693, 943929, 675273, 79387, 508137, 557390, 756177, 155183, 262598,
         572185]


def main(n_parallel=1, budget=1, n_rl_itr=1, n_cont_samples=10, seed=0,
         log_dir=None, snapshot_mode='gap', snapshot_gap=500, bound_type=LOWER,
         src_filepath=None, discount=1.):
    @wrap_experiment(log_dir=log_dir, snapshot_mode=snapshot_mode,
                     snapshot_gap=snapshot_gap)
    def trpo_source(ctxt=None, n_parallel=1, budget=1, n_rl_itr=1,
                   n_cont_samples=10, seed=0, src_filepath=None,
                   discount=1.):
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
            trpo = data['algo']
        else:
            print("creating new policy")
            layer_size = 256
            design_space = BatchBox(low=-4., high=4., shape=(1, 1, 1, 2))
            obs_space = BatchBox(low=torch.as_tensor([-8., -8., -3.]),
                                 high=torch.as_tensor([8., 8., 10.])
                                 )
            model = SourceModel(n_parallel=n_parallel)

            def make_env(design_space, obs_space, model, budget, n_cont_samples,
                         bound_type, true_model=None):
                env = GarageEnv(
                    normalize(
                        AdaptiveDesignEnv(
                            design_space, obs_space, model, budget,
                            n_cont_samples, true_model=true_model,
                            bound_type=bound_type),
                        normalize_obs=True
                    )
                )
                return env

            def make_policy():
                return AdaptiveGaussianMLPPolicy(
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

            def make_v_func():
                return AdaptiveMLPValueFunction(
                    env_spec=env.spec,
                    encoder_sizes=[layer_size, layer_size],
                    encoder_nonlinearity=nn.ReLU,
                    encoder_output_nonlinearity=None,
                    emitter_sizes=[layer_size, layer_size],
                    emitter_nonlinearity=nn.ReLU,
                    emitter_output_nonlinearity=None,
                    encoding_dim=16
                )

            env = make_env(design_space, obs_space, model, budget,
                           n_cont_samples, bound_type)
            policy = make_policy()
            qf = make_v_func()
            trpo = TRPO(env_spec=env.spec,
                        policy=policy,
                        value_function=qf,
                        max_path_length=budget,
                        discount=discount,
                        center_adv=False,
                        )

        runner = LocalRunner(snapshot_config=ctxt)
        runner.setup(algo=trpo, env=env, sampler_cls=LocalSampler,
                     worker_class=VectorWorker)
        runner.train(n_epochs=n_rl_itr, batch_size=n_parallel * budget)

    trpo_source(n_parallel=n_parallel, budget=budget, n_rl_itr=n_rl_itr,
               n_cont_samples=n_cont_samples, seed=seed,
               src_filepath=src_filepath, discount=discount)


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
    args = parser.parse_args()
    bound_type_dict = {"lower": LOWER, "upper": UPPER, "terminal": TERMINAL}
    bound_type = bound_type_dict[args.bound_type]
    exp_id = args.id
    main(n_parallel=args.n_parallel, budget=args.budget, n_rl_itr=args.n_rl_itr,
         n_cont_samples=args.n_contr_samples, seed=seeds[exp_id - 1],
         log_dir=args.log_dir, snapshot_mode=args.snapshot_mode,
         snapshot_gap=args.snapshot_gap, bound_type=bound_type,
         src_filepath=args.src_filepath, discount=args.discount)
