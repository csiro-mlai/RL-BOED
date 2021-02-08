import argparse
import torch
import pyro
import numpy as np

from garage import wrap_experiment
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.envs import GarageEnv, normalize
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.sampler import LocalSampler
from gym import spaces
from pyro.models.experiment_model import CESModel
from pyro.envs.design_env import DesignEnv
from pyro.sampler.vector_worker import VectorWorker
from torch import nn
from torch.nn import functional as F


seeds = [126127, 911353, 783935, 631280, 100573, 677846, 692965, 516184, 165479,
         643024]


@wrap_experiment(snapshot_mode='last')
def sac_ces(ctxt=None, seed=1):
    deterministic.set_seed(seed)
    pyro.set_rng_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    n_parallel = 100
    budget = 1
    layer_size = 128
    design_space = spaces.Box(low=0.01, high=100, shape=(1, 1, 1, 6))
    model_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7))
    model = CESModel(n_parallel=n_parallel, n_elbo_steps=1000,
                     n_elbo_samples=10)
    env = GarageEnv(
        normalize(
            DesignEnv(design_space, model_space, model, budget),
            normalize_obs=False
        )
    )
    true_model = pyro.condition(
        model.make_model(),
        {
            "rho": torch.tensor([.9, .1]),
            "alpha": torch.tensor([.2, .3, .5]),
            "u": torch.tensor(10.)
        },
    )
    eval_env = GarageEnv(
        normalize(
            DesignEnv(design_space, model_space, model, budget, true_model),
            normalize_obs=False
        )
    )

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[layer_size, layer_size],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        init_std=np.sqrt(1 / 3),
        min_std=np.exp(-20.),
        max_std=np.exp(0.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[layer_size, layer_size],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[layer_size, layer_size],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=1000,
              max_path_length=budget,
              replay_buffer=replay_buffer,
              min_buffer_size=int(n_parallel),
              target_update_tau=5e-3,
              discount=1.0,
              buffer_batch_size=256,
              reward_scale=1.,
              steps_per_epoch=1,
              num_evaluation_trajectories=n_parallel,
              eval_env=eval_env)

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    runner.setup(algo=sac, env=env, sampler_cls=LocalSampler,
                 worker_class=VectorWorker)
    runner.train(n_epochs=10000, batch_size=n_parallel * budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="1", type=int)
    args = parser.parse_args()
    exp_id = args.id
    sac_ces(seed=seeds[exp_id - 1])
