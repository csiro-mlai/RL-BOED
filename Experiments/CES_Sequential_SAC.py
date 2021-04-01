import argparse
import torch
import pyro
import numpy as np
from dowel import logger

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
from pyro.contrib.util import lexpand
from pyro.envs.design_env import DesignEnv
from pyro.models.experiment_model import CESModel
from pyro.sampler.vector_worker import VectorWorker
from torch import nn
from torch.nn import functional as F

seeds = [126127, 911353, 783935, 631280, 100573, 677846, 692965, 516184, 165479,
         643024]


@wrap_experiment(snapshot_mode='last')
def sac_ces(ctxt=None, n_parallel=1, budget=1, seq_length=1, n_rl_itr=1,
            seed=0):
    # one-time setup
    deterministic.set_seed(seed)
    pyro.set_rng_seed(seed)
    posteriors = []
    infogains = []
    denormalised_designs = []
    normalised_designs = []
    ys = []
    layer_size = 128
    design_space = spaces.Box(low=0.01, high=100, shape=(1, 1, 1, 6))
    model_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 7))
    recorded_data = False
    markovian = False
    reset_policy = True

    model = CESModel(n_parallel=n_parallel, n_elbo_steps=1000,
                     n_elbo_samples=10, markovian=markovian)
    env = GarageEnv(
        normalize(
            DesignEnv(design_space, model_space, model, budget),
            normalize_obs=False
        )
    )
    posteriors.append(env.get_obs()[:1])

    def make_policy():
        return TanhGaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=[layer_size, layer_size],
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            init_std=np.sqrt(1 / 3),
            min_std=np.exp(-20.),
            max_std=np.exp(0.),
        )

    def make_q_func():
        return ContinuousMLPQFunction(env_spec=env.spec,
                                      hidden_sizes=[layer_size, layer_size],
                                      hidden_nonlinearity=F.relu)

    policy = make_policy()
    qf1 = make_q_func()
    qf2 = make_q_func()
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    output_dir = "/".join(logger._outputs[1]._log_file.name.split("/")[:-1])
    output_file = output_dir + "/posteriors.npz"

    # repeat for each experiment
    for i in range(seq_length):
        runner = LocalRunner(snapshot_config=ctxt)
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

        sac.to()
        runner.setup(algo=sac, env=env, sampler_cls=LocalSampler,
                     worker_class=VectorWorker)
        runner.train(n_epochs=n_rl_itr, batch_size=n_parallel * budget)
        if i == seq_length:
            break

        # conduct experiment
        prior = eval_env.reset(n_parallel=n_parallel)
        # make sure this is the right shape
        design = policy.get_action(prior)[1]['mean']
        design = np.array(lexpand(torch.tensor(design[0]), n_parallel))
        d_shape = (n_parallel,) + env.action_space.shape[1:]

        # if we want to replay previously seen experiments
        if recorded_data:
            rl_data = np.load("/home/bla363/boed/Experiments/data/local/experiment/sac_ces_431/posteriors.npz")
            design_array = torch.tensor(rl_data['normalised_designs'])
            design = np.array(lexpand(design_array[i], n_parallel))

        # pyro.set_rng_seed(10)
        posterior, infogain, _, info = eval_env.step(design.reshape(d_shape))
        normalised_designs.append(design[:1])
        ys.append(info['y'])
        posterior, infogain = posterior[:1], infogain[:1]
        posteriors.append(posterior)
        infogains.append(infogain)
        denormalised_designs.append(np.array(model.ds[:1, 0, -1]))

        # save experiment results
        np.savez_compressed(
            output_file,
            posteriors=np.concatenate(posteriors),
            infogains=np.concatenate(infogains),
            normalised_designs=np.concatenate(normalised_designs),
            denormalised_designs=np.concatenate(denormalised_designs),
            ys=np.stack(ys)
        )

        # prepare MDP for next experiment
        init_rho = lexpand(torch.tensor(posterior[:, :2]), n_parallel)
        init_alpha = lexpand(torch.tensor(posterior[:, 2:5]), n_parallel)
        init_mu = lexpand(torch.tensor(posterior[0, 5:6]), n_parallel)
        init_sig = lexpand(torch.tensor(posterior[0, 6:7]), n_parallel)
        init_rho_guide = init_rho
        init_alpha_guide = init_alpha
        init_mu_guide = init_mu
        init_sig_guide = init_sig
        if markovian:
            init_rho_model = init_rho
            init_alpha_model = init_alpha
            init_mu_model = init_mu
            init_sig_model = init_sig
        else:
            init_rho_model = None
            init_alpha_model = None
            init_mu_model = None
            init_sig_model = None
        model = CESModel(init_rho_model=init_rho_model,
                         init_alpha_model=init_alpha_model,
                         init_mu_model=init_mu_model,
                         init_sig_model=init_sig_model,
                         init_rho_guide=init_rho_guide,
                         init_alpha_guide=init_alpha_guide,
                         init_mu_guide=init_mu_guide,
                         init_sig_guide=init_sig_guide,
                         init_ys=model.ys,
                         init_ds=model.ds,
                         n_parallel=n_parallel, n_elbo_steps=1000,
                         n_elbo_samples=10, markovian=markovian)
        env.close()
        eval_env.close()
        env = GarageEnv(
            normalize(
                DesignEnv(design_space, model_space, model, budget),
                normalize_obs=False
            )
        )
        if reset_policy:
            policy = make_policy()
            qf1 = make_q_func()
            qf2 = make_q_func()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default="1", type=int)
    parser.add_argument("--n-parallel", default="100", type=int)
    parser.add_argument("--budget", default="1", type=int)
    parser.add_argument("--seq-length", default="20", type=int)
    parser.add_argument("--n-rl-itr", default="50", type=int)
    args = parser.parse_args()
    exp_id = args.id
    sac_ces(n_parallel=args.n_parallel, budget=args.budget,
            seq_length=args.seq_length, n_rl_itr=args.n_rl_itr,
            seed=seeds[exp_id - 1])
