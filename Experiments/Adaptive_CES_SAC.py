import argparse
import torch
import pyro
import numpy as np

from dowel import logger
from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic, LocalRunner
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from gym import spaces
from pyro.algos import SAC
from pyro.contrib.util import lexpand
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
def sac_ces(ctxt=None, n_parallel=1, budget=1, seq_length=1, n_rl_itr=1,
            n_cont_samples=10, seed=0):
    # one-time setup
    deterministic.set_seed(seed)
    pyro.set_rng_seed(seed)
    denormalised_designs = []
    normalised_designs = []
    ys = []
    layer_size = 128
    design_space = spaces.Box(low=0.01, high=100, shape=(1, 1, 1, 6))
    obs_space = BatchBox(low=np.array([-1.] * 6 + [0.]), high=np.ones((7,)))
    recorded_data = False
    reset_policy = True

    model = CESModel(n_parallel=n_parallel, n_elbo_steps=1000,
                     n_elbo_samples=10)

    def make_env(design_space, obs_space, model, budget, n_cont_samples,
                 true_model=None):
        env = GarageEnv(
            normalize(
                AdaptiveDesignEnv(design_space, obs_space, model, budget,
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

    env = make_env(design_space, obs_space, model, budget, n_cont_samples)
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
        eval_env = make_env(design_space, obs_space, model, budget,
                            n_cont_samples, true_model=true_model)
        replay_buffer = ListBuffer(capacity_in_transitions=int(1e6))

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
        denormalised_designs.append(np.array(model.ds[:1, 0, -1]))

        # save experiment results
        np.savez_compressed(
            output_file,
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
                         n_elbo_samples=10)
        env.close()
        eval_env.close()
        env = make_env(design_space, obs_space, model, budget, n_cont_samples)
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
    parser.add_argument("--n-contr-samples", default="10", type=int)
    args = parser.parse_args()
    exp_id = args.id
    sac_ces(n_parallel=args.n_parallel, budget=args.budget,
            seq_length=args.seq_length, n_rl_itr=args.n_rl_itr,
            n_cont_samples=args.n_contr_samples, seed=seeds[exp_id - 1])
