import argparse
import sys

import joblib
import pickle
import numpy as np
import torch

from time import time
from pyro.contrib.util import lexpand
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from garage.experiment import deterministic


if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')


def main(src, results, dest, n_contrastive_samples, n_parallel,
         seq_length, edit_type, n_samples, seed, bound_type):
    deterministic.set_seed(seed)
    if edit_type != 'a' and edit_type != 'w':
        sys.exit(f"inadmissible edit_type: {edit_type}")
    torch.set_printoptions(threshold=int(1e10))
    data = joblib.load(src)
    print(f"loaded data from {src}")
    if hasattr(data['algo'], '_sampler'):
        del data['algo']._sampler
    torch.cuda.empty_cache()
    algo, env = data['algo'], data['env']
    pi = algo.policy
    env.env.l = n_contrastive_samples
    env.env.n_parallel = n_parallel
    env.env.bound_type = bound_type
    rewards = []
    rep = n_samples // env.env.n_parallel
    print(f"{n_samples} / {env.env.n_parallel} = {rep} iterations to run")
    t0 = time()
    random = False
    if results is None:
        for j in range(rep):
            obs, _ = env.reset(n_parallel=n_parallel)
            # print("\n", env.env.theta0['theta'][0, 0], "\n")
            rewards.append([])
            for i in range(seq_length):
                # exp_obs = lexpand(obs, 100)
                act, dist_info = pi.get_actions(obs)
                # act = dist_info['mean']
                # opt_index = torch.argmax(
                #     torch.min(qf1(exp_obs, act), qf2(exp_obs, act)),
                #     dim=0,
                #     keepdim=True)
                # opt_index = opt_index.expand((1,) + act.shape[1:])
                # act = torch.gather(act, 0, opt_index).squeeze()
                if random:
                    # act = env.action_space.sample((n_parallel,))/8.
                    low, high = env.action_space.bounds
                    act_dist = torch.distributions.Normal(
                        torch.zeros_like(low), torch.ones_like(high))
                    act = act_dist.sample((n_parallel,))/8.
                act = act.reshape(env.env.n_parallel, 1, 1, -1)
                # print(f"act {act[0] * 4}")
                # print(f"mean {dist_info['mean'][0] * 4}")
                # print(f"std {dist_info['log_std'][0].exp() * 4}")
                es = env.step(act)
                obs, reward = es.observation, es.reward
                # obs_lb, obs_ub = env.observation_space.bounds
                # print(f"obs {obs[0][-1] * (obs_ub - obs_lb) + obs_lb}")
                # print(f"reward {reward[0]}")
                if bound_type == UPPER:
                    reward *= -1
                rewards[-1].append(reward)
            rewards[-1] = torch.stack(rewards[-1])
    else:
        with open(results, 'rb') as results_file:
            ys = []
            designs = []
            for i in range(seq_length):
                data = pickle.load(results_file)
                if i == 0:
                    if not hasattr(data['theta0'], "items"):
                        data['theta0'] = {'theta': data['theta0']}
                    theta0 = {k: v.cuda() for k, v in data['theta0'].items()}
                ys.append(data['y'].cuda())
                designs.append(data['d_star_design'].cuda())

            for j in range(rep):
                env.reset(n_parallel=n_parallel)
                for k, v in theta0.items():
                    env.env.thetas[k][0] = \
                        v[j * n_parallel:(j + 1) * n_parallel]
                rewards.append([])
                for i in range(seq_length):
                    y = ys[i]
                    design = designs[i]
                    reward = env.env.get_reward(
                        y[j*n_parallel:(j+1)*n_parallel],
                        design[j*n_parallel:(j+1)*n_parallel])
                    rewards[-1].append(reward)
                rewards[-1] = torch.stack(rewards[-1])
    rewards = torch.cat(rewards, dim=1)#.squeeze()
    print(rewards.shape)
    cumsum_rewards = torch.cumsum(rewards, dim=0)
    sum_rewards = torch.sum(rewards, dim=0)
    print(cumsum_rewards.shape)
    print(cumsum_rewards.transpose(1, 0))
    t1 = time()
    print(f"compute time {t1-t0} seconds")
    print(f"saving results to {dest}")
    with open(dest, edit_type) as destfile:
        destfile.writelines("\n".join([
            src,
            str(sum_rewards.mean().item()),
            str(sum_rewards.std().item() / np.sqrt(sum_rewards.numel())),
            str(cumsum_rewards.transpose(1, 0))
        ]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str)
    parser.add_argument("--results", default=None, type=str)
    parser.add_argument("--dest", type=str)
    parser.add_argument("--n_contrastive_samples", default=int(1e8), type=int)
    parser.add_argument("--n_parallel", default=1, type=int)
    parser.add_argument("--n_samples", default=100, type=int)
    parser.add_argument("--seq_length", default=20, type=int)
    parser.add_argument("--edit_type", default="a", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--bound_type", default="lower", type=str.lower,
                        choices=["lower", "upper", "terminal"])
    args = parser.parse_args()
    bound_type = {
        "lower": LOWER, "upper": UPPER, "terminal": TERMINAL}[args.bound_type]
    main(args.src, args.results, args.dest, args.n_contrastive_samples,
         args.n_parallel, args.seq_length, args.edit_type, args.n_samples,
         args.seed, bound_type)
