"""
A script to evaluate the true error of the Q-function estimator

err(s_t, a_t) = E_pi[qf_pi(s_t,a_t) - Q*_pi(s_t,a_t)]

Q*_pi(s_t,a_t) = E[r(s_t,s_t)] if t = T
                 E_pi[ r(s_t,a_t) + sum(gamma^i * r(s_t+l, a_t+l)) ] if t != T

where qf_pi is the estimator and Q*_pi the true Q function
"""


import argparse

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from pyro.contrib.util import lexpand, rexpand
from pyro.envs.adaptive_design_env import LOWER, UPPER, TERMINAL
from garage.experiment import deterministic


def set_state(env, history, log_prod, last_logsumprod):
    env.history = history
    env.log_products = log_prod
    env.last_logsumprod = last_logsumprod


def main(src, n_contrastive_samples, n_parallel, seq_length, seed, bound_type):
    # set up environment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    deterministic.set_seed(seed)
    torch.set_printoptions(threshold=int(1e10))
    loss_fn = F.mse_loss

    # load model
    data = joblib.load(src)
    print(f"loaded data from {src}")
    del data['algo']._sampler
    torch.cuda.empty_cache()
    algo, env = data['algo'], data['env']
    pi = algo.policy
    qf1, qf2 = algo._qf1, algo._qf2
    env.env.l = n_contrastive_samples
    env.env.n_parallel = n_parallel
    env.env.bound_type = bound_type

    errs1, errs2 = [], []
    obs, _ = env.reset(n_parallel=n_parallel)
    for j in range(0, seq_length):
        print(f"begin {j}th timestep")
        # generate n_parallel trajectories up to the j-th timestep
        cur_obs = obs
        cur_act, _ = pi.get_actions(cur_obs)
        cur_act = cur_act.reshape(env.env.n_parallel, 1, 1, -1)
        cur_hist = [e.detach().clone() for e in env.history]
        cur_logprod = env.log_products.detach().clone()
        cur_last_logsumprod = env.last_logsumprod.detach().clone()
        errs1.append([])
        errs2.append([])

        # for trajectory i at timestep j, generate n subtrajectories from j to T
        for i in range(n_parallel):
            # set env state to s_ij
            hist = [lexpand(e[i], n_parallel) for e in cur_hist]
            logprod = rexpand(cur_logprod[:, i], n_parallel).clone()
            logsumprod = rexpand(cur_last_logsumprod[i], n_parallel).clone()
            set_state(env.env, hist, logprod, logsumprod)
            env._step_cnt = j+1
            # execute (possibly off-policy) action a_ij
            act = lexpand(cur_act[i], n_parallel).clone()
            sub_es = env.step(act)
            # follow policy pi for the rest of the trajectory
            sub_obs = sub_es.observation
            sub_rews = [sub_es.reward]
            for k in range(j+1, seq_length):
                sub_act, _ = pi.get_actions(sub_obs)
                sub_act = sub_act.reshape(n_parallel, 1, 1, -1)
                sub_es = env.step(sub_act)
                sub_obs = sub_es.observation
                sub_rews.append(sub_es.reward)

            # compute Q*_pi(s_ij, a_ij) as average return-to-go
            true_q = sum(sub_rews).mean()
            # compute err(s_ij, a_ij)
            q1_pred = qf1(cur_obs[i:i+1], cur_act[i].reshape(1, -1)).squeeze()
            q2_pred = qf2(cur_obs[i:i+1], cur_act[i].reshape(1, -1)).squeeze()
            errs1[-1].append(loss_fn(q1_pred, true_q))
            errs2[-1].append(loss_fn(q2_pred, true_q))

        # advance trajectories to the next timestep
        env._step_cnt = j
        es = env.step(cur_act)
        obs, rew = es.observation, es.reward
        set_state(env.env, cur_hist, cur_logprod, cur_last_logsumprod)

    errs1 = torch.tensor(errs1).numpy()
    errs2 = torch.tensor(errs2).numpy()
    print("qf1 error: ")
    print(f"mean = {errs1.mean()}\t SD = {errs1.std()} \n"
          f"median = {np.percentile(errs1, 50)}")
    print("qf2 error: ")
    print(f"mean = {errs2.mean()}\t SD = {errs2.std()} \n"
          f"median = {np.percentile(errs2, 50)}")


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
    main(args.src, args.n_contrastive_samples, args.n_parallel, args.seq_length,
         args.seed, bound_type)