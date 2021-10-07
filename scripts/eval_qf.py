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


def pad_obs(maxlen, obs):
    pad_shape = list(obs.shape)
    pad_shape[1] = maxlen - pad_shape[1]
    pad = torch.zeros(pad_shape)
    padded_obs = torch.cat([obs, pad], dim=1)
    mask = torch.cat([torch.ones_like(obs), pad], dim=1)[..., :1]
    return padded_obs, mask


def main(src, n_contrastive_samples, n_parallel, seq_length, seed, bound_type,
         myopic, dest):
    # set up environment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    deterministic.set_seed(seed)
    torch.set_printoptions(threshold=int(1e10))
    loss_fn = F.mse_loss

    # load model
    data = joblib.load(src)
    print(f"loaded data from {src}")
    if hasattr(data['algo'], '_sampler'):
        del data['algo']._sampler
    torch.cuda.empty_cache()
    algo, env = data['algo'], data['env']
    pi = algo.policy
    qf1, qf2 = algo._qf1, algo._qf2
    env.env.l = n_contrastive_samples
    env.env.n_parallel = n_parallel
    env.env.bound_type = bound_type

    errs1, errs2, errsmin = [], [], []
    est_qs, pred_qs = [], []
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
        errsmin.append([])
        est_qs.append([])
        pred_qs.append([])


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
            if myopic:
                true_q = sub_rews[0].mean()
            else:
                true_q = sum(sub_rews).mean()
            # compute err(s_ij, a_ij)
            padded_obs, mask = pad_obs(env.budget, cur_obs[i:i+1])
            reshaped_act = cur_act[i].reshape(1, -1)
            q1_pred = qf1(padded_obs, reshaped_act).squeeze()
            q2_pred = qf2(padded_obs, reshaped_act).squeeze()
            qmin_pred = torch.min(q1_pred, q2_pred)
            pred_qs[-1].append(qmin_pred.detach())
            est_qs[-1].append(true_q)
            # print(true_q.item(),
            #       qf1(padded_obs, reshaped_act, mask).squeeze().item(),
            #       qf1(cur_obs[i:i+1], reshaped_act).squeeze().item(),
            #       qf1(padded_obs, reshaped_act).squeeze().item())
            # errs1[-1].append(loss_fn(q1_pred, true_q))
            # errs2[-1].append(loss_fn(q2_pred, true_q))
            # errsmin[-1].append(loss_fn(qmin_pred, true_q))

        # advance trajectories to the next timestep
        set_state(env.env, cur_hist, cur_logprod, cur_last_logsumprod)
        env._step_cnt = j
        es = env.step(cur_act)
        obs, rew = es.observation, es.reward

    pred_qs = torch.tensor(pred_qs).cpu().numpy()
    est_qs = torch.tensor(est_qs).cpu().numpy()
    sq_err = np.square(pred_qs - est_qs)
    np.savez_compressed(dest, pred_qs=pred_qs, est_qs=est_qs, sq_err=sq_err)
    print(f"mean Q = {est_qs.mean(axis=-1)}")
    print(f"mean pred = {pred_qs.mean(axis=-1)}")
    print(f"mse = {sq_err.mean(axis=-1)}")
    print(f"se sd = {sq_err.std(axis=-1)}")
    # errs1 = torch.tensor(errs1).cpu().numpy()
    # errs2 = torch.tensor(errs2).cpu().numpy()
    # errsmin = torch.tensor(errsmin).cpu().numpy()
    # print("qf1 error: ")
    # print(f"mean = {errs1.mean(axis=1)}\t SD = {errs1.std()} \n"
    #       f"median = {np.percentile(errs1, 50)}")
    # print("qf2 error: ")
    # print(f"mean = {errs2.mean(axis=1)}\t SD = {errs2.std()} \n"
    #       f"median = {np.percentile(errs2, 50)}")
    # print("qfmin error: ")
    # print(f"mean = {errsmin.mean(axis=1)}\t SD = {errsmin.std()} \n"
    #       f"median = {np.percentile(errsmin, 50)}")


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
    parser.add_argument("--myopic", action="store_true")
    parser.add_argument("--bound_type", default="lower", type=str.lower,
                        choices=["lower", "upper", "terminal"])
    args = parser.parse_args()
    bound_type = {
        "lower": LOWER, "upper": UPPER, "terminal": TERMINAL}[args.bound_type]
    main(args.src, args.n_contrastive_samples, args.n_parallel, args.seq_length,
         args.seed, bound_type, args.myopic, args.dest)
