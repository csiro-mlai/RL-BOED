import argparse
import datetime
import joblib
import logging
import os
import pickle
import pyro
import pyro.distributions as dist
import pyro.optim as optim
import subprocess
import time
import torch

from contextlib import ExitStack
from functools import partial
import pyro.contrib.gp as gp
from pyro.contrib.oed.eig import elbo_learn, opt_eig_ape_loss
from pyro.contrib.oed.differentiable_eig import differentiable_pce_eig
from pyro.contrib.util import iter_plates_to_shape, lexpand, rmv
from pyro.envs.adaptive_design_env import AdaptiveDesignEnv, UPPER, LOWER
from pyro.models.adaptive_experiment_model import CESModel
from torch.distributions import LogNormal, Dirichlet, transform_to


# TODO read from torch float spec
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
epsilon = torch.tensor(2 ** -22)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def make_learn_xi_model(model):
    def model_learn_xi(design_prototype):
        design = pyro.param("xi")
        design = design.expand(design_prototype.shape)
        return model(design)

    return model_learn_xi


def elboguide(design, dim=10):
    rho_con = pyro.param("rho_con", torch.ones(dim, 1, 2),
                         constraint=torch.distributions.constraints.positive)
    alpha_con = pyro.param("alpha_con", torch.ones(dim, 1, 3),
                           constraint=torch.distributions.constraints.positive)
    u_mu = pyro.param("u_mu", torch.ones(dim, 1))
    u_sig = pyro.param("u_sig", 3. * torch.ones(dim, 1),
                       constraint=torch.distributions.constraints.positive)
    batch_shape = design.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)
        rho_shape = batch_shape + (rho_con.shape[-1],)
        pyro.sample("rho", dist.Dirichlet(rho_con.expand(rho_shape)))
        alpha_shape = batch_shape + (alpha_con.shape[-1],)
        pyro.sample("alpha", dist.Dirichlet(alpha_con.expand(alpha_shape)))
        pyro.sample("u", dist.LogNormal(u_mu.expand(batch_shape),
                                        u_sig.expand(batch_shape)))


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))

    return new_loss


def main(num_steps, num_parallel, experiment_name, typs, seed, lengthscale,
         num_gradient_steps, num_samples, num_contrast_samples, num_acquisition,
         loglevel, policy_src):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: {}".format(loglevel))
    logging.basicConfig(level=numeric_level)

    output_dir = "run_outputs/ces/"
    if not experiment_name:
        experiment_name = output_dir + "{}".format(datetime.datetime.now().isoformat())
    else:
        experiment_name = output_dir + experiment_name
    results_file = experiment_name + '.result_stream.pickle'
    results_file = os.path.join(os.path.dirname(__file__), results_file)
    try:
        os.remove(results_file)
    except OSError:
        logging.info("File {} does not exist yet".format(results_file))
    typs = typs.split(",")

    for typ in typs:
        logging.info("Type {}".format(typ))
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)
        elbo_n_samples, elbo_n_steps, elbo_lr = 10, 1000, 0.04
        num_bo_steps = 4
        design_dim = 6

        env_lower = AdaptiveDesignEnv(None, torch.zeros(2),
                                CESModel(n_parallel=num_parallel), num_steps,
                                int(1e6), bound_type=LOWER)
        env_upper = AdaptiveDesignEnv(None, torch.zeros(2),
                                CESModel(n_parallel=num_parallel), num_steps,
                                int(1e6), bound_type=UPPER)
        env_lower.reset(num_parallel)
        env_upper.reset(num_parallel)
        spce, snmc = 0, 0
        model = CESModel(n_parallel=num_parallel)
        init_entropy = Dirichlet(model.rho_con_model).entropy() +\
                       Dirichlet(model.alpha_con_model).entropy() +\
                       LogNormal(model.u_mu_model, model.u_sig_model).entropy()
        true_theta = env_lower.theta0
        env_upper.theta0, env_upper.thetas = env_lower.theta0, env_lower.thetas
        d_stars = torch.tensor([])
        y_stars = torch.tensor([])
        results = {
            'typ': typ, 'git-hash': get_git_revision_hash(), 'seed': seed,
            'num_samples': num_samples,
            'num_gradient_steps': num_gradient_steps,
            'num_contrast_samples': num_contrast_samples,
            'num_acquisition': num_acquisition
        }

        for step in range(num_steps):
            logging.info("Step {}".format(step))

            # Design phase
            t0 = time.time()

            if typ == 'bo':
                grad_start_lr, grad_end_lr = 0.001, 0.001
                noise = torch.tensor(0.2).pow(2)
                constraint = torch.distributions.constraints.interval(1e-6, 100.)
                X = .01 + 99.99 * torch.rand((num_parallel, num_acquisition, 1, design_dim))
                eig_loss = lambda d, N, **kwargs: differentiable_pce_eig(
                    model=model.make_model(), design=d, observation_labels=["y"],
                    target_labels=["rho", "alpha", "u"],
                    N=N, M=num_contrast_samples, **kwargs)
                loss = neg_loss(eig_loss)
                start_lr, end_lr = grad_start_lr, grad_end_lr
                gamma = (end_lr / start_lr) ** (1 / num_gradient_steps)
                scheduler = pyro.optim.ExponentialLR({
                    'optimizer': torch.optim.Adam,
                    'optim_args': {'lr': start_lr},
                    'gamma': gamma})

                def f(X):
                    return opt_eig_ape_loss(
                        X, eig_loss, num_samples=num_samples,
                        num_steps=0, optim=scheduler,
                        final_num_samples=500
                    )

                y = f(X).detach().clone()
                kernel = gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(lengthscale),
                                             variance=y.var(unbiased=True))
                X = X.squeeze(-2)

                for i in range(num_bo_steps):
                    Kff = kernel(X)
                    Kff += noise * torch.eye(Kff.shape[-1])
                    Lff = Kff.cholesky(upper=False)
                    Xinit = .01 + 99.99 * torch.rand((num_parallel, num_acquisition, design_dim))
                    unconstrained_Xnew = transform_to(constraint).inv(Xinit).detach().clone().requires_grad_(True)
                    minimizer = torch.optim.LBFGS([unconstrained_Xnew], max_eval=20)

                    def gp_ucb1():
                        minimizer.zero_grad()
                        Xnew = transform_to(constraint)(unconstrained_Xnew)
                        # Xnew.register_hook(lambda x: print('Xnew grad', x))
                        KXXnew = kernel(X, Xnew)
                        LiK = torch.triangular_solve(KXXnew, Lff, upper=False)[0]
                        Liy = torch.triangular_solve(y.unsqueeze(-1).clamp(max=20.), Lff, upper=False)[0]
                        mean = rmv(LiK.transpose(-1, -2), Liy.squeeze(-1))
                        KXnewXnew = kernel(Xnew)
                        var = (KXnewXnew - LiK.transpose(-1, -2).matmul(LiK)).diagonal(dim1=-2, dim2=-1)
                        ucb = -(mean + 2 * var.sqrt())
                        loss = ucb.sum()
                        torch.autograd.backward(unconstrained_Xnew,
                                                torch.autograd.grad(loss, unconstrained_Xnew, retain_graph=True))
                        return loss

                    minimizer.step(gp_ucb1)
                    X_acquire = transform_to(constraint)(unconstrained_Xnew).detach().clone()
                    # print('X_acquire', X_acquire)
                    y_acquire = f(X_acquire.unsqueeze(-2)).detach().clone()
                    # print('y_acquire', y_acquire)

                    X = torch.cat([X, X_acquire], dim=1)
                    y = torch.cat([y, y_acquire], dim=1)

                max_eig, d_star_index = torch.max(y, dim=1)
                logging.info('max EIG {}'.format(max_eig))
                results['max EIG'] = max_eig
                d_star = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2).unsqueeze(-2)

            if typ == 'pce':
                model_learn_xi = make_learn_xi_model(model.make_model())
                grad_start_lr, grad_end_lr = 0.001, 0.001

                # Suggested num_gradient_steps = 2500
                eig_loss = lambda d, N, **kwargs: differentiable_pce_eig(
                    model=model_learn_xi, design=d, observation_labels=["y"],
                    target_labels=["rho", "alpha", "u"],
                    N=N, M=num_contrast_samples, **kwargs)
                loss = neg_loss(eig_loss)
                constraint = torch.distributions.constraints.interval(1e-6,
                                                                      100.)
                xi_init = .01 + 99.99 * torch.rand(
                    (num_parallel, num_acquisition, 1, design_dim))
                logging.info(f'init_design {xi_init.squeeze()} {xi_init.shape}')
                pyro.param("xi", xi_init, constraint=constraint)
                pyro.get_param_store().replace_param(
                    "xi", xi_init, pyro.param("xi"))
                design_prototype = torch.zeros(
                    num_parallel, num_acquisition, 1, design_dim)

                start_lr, end_lr = grad_start_lr, grad_end_lr
                gamma = (end_lr / start_lr) ** (1 / num_gradient_steps)
                scheduler = pyro.optim.ExponentialLR({
                    'optimizer': torch.optim.Adam,
                    'optim_args': {'lr': start_lr},
                    'gamma': gamma})
                ape = opt_eig_ape_loss(
                    design_prototype, loss, num_samples=num_samples,
                    num_steps=num_gradient_steps, optim=scheduler,
                    final_num_samples=500
                )
                min_ape, d_star_index = torch.min(ape, dim=1)
                logging.info('min loss {}'.format(min_ape))
                results['min loss'] = min_ape
                X = pyro.param("xi").detach().clone()
                d_star = X[torch.arange(num_parallel), d_star_index, ...]
                d_star = d_star.unsqueeze(-2)

            elif typ == 'rand':
                d_star = .01 + 99.99 * torch.rand((1, 1, 1, design_dim))
                d_star = lexpand(d_star[0], num_parallel)

            elif typ == "policy":
                if step == 0:
                    data = joblib.load(policy_src)
                    algo, env = data['algo'], data['env']
                    pi, qf1, qf2 = algo.policy, algo._qf1, algo._qf2
                    norm_env = env.env
                    o_lb = norm_env.observation_space.low
                    o_ub = norm_env.observation_space.high
                    norm_obs = env.reset(n_parallel=num_parallel)
                else:
                    obs = torch.cat(
                        [
                            d_stars.squeeze(dim=-3),
                            torch.transpose(y_stars, 1, 2)
                        ],
                        dim=-1
                    )
                    norm_obs = (obs - o_lb) / (o_ub - o_lb)
                act = pi.get_actions(norm_obs)[-1]['mean'].reshape(
                    num_parallel, 1, 1, design_dim)
                tiled_obs = norm_obs.unsqueeze(0).repeat(100, 1, 1, 1)
                # act = pi.get_actions(tiled_obs)[0]
                # max_q = torch.min(
                #     qf1(tiled_obs, act), qf2(tiled_obs, act)
                # ).squeeze().argmax(dim=0)
                # act = act[max_q, torch.arange(len(max_q))].reshape(
                #     num_parallel, 1, 1, design_dim)
                lb, ub = norm_env.action_space.low, norm_env.action_space.high
                denorm_act = lb + (act + 1) * 0.5 * (ub - lb)
                d_star = torch.tensor(denorm_act)

            elapsed = time.time() - t0
            logging.info('elapsed design time {}'.format(elapsed))
            results['rng_state'] = torch.get_rng_state()
            results['time'] = elapsed
            results['d_star'] = d_star
            logging.info('design {} {}'.format(d_star.squeeze(), d_star.shape))

            d_stars = torch.cat([d_stars, d_star], dim=-2)
            y_star = model.run_experiment(d_star, true_theta)
            y_stars = torch.cat([y_stars, y_star], dim=-1)
            logging.info(f'y_stars {y_stars.squeeze()} {y_stars.shape}')
            results['y'] = y_star

            # learn posterior with VI
            t1 = time.time()
            if typ in ['pce', 'bo']:
                model.reset(num_parallel)
                prior = model.make_model()
                loss = elbo_learn(
                    prior, d_stars, ["y"], ["rho", "alpha", "u"], elbo_n_samples,
                    elbo_n_steps, partial(elboguide, dim=num_parallel),
                    {"y": y_stars}, optim.Adam({"lr": elbo_lr})
                )
                rho_con = pyro.param("rho_con").detach().data.clone()
                alpha_con = pyro.param("alpha_con").detach().data.clone()
                u_mu = pyro.param("u_mu").detach().data.clone()
                u_sig = pyro.param("u_sig").detach().data.clone()
                model.rho_con_model, model.alpha_con_model = rho_con, alpha_con
                model.u_mu_model, model.u_sig_model = u_mu, u_sig
                entropy = Dirichlet(model.rho_con_model).entropy() +\
                    Dirichlet(model.alpha_con_model).entropy() +\
                    LogNormal(model.u_mu_model, model.u_sig_model).entropy()
                logging.info(f'EIG {(init_entropy - entropy).squeeze()}')
                # logging.info(f"rho_con {rho_con.squeeze()} \n alpha_con "
                #              f"{alpha_con.squeeze()} \n u_mu {u_mu.squeeze()} \n"
                #              f"u_sig {u_sig.squeeze()}")
                results['rho_con'], results['alpha_con'] = rho_con, alpha_con
                results['u_mu'], results['u_sig'] = u_mu, u_sig

            results['time'] = time.time() - t0
            logging.info(f'posterior learning time {time.time() - t1}')
            # estimate EIG with sPCE
            spce += env_lower.get_reward(y_star, d_star)
            snmc += env_upper.get_reward(y_star, d_star)
            results['spce'] = spce
            logging.info(f"spce {spce} {spce.shape}")
            results['snmc'] = snmc
            logging.info(f"snmc {snmc} {snmc.shape}")
            for k, v in results.items():
                if hasattr(v, "cpu"):
                    results[k] = v.cpu()

            with open(results_file, 'ab') as f:
                pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CES (Constant Elasticity of Substitution) indifference"
                                                 " iterated experiment design")
    parser.add_argument("--num-steps", nargs="?", default=10, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=10, type=int)
    parser.add_argument("--name", nargs="?", default="", type=str)
    parser.add_argument("--typs", nargs="?", default="rand", type=str)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--lengthscale", nargs="?", default=10., type=float)

    parser.add_argument("--loglevel", default="info", type=str)
    parser.add_argument("--num-gradient-steps", default=2500, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--num-contrast-samples", default=10, type=int)
    parser.add_argument("--num-acquisition", default=1, type=int)
    parser.add_argument("--policy-src", default="", type=str)
    args = parser.parse_args()
    main(args.num_steps, args.num_parallel, args.name, args.typs, args.seed, args.lengthscale,
         args.num_gradient_steps, args.num_samples, args.num_contrast_samples,
         args.num_acquisition, args.loglevel, args.policy_src)
