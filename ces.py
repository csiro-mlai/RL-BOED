import argparse
import datetime
import joblib
import logging
from torch.distributions import transform_to
import os
import pickle
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.optim as optim
import subprocess
import time
import torch

from ces_gradients import PosteriorGuide, LinearPosteriorGuide
from contextlib import ExitStack
from functools import partial
from pyro.contrib.oed.eig import marginal_eig, elbo_learn, nmc_eig, pce_eig, \
    opt_eig_ape_loss
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss, \
    differentiable_pce_eig, _differentiable_ace_eig_loss
from pyro.contrib.util import iter_plates_to_shape, lexpand, rexpand, rmv
from pyro.util import is_bad

# TODO read from torch float spec
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
epsilon = torch.tensor(2 ** -22)


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


def make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma, observation_sd,
                   observation_label="y"):
    def ces_model(design):
        # pyro.set_rng_seed(10)
        if is_bad(design):
            raise ArithmeticError("bad design, contains nan or inf")
        batch_shape = design.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)
            rho_shape = batch_shape + (rho_concentration.shape[-1],)
            rho = 0.01 + 0.99 * pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape))).select(-1, 0)
            alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
            alpha = pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
            slope = pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape), slope_sigma.expand(batch_shape)))
            rho, slope = rexpand(rho, design.shape[-2]), rexpand(slope, design.shape[-2])
            d1, d2 = design[..., 0:3], design[..., 3:6]
            U1rho = (rmv(d1.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
            U2rho = (rmv(d2.pow(rho.unsqueeze(-1)), alpha)).pow(1. / rho)
            mean = slope * (U1rho - U2rho)
            sd = slope * observation_sd * (1 + torch.norm(d1 - d2, dim=-1, p=2))

            logging.debug('rho max {} min {}'.format(rho.max().item(), rho.min().item()))
            logging.debug('latent samples: rho {} alpha {} slope mean {} slope median {}'.format(
                rho.mean().item(), alpha.mean().item(), slope.mean().item(), slope.median().item()))
            logging.debug('mean: mean {} sd {} min {} max {}'.format(
                mean.mean().item(), mean.std().item(), mean.min().item(), mean.max().item()))
            logging.debug('sd: mean {}, sd {}, min {}, max {}'.format(sd.mean(), sd.std(), sd.min(), sd.max()))

            emission_dist = dist.CensoredSigmoidNormal(mean, sd, 1 - epsilon, epsilon).to_event(1)
            y = pyro.sample(observation_label, emission_dist)
            return y

    return ces_model


def make_learn_xi_model(model):
    def model_learn_xi(design_prototype):
        design = pyro.param("xi")
        design = design.expand(design_prototype.shape)
        return model(design)

    return model_learn_xi


def elboguide(design, dim=10):
    # pyro.set_rng_seed(10)
    rho_concentration = pyro.param("rho_concentration", torch.ones(dim, 1, 2),
                                   constraint=torch.distributions.constraints.positive)
    alpha_concentration = pyro.param("alpha_concentration", torch.ones(dim, 1, 3),
                                     constraint=torch.distributions.constraints.positive)
    slope_mu = pyro.param("slope_mu", torch.ones(dim, 1))
    slope_sigma = pyro.param("slope_sigma", 3. * torch.ones(dim, 1),
                             constraint=torch.distributions.constraints.positive)
    batch_shape = design.shape[:-2]
    with ExitStack() as stack:
        for plate in iter_plates_to_shape(batch_shape):
            stack.enter_context(plate)
        rho_shape = batch_shape + (rho_concentration.shape[-1],)
        pyro.sample("rho", dist.Dirichlet(rho_concentration.expand(rho_shape)))
        alpha_shape = batch_shape + (alpha_concentration.shape[-1],)
        pyro.sample("alpha", dist.Dirichlet(alpha_concentration.expand(alpha_shape)))
        pyro.sample("slope", dist.LogNormal(slope_mu.expand(batch_shape),
                                            slope_sigma.expand(batch_shape)))


def marginal_guide(mu_init, log_sigma_init, shape, label):
    def guide(design, observation_labels, target_labels):
        mu = pyro.param("marginal_mu", mu_init * torch.ones(*shape))
        log_sigma = pyro.param("marginal_log_sigma", log_sigma_init * torch.ones(*shape))
        ends = pyro.param("marginal_ends", 1. / 3 * torch.ones(*shape, 3),
                          constraint=torch.distributions.constraints.simplex)
        response_dist = dist.CensoredSigmoidNormalEnds(
            loc=mu, scale=torch.exp(log_sigma), upper_lim=1. - epsilon, lower_lim=epsilon,
            p0=ends[..., 0], p1=ends[..., 1], p2=ends[..., 2]
        ).to_event(1)
        pyro.sample(label, response_dist)

    return guide


def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))

    return new_loss


def main(num_steps, num_parallel, experiment_name, typs, seed, lengthscale,
         num_gradient_steps, num_samples, num_contrast_samples, num_acquisition,
         obs_sd, loglevel, policy_src, estimate_eig):
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
    observation_sd = torch.tensor(obs_sd)

    for typ in typs:
        logging.info("Type {}".format(typ))
        pyro.clear_param_store()
        if seed >= 0:
            pyro.set_rng_seed(seed)
        else:
            seed = int(torch.rand(tuple()) * 2 ** 30)
            pyro.set_rng_seed(seed)
        marginal_mu_init, marginal_log_sigma_init = 0., 6.
        oed_n_samples, oed_n_steps, oed_final_n_samples, oed_lr = 10, 1250, 2000, [0.1, 0.01, 0.001]
        elbo_n_samples, elbo_n_steps, elbo_lr = 10, 1000, 0.04
        num_bo_steps = 4
        design_dim = 6

        guide = marginal_guide(marginal_mu_init, marginal_log_sigma_init, (num_parallel, num_acquisition, 1), "y")

        rho_concentration = torch.ones(num_parallel, 1, 2)
        alpha_concentration = torch.ones(num_parallel, 1, 3)
        slope_mu, slope_sigma = torch.ones(num_parallel, 1), 3. * torch.ones(num_parallel, 1)

        true_model = pyro.condition(make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma,
                                                   observation_sd),
                                    {"rho": torch.tensor([.9, .1]), "alpha": torch.tensor([.2, .3, .5]),
                                     "slope": torch.tensor(10.)})
        if estimate_eig:
            print("estimate_eig")
            data = joblib.load(policy_src)
            eval_env = data['env'].env.env
            eval_env.l = int(1e4)
            eval_env.reset(n_parallel=eval_env.n_parallel)
            spce = 0

        d_star_designs = torch.tensor([])
        ys = torch.tensor([])

        for step in range(num_steps):
            logging.info("Step {}".format(step))
            model = make_ces_model(rho_concentration, alpha_concentration, slope_mu, slope_sigma, observation_sd)
            results = {'typ': typ, 'step': step, 'git-hash': get_git_revision_hash(), 'seed': seed,
                       'lengthscale': lengthscale, 'observation_sd': observation_sd,
                       'num_gradient_steps': num_gradient_steps, 'num_samples': num_samples,
                       'num_contrast_samples': num_contrast_samples, 'num_acquisition': num_acquisition}

            # Design phase
            t = time.time()

            if typ in ['marginal', 'nmc']:
                # Suggested num_acquisition = 50
                if num_acquisition < 50:
                    raise ValueError("Setting num_acquisition too low")
                # Initialization
                noise = torch.tensor(0.2).pow(2)
                # X = 100*rexpand(torch.rand((num_parallel, num_acq)), 4)
                X = .01 + 99.99 * torch.rand((num_parallel, num_acquisition, 1, design_dim))

                if typ == 'marginal':
                    def f(X):
                        n_steps = oed_n_steps // len(oed_lr)
                        for lr in oed_lr:
                            marginal_eig(model, X, observation_labels=["y"], target_labels=["rho", "alpha", "slope"],
                                         num_samples=oed_n_samples, num_steps=n_steps, guide=guide,
                                         optim=optim.Adam({"lr": lr}))
                        return marginal_eig(model, X, observation_labels=["y"], target_labels=["rho", "alpha", "slope"],
                                            num_samples=oed_n_samples, num_steps=1, guide=guide,
                                            final_num_samples=oed_final_n_samples, optim=optim.Adam({"lr": 1e-6}))
                elif typ == 'nmc':
                    def f(X):
                        return torch.cat([nmc_eig(model, X[:, 25 * i:25 * (i + 1), ...], ["y"],
                                                  ["rho", "alpha", "slope"], N=70 * 70, M=70)
                                          for i in range(X.shape[1] // 25)], dim=1)

                y = f(X)

                # Random search
                # # print(y.mean(1), y.max(1), y.min(1), y.std(1))
                # d_star_index = torch.argmax(y, dim=1)
                # # print(d_star_index.shape)
                # # print(d_star_index)
                # d_star_design = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2)

                # GPBO
                y = y.detach().clone()
                kernel = gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(lengthscale),
                                             variance=y.var(unbiased=True))
                X = X.squeeze(-2)
                constraint = torch.distributions.constraints.interval(1e-6, 100.)

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
                d_star_design = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2).unsqueeze(-2)

            elif typ in ['posterior-grad', 'pce-grad', 'ace-grad']:
                model_learn_xi = make_learn_xi_model(model)
                grad_start_lr, grad_end_lr = 0.001, 0.001

                if typ == 'posterior-grad':

                    # Suggested num_gradient_steps = 5000
                    posterior_guide = LinearPosteriorGuide((num_parallel, num_acquisition))
                    posterior_guide.set_prior(rho_concentration, alpha_concentration, slope_mu, slope_sigma)
                    loss = _differentiable_posterior_loss(model_learn_xi, posterior_guide, ["y"],
                                                          ["rho", "alpha", "slope"])

                elif typ == 'pce-grad':

                    # Suggested num_gradient_steps = 2500
                    eig_loss = lambda d, N, **kwargs: differentiable_pce_eig(
                        model=model_learn_xi, design=d, observation_labels=["y"],
                        target_labels=["rho", "alpha", "slope"],
                        N=N, M=num_contrast_samples, **kwargs)
                    loss = neg_loss(eig_loss)

                elif typ == 'ace-grad':

                    # Suggested num_gradient_steps = 1500
                    posterior_guide = LinearPosteriorGuide((num_parallel, num_acquisition))
                    posterior_guide.set_prior(rho_concentration, alpha_concentration, slope_mu, slope_sigma)
                    eig_loss = _differentiable_ace_eig_loss(model_learn_xi, posterior_guide, num_contrast_samples,
                                                            ["y"], ["rho", "alpha", "slope"])
                    loss = neg_loss(eig_loss)

                constraint = torch.distributions.constraints.interval(1e-6, 100.)
                # xi_init = .01 + 99.99 * torch.rand((num_parallel, num_acquisition, 1, design_dim // 2))
                # xi_init = torch.cat([xi_init, xi_init], dim=-1)
                xi_init = .01 + 99.99 * torch.rand((num_parallel, num_acquisition, 1, design_dim))
                logging.info('init_design {} {}'.format(xi_init.squeeze(), xi_init.shape))
                pyro.param("xi", xi_init, constraint=constraint)
                pyro.get_param_store().replace_param("xi", xi_init, pyro.param("xi"))
                design_prototype = torch.zeros(num_parallel, num_acquisition, 1,
                                               design_dim)  # this is annoying, code needs refactor

                start_lr, end_lr = grad_start_lr, grad_end_lr
                gamma = (end_lr / start_lr) ** (1 / num_gradient_steps)
                scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr},
                                                      'gamma': gamma})
                ape = opt_eig_ape_loss(design_prototype, loss, num_samples=num_samples, num_steps=num_gradient_steps,
                                       optim=scheduler, final_num_samples=500)
                min_ape, d_star_index = torch.min(ape, dim=1)
                logging.info('min loss {}'.format(min_ape))
                results['min loss'] = min_ape
                X = pyro.param("xi").detach().clone()
                d_star_design = X[torch.arange(num_parallel), d_star_index, ...].unsqueeze(-2)

            elif typ == 'rand':
                d_star_design = .01 + 99.99 * torch.rand((num_parallel, 1, 1, design_dim))
                d_star_design = lexpand(d_star_design[0], num_parallel)

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
                            d_star_designs.squeeze(dim=-3),
                            torch.transpose(ys, 1, 2)
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
                d_star_design = torch.tensor(denorm_act)

            elif typ == 'const':
                import numpy as np
                rl_data = np.load("/home/bla363/boed/Experiments/data/local/experiment/sac_ces_654/posteriors.npz")
                design_array = torch.tensor(rl_data['denormalised_designs'])
                # y_array = torch.tensor(rl_data['ys'])

                # design_array = []
                # y_array = []
                # rng_array = []
                # with open("/home/bla363/boed/run_outputs/ces/repr-pce-grad-1.result_stream.pickle", 'rb') as f:
                #     try:
                #         while True:
                #             data = pickle.load(f)
                #             design_array.append(data['d_star_design'])
                #             y_array.append(data['y'])
                #             # rng_array.append(data['rng_state'])
                #     except EOFError:
                #         pass
                # # torch.set_rng_state(rng_array[step])

                d_star_design = lexpand(design_array[step][0], num_parallel)

            results['rng_state'] = torch.get_rng_state()
            # update using only the result of the first experiment
            markovian = False
            if markovian:
                d_star_designs = torch.tensor([])
                ys = torch.tensor([])
                prior = make_ces_model(rho_concentration.detach().clone(),
                                       alpha_concentration.detach().clone(),
                                       slope_mu.detach().clone(),
                                       slope_sigma.detach().clone(),
                                       observation_sd)
            else:
                prior = make_ces_model(torch.ones(num_parallel, 1, 2), torch.ones(num_parallel, 1, 3),
                                       torch.ones(num_parallel, 1), 3. * torch.ones(num_parallel, 1), observation_sd)
            elapsed = time.time() - t
            logging.info('elapsed design time {}'.format(elapsed))
            results['design_time'] = elapsed
            results['d_star_design'] = d_star_design
            logging.info('design {} {}'.format(d_star_design.squeeze(), d_star_design.shape))
            # d_star_design = lexpand(d_star_design[0], num_parallel)
            d_star_designs = torch.cat([d_star_designs, d_star_design], dim=-2)
            # pyro.set_rng_seed(10)
            if estimate_eig:
                y = eval_env.model.run_experiment(d_star_design, eval_env.theta0)
                spce += eval_env.get_reward(y, d_star_design)
                results['spce'] = spce
                if step == 0:
                    results['theta0'] = {
                        k: v.cpu() for k, v in eval_env.theta0.items()}
                logging.info(f"spce {spce} {spce.shape}")
            else:
                y = true_model(d_star_design)
            # if typ == 'const':
            #     y = y * 0 + lexpand(y_array[step][0], num_parallel)
            ys = torch.cat([ys, y], dim=-1)
            logging.info('ys {} {}'.format(ys.squeeze(), ys.shape))
            results['y'] = y

            # pyro.set_rng_seed(10)
            loss = elbo_learn(
                prior, d_star_designs, ["y"], ["rho", "alpha", "slope"], elbo_n_samples, elbo_n_steps,
                partial(elboguide, dim=num_parallel), {"y": ys}, optim.Adam({"lr": elbo_lr})
            )
            mindex = loss.argmin()
            rho_concentration = pyro.param("rho_concentration").detach().data.clone()
            alpha_concentration = pyro.param("alpha_concentration").detach().data.clone()
            slope_mu = pyro.param("slope_mu").detach().data.clone()
            slope_sigma = pyro.param("slope_sigma").detach().data.clone()
            if markovian:
                rho_concentration = pyro.param("rho_concentration")[mindex].detach().data.clone()
                rho_concentration = lexpand(rho_concentration, num_parallel)
                alpha_concentration = pyro.param("alpha_concentration")[mindex].detach().data.clone()
                alpha_concentration = lexpand(alpha_concentration, num_parallel)
                slope_mu = pyro.param("slope_mu")[mindex].detach().data.clone()
                slope_mu = lexpand(slope_mu, num_parallel)
                slope_sigma = pyro.param("slope_sigma")[mindex].detach().data.clone()
                slope_sigma = lexpand(slope_sigma, num_parallel)

            realistic = False
            if realistic:
                rho_concentration = lexpand(rho_concentration[0], num_parallel)
                alpha_concentration = lexpand(alpha_concentration[0], num_parallel)
                slope_mu = lexpand(slope_mu[0], num_parallel)
                slope_sigma = lexpand(slope_sigma[0], num_parallel)
                pstore = pyro.get_param_store()
                pstore["rho_concentration"] = rho_concentration.detach().data.clone()
                pstore["alpha_concentration"] = alpha_concentration.detach().data.clone()
                pstore["slope_mu"] = slope_mu.detach().data.clone()
                pstore["slope_sigma"] = slope_sigma.detach().data.clone()

            logging.info("rho_concentration {} \n alpha_concentration {} \n slope_mu {} \n slope_sigma {}".format(
                rho_concentration.squeeze(), alpha_concentration.squeeze(), slope_mu.squeeze(), slope_sigma.squeeze()))
            results['rho_concentration'], results['alpha_concentration'] = rho_concentration, alpha_concentration
            results['slope_mu'], results['slope_sigma'] = slope_mu, slope_sigma
            for k, v in results.items():
                if hasattr(v, "cpu"):
                    results[k] = v.cpu()

            with open(results_file, 'ab') as f:
                pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CES (Constant Elasticity of Substitution) indifference"
                                                 " iterated experiment design")
    parser.add_argument("--num-steps", nargs="?", default=20, type=int)
    parser.add_argument("--num-parallel", nargs="?", default=10, type=int)
    parser.add_argument("--name", nargs="?", default="", type=str)
    parser.add_argument("--typs", nargs="?", default="rand", type=str)
    parser.add_argument("--seed", nargs="?", default=-1, type=int)
    parser.add_argument("--lengthscale", nargs="?", default=10., type=float)
    parser.add_argument("--loglevel", default="info", type=str)
    parser.add_argument("--num-gradient-steps", default=1000, type=int)
    parser.add_argument("--num-samples", default=10, type=int)
    parser.add_argument("--num-contrast-samples", default=10, type=int)
    parser.add_argument("--num-acquisition", default=8, type=int)
    parser.add_argument("--observation-sd", default=0.005, type=float)
    parser.add_argument("--policy-src", default="", type=str)
    parser.add_argument("--estimate-eig", dest="estimate_eig",
                        action='store_true')
    parser.set_defaults(estimate_eig=False)
    args = parser.parse_args()
    main(args.num_steps, args.num_parallel, args.name, args.typs, args.seed, args.lengthscale,
         args.num_gradient_steps, args.num_samples, args.num_contrast_samples, args.num_acquisition,
         args.observation_sd, args.loglevel, args.policy_src, args.estimate_eig)
