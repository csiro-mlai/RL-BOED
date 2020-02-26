import argparse
from contextlib import ExitStack
import math
import subprocess
import pickle
import numpy as np
import pyro.contrib.gp as gp
import time

import torch
from torch.distributions import constraints
from torch.nn.functional import softmax

import pyro
import pyro.distributions as dist
import pyro.optim as optim
import pyro.poutine as poutine
from pyro.contrib.oed.eig import pce_eig, _ace_eig_loss
from pyro.contrib.oed.differentiable_eig import _differentiable_posterior_loss
from pyro.contrib.util import rmv, iter_plates_to_shape, lexpand


# get git hash to help with reproducibility
def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'])


# find the exact EIG (up to a constant) for a design [delta, delta..., epsilon, epsilon, ...]
def analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq):
    oneplusd = 1.0 + 1.0 / D
    term1 = 0.5 * D * (oneplusd + delta / sigmasq).log()
    term2 = 0.5 * D * (oneplusd + epsilon / sigmasq).log()
    term3 = (1.0 - 0.5 * (alpha * alpha / (oneplusd + delta / sigmasq) + 1.0 / (oneplusd + epsilon / sigmasq))).log()
    return 0.5 * (term1 + term2 + term3)


# find the exact EIG (up to a constant) for a design B
def analytic_eig_budget(alpha, D, B, sigmasq):
    oneplusd = 1.0 + 1.0 / D
    term1 = (oneplusd + B / sigmasq).log().sum(-1)
    term2a = (alpha * alpha / (oneplusd + B[..., 0:int(D/2)] / sigmasq)).sum(-1)
    term2b = (1.0 / (oneplusd + B[..., int(D/2):] / sigmasq)).sum(-1)
    term2 = (1.0 - (term2a + term2b) / D).log()
    eig = 0.5 * (term1 + term2)
    if eig.numel() == 1:
        return eig.item()
    return eig


# find the optimal EIG (up to a constant) for D dimensions. we reduce the optimization
# to a univariate problem and use brute-force bisectioning to identify the optimum.
def optimal_eig(alpha, D, sigmasq, S=100 * 1000):
    delta = torch.arange(S + 1).float() / float(S)
    epsilon = 1.0 - delta
    max_eig, optimal_delta = analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq).max(dim=-1)

    for zoom in [1.0e-2, 1.0e-4, 1.0e-6]:
        delta = delta[optimal_delta].item() - zoom + 2.0 * zoom * torch.arange(S + 1).float() / float(S)
        epsilon = 1.0 - delta
        max_eig, optimal_delta = analytic_eig_delta_epsilon(alpha, D, delta, epsilon, sigmasq).max(dim=-1)

    return max_eig.item(), delta[optimal_delta].item(), epsilon[optimal_delta].item()


# get the prior precision matrix
def get_prior_precision(alpha, D):
    u = torch.ones(D).unsqueeze(-1)
    u[0:int(D/2), 0] = alpha
    prior_precision = (1.0 + 1.0 / D) * torch.eye(D) - (1.0 / D) * torch.mm(u, u.t())
    return prior_precision


# get the cholesky matrix of the prior precision matrix
def get_prior_scale_tril(alpha, D):
    precision = get_prior_precision(alpha, D)
    return precision.inverse().cholesky()


# the total budget in D dimensions
def total_budget(D):
    return 0.5 * D


# produced a normalized design (i.e budget) from an unconstrained design
def normalized_design(design, D):
    return total_budget(D) * softmax(design, dim=-1)


# specify the model for given D/alpha/sigma. if learn_design==True,
# the design is a pyro parameter that gets optimized; otherwise it's a fixed constant.
def make_model(D=4, alpha=0.5, sigma=1.0, learn_design=True):
    def init_budget_fn():
        b = torch.rand(D)
        return b / b.sum()

    def model_learn_design(design_prototype):
        B = total_budget(D) * pyro.param('budget', init_budget_fn, constraint=constraints.simplex)
        batch_shape = design_prototype.shape[:-2]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            x = pyro.sample("x", dist.MultivariateNormal(torch.zeros(D), precision_matrix=get_prior_precision(alpha, D)))
            return pyro.sample("y", dist.Normal(x * B, sigma * B.sqrt()).to_event(1))

    def model_fixed_design(design):
        batch_shape = design.shape[:-1]
        with ExitStack() as stack:
            for plate in iter_plates_to_shape(batch_shape):
                stack.enter_context(plate)

            x = pyro.sample("x", dist.MultivariateNormal(torch.zeros(batch_shape + (D,)),
                                                         precision_matrix=get_prior_precision(alpha, D)))
            return pyro.sample("y", dist.Normal(x * design, sigma * design.sqrt()).to_event(1))

    if learn_design:
        return model_learn_design
    else:
        return model_fixed_design


# make a multivariate normal variational distribution parameterized by a cholesky factor
def make_posterior_guide(d, alpha, D):
    def posterior_guide(y_dict, design, observation_labels, target_labels, **kwargs):
        y = torch.cat(list(y_dict.values()), dim=-1)
        A = pyro.param("A", lambda: torch.zeros(d, 1, D))
        scale_tril = pyro.param("scale_tril", lambda: lexpand(get_prior_scale_tril(alpha, D), d),
                                constraint=torch.distributions.constraints.lower_cholesky)
        pyro.sample("x", dist.MultivariateNormal(A * y, scale_tril=scale_tril))
    return posterior_guide


# used to negate loss functions
def neg_loss(loss):
    def new_loss(*args, **kwargs):
        return (-a for a in loss(*args, **kwargs))
    return new_loss


def double_reparam_ace_loss(N, M, D=4, alpha=0.5, sigma=1.0):
    def init_budget_fn():
        b = torch.rand(D)
        return b / b.sum()

    B = total_budget(D) * pyro.param('budget', init_budget_fn(), constraint=constraints.simplex)
    # Draw N samples
    prior_precision = get_prior_precision(alpha, D)
    with pyro.plate("plate", N):
        theta0 = pyro.sample("x", dist.MultivariateNormal(torch.zeros(D), precision_matrix=prior_precision))
        y = pyro.sample("y", dist.Normal(theta0 * B, sigma * B.sqrt()).to_event(1))
    # Compute log prob, detaching B
    lp_theta_0 = dist.MultivariateNormal(torch.zeros(D), precision_matrix=prior_precision).log_prob(theta0)
    lp_y_theta_0 = dist.Normal(theta0 * B, sigma * B.sqrt()).to_event(1).log_prob(y)
    # Compute q_phi(theta_0 | y), detaching phi and y separately
    A = pyro.param("A", lambda: torch.zeros(1, D))
    scale_tril = pyro.param("scale_tril", lambda: lexpand(get_prior_scale_tril(alpha, D), N),
                            constraint=torch.distributions.constraints.lower_cholesky)
    lq_theta0_y_detach_phi = dist.MultivariateNormal(A.detach() * y, scale_tril=scale_tril.detach()).log_prob(theta0)
    lq_theta0_y_detach_y = dist.MultivariateNormal(A * y.detach(), scale_tril=scale_tril).log_prob(theta0)

    # Sample contrastive samples
    with pyro.plate("plate1", N):
        with pyro.plate("plate2", M):
            thetal_detach_phi = pyro.sample("xldphi", dist.MultivariateNormal(A.detach() * y, scale_tril=scale_tril.detach()))
            thetal_detach_y = pyro.sample("xldy", dist.MultivariateNormal(A * y.detach(), scale_tril=scale_tril))

    lp_theta_l_detach_phi = dist.MultivariateNormal(torch.zeros(D), precision_matrix=prior_precision).log_prob(thetal_detach_phi)
    lp_theta_l_detach_y = dist.MultivariateNormal(torch.zeros(D), precision_matrix=prior_precision).log_prob(thetal_detach_y)

    lp_y_theta_l_detach_phi = dist.Normal(thetal_detach_phi * B, sigma * B.sqrt()).to_event(1).log_prob(y)
    lp_y_theta_l_detach_y = dist.Normal(thetal_detach_y * B.detach(), sigma * B.detach().sqrt()).to_event(1).log_prob(y.detach())

    lq_theta_l_y_detach_phi = dist.MultivariateNormal(A.detach() * y, scale_tril=scale_tril.detach()).log_prob(thetal_detach_phi)
    lq_theta_l_y_detach_y = dist.MultivariateNormal(A * y.detach(), scale_tril=scale_tril).log_prob(thetal_detach_y)

    y_loss = -torch.cat([lexpand(lp_theta_0 + lp_y_theta_0 - lq_theta0_y_detach_phi, 1),
                         lp_theta_l_detach_phi + lp_y_theta_l_detach_phi - lq_theta_l_y_detach_phi], dim=0).logsumexp(0) \
             + math.log(M+1) \
             + lp_y_theta_0
    y_loss = y_loss.mean(0)

    log_wl = lp_theta_l_detach_y + lp_y_theta_l_detach_y - lq_theta_l_y_detach_y
    log_w0 = lp_theta_0 + lp_y_theta_0 - lq_theta0_y_detach_y
    log_wsum = torch.cat([lexpand(log_w0, 1), log_wl], dim=0).logsumexp(0)
    phi_loss = (log_w0 - log_wsum).exp().detach() * lq_theta0_y_detach_y - (log_wl - log_wsum).exp().pow(2).detach() * log_wl

    surrogate_loss = (y_loss + phi_loss).mean(0).sum()
    eig_estimate = y_loss

    return surrogate_loss, eig_estimate


# helper function for doing gradient-based minimization of a loss function
def opt_eig_loss_w_history(design, loss_fn, num_samples, num_steps, optim):
    params = None
    xi_history = []

    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            agg_loss, loss = loss_fn(design, num_samples)
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(agg_loss):
            raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
        agg_loss.backward()
        xi_history.append(pyro.param('budget').detach().clone())
        optim(params)
        optim.step()

    xi_history.append(pyro.param('budget').detach().clone())

    return torch.stack(xi_history)


# main experimental loop
def main(num_trials, estimator, seed, verbose, hyper):
    assert estimator in ['nce', 'ace', 'ace.dreg', 'posterior', 'exact.bo']

    alpha = 0.1  # controls the prior precision matrix
    sigma = 1.0  # observation likelihood noise parameter
    sigmasq = sigma ** 2
    start_lr = 0.1  # learning rate hyperparameters for gradient methods

    # BO hyperparameters
    num_acquisition, num_parallel = 5, 3
    lengthscale, jitter = torch.tensor(0.2), 1.0e-6

    Ds = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    if estimator=='ace':
        num_steps = 10000  # gradient steps
        end_lr = 0.0001
    elif estimator=='ace.dreg':
        num_steps = 9000  # gradient steps
        end_lr = 0.0001
    elif estimator=='nce':
        num_steps = 20000  # gradient steps
        end_lr = 0.00001
    elif estimator=='posterior':
        num_steps = 18000  # gradient steps
        end_lr = 0.0003
    elif estimator=='exact.bo':
        num_steps = 110  # BO steps

    results = {'estimator': estimator, 'git-hash': get_git_revision_hash(), 'seed': seed,
               'alpha': alpha, 'sigma': sigma, 'num_trials': num_trials, 'num_steps': num_steps,
               'dimensions': Ds,
               'elapsed_times': {},
               'eig_errors': {},
               'design_errors': {}}

    for D in Ds:
        results['elapsed_times'][D] = []
        results['eig_errors'][D] = []
        results['design_errors'][D] = []

        for trial in range(num_trials):
            pyro.set_rng_seed(seed + 1000 * D + trial)
            t0 = time.time()
            pyro.clear_param_store()

            learn_design = False if 'bo' in estimator else True
            model = make_model(D=D, alpha=alpha, sigma=sigma, learn_design=learn_design)

            if estimator == 'posterior':
                guide = make_posterior_guide(1, alpha, D)
                loss = _differentiable_posterior_loss(model, guide, "y", "x")

            elif estimator == 'pce':
                eig_loss = lambda d, N, **kwargs: pce_eig(model=model, design=d, observation_labels="y",
                                                          target_labels="x", N=N, M=10, **kwargs)
                loss = neg_loss(eig_loss)

            elif estimator == 'ace':
                guide = make_posterior_guide(1, alpha, D)
                loss = neg_loss(_ace_eig_loss(model, guide, 10, "y", "x"))

            elif estimator == 'ace.dreg':
                loss = neg_loss(lambda d, N: double_reparam_ace_loss(N, 10, D, alpha, sigma))

            elif estimator == 'exact.bo':
                eig = lambda d: analytic_eig_budget(alpha, D, normalized_design(d, D), sigmasq)[0, ...]

                # sample initial designs
                design = 0.3 * torch.randn(1, num_acquisition, D).double()
                y = eig(design).detach().clone()

                kernel = gp.kernels.Matern52(input_dim=D, lengthscale=lengthscale.double(), variance=y.var(unbiased=True))

                # BO loop
                for i in range(num_steps):
                    Kff = kernel(design)
                    Kff += jitter * torch.eye(Kff.shape[-1]).type_as(Kff)
                    Lff = Kff.cholesky(upper=False)
                    new_design = 0.3 * torch.randn(num_parallel, num_acquisition, D).double()
                    new_design.requires_grad_(True)
                    minimizer = torch.optim.LBFGS([new_design], max_eval=20)

                    # define ucb acquisition function
                    def gp_ucb1():
                        minimizer.zero_grad()
                        KXXnew = kernel(design, new_design)
                        LiK = torch.triangular_solve(KXXnew, Lff, upper=False)[0]
                        Liy = torch.triangular_solve(y.unsqueeze(-1), Lff, upper=False)[0]
                        mean = rmv(LiK.transpose(-1, -2), Liy.squeeze(-1))
                        KXnewXnew = kernel(new_design)
                        var = (KXnewXnew - LiK.transpose(-1, -2).matmul(LiK)).diagonal(dim1=-2, dim2=-1)
                        ucb = -(mean + 2*var.clamp(min=0.0).sqrt())
                        loss = ucb.sum()
                        torch.autograd.backward(new_design,
                                                torch.autograd.grad(loss, new_design, retain_graph=True))
                        return loss

                    minimizer.step(gp_ucb1)  # do minimization with LBFGS
                    new_design = new_design.reshape(-1, D).unsqueeze(0)
                    new_y = eig(new_design).detach().clone()
                    design = torch.cat([design, new_design], dim=1)
                    y = torch.cat([y, new_y])

                # choose the best design seen so far
                _, final_design = torch.max(y, dim=-1)
                final_design = normalized_design(design, D)[0, final_design, :].float()

            else:
                raise ValueError("Unexpected estimator")

            if 'bo' not in estimator:  # do gradient-based optimization for nce/ace/posterior methods
                gamma = (end_lr / start_lr) ** (1 / num_steps)
                scheduler = pyro.optim.ExponentialLR({'optimizer': torch.optim.Adam, 'optim_args': {'lr': start_lr}, 'gamma': gamma})

                design_prototype = torch.zeros(1, 1, D)
                num_samples = 100 if estimator=='posterior' else 10
                design_history = opt_eig_loss_w_history(design_prototype, loss, num_samples=num_samples,
                                                        num_steps=num_steps, optim=scheduler)
                final_design = total_budget(D) * design_history[-1]

            tf = time.time()
            if verbose:
                print("Elapsed time", tf - t0)
            results['elapsed_times'][D].append(tf - t0)

            # compute optimal EIG and optimal design
            opt_eig, opt_delta, opt_epsilon = optimal_eig(alpha, D, sigmasq)
            opt_design = opt_delta * torch.ones(D)
            opt_design[int(D/2):] = opt_epsilon

            # normalize EIG w.r.t. the EIG for a flat budget
            flat_eig = analytic_eig_budget(alpha, D, total_budget(D) * torch.ones(D) / D, sigmasq)
            final_eig = analytic_eig_budget(alpha, D, final_design, sigmasq)
            eig_normalizer = opt_eig - flat_eig

            eig_error = math.fabs(final_eig - opt_eig) / eig_normalizer
            design_error = (final_design - opt_design).abs().sum().item()
            results['eig_errors'][D].append(eig_error)
            results['design_errors'][D].append(design_error)

            if verbose:
                print("*** DIMENSION %d ***" % D)
                print("Flat/Final/Optimal EIG:  %.5f /  %.5f / %.5f" % (flat_eig, final_eig, opt_eig))
                print("Final design:\n", final_design.data.numpy())
                print("Optimal design:\n", opt_design.data.numpy())
                print("Mean Absolute EIG Error: %.5f" % math.fabs(final_eig - opt_eig))
                print("Normalized Mean Absolute EIG Error: %.5f" % eig_error)
                print("Mean Absolute Design Error: %.5f" % design_error)

    for D in Ds:
        summary = "[Dimension %d]  Mean Errors: %.4f  %.4f   Median Errors: %.4f %.4f  Time: %.4f +- %.4f"
        print(summary % (D, np.mean(results['eig_errors'][D]), np.mean(results['design_errors'][D]),
              np.median(results['eig_errors'][D]), np.median(results['design_errors'][D]),
              np.mean(results['elapsed_times'][D]), np.std(results['elapsed_times'][D])))

    with open(estimator + '.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)

    print(results)
    #print('hyper', hyper)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradient-based design optimization (one shot) with a conjugate gaussian model")
    parser.add_argument("--num-trials", default=10, type=int)
    parser.add_argument("--estimator", default="ace", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--hyper", default=0, type=float)
    parser.add_argument("--verbose", default=0, type=int)
    args = parser.parse_args()
    main(args.num_trials, args.estimator, args.seed, args.verbose, args.hyper)
