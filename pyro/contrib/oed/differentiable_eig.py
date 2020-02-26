import torch
import math
import logging

import pyro
from pyro import poutine
from pyro.util import is_bad
from pyro.contrib.util import lexpand
from pyro.contrib.oed.eig import _safe_mean_terms


def _differentiable_posterior_loss(model, guide, observation_labels, target_labels):
    """This version of the loss function deals with the case that `y` is not reparametrizable."""

    def loss_fn(design, num_particles, control_variate=0., **kwargs):

        expanded_design = lexpand(design, num_particles)

        # Sample from p(y, theta | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        # Run through q(theta | y, d)
        conditional_guide = pyro.condition(guide, data=theta_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(
            y_dict, expanded_design, observation_labels, target_labels)
        cond_trace.compute_log_prob()

        terms = -sum(cond_trace.nodes[l]["log_prob"] for l in target_labels)
        ape_estimate = _safe_mean_terms(terms)[1]

        # Calculate the score parts
        trace.compute_score_parts()
        prescore_function = sum(trace.nodes[l]["score_parts"][1] for l in observation_labels)
        terms += (terms.detach() - control_variate) * prescore_function

        result = _safe_mean_terms(terms)[0]
        return result, ape_estimate

    return loss_fn


def differentiable_pce_eig(model, design, observation_labels, target_labels=None, N=100, M=10, control_variate=0.,
                           **kwargs):

    # Take N samples of the model
    expanded_design = lexpand(design, N)  # N copies of the model
    trace = poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    conditional_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels)

    y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    # Using (M, 1) instead of (M, N) - acceptable to re-use thetas between ys because
    # theta comes before y in graphical model
    reexpanded_design = lexpand(design, M, N)  # sample M theta
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    retrace.compute_log_prob()
    marginal_log_probs = torch.cat([lexpand(conditional_lp, 1),
                                    sum(retrace.nodes[l]["log_prob"] for l in observation_labels)], dim=0)
    marginal_lp = marginal_log_probs.logsumexp(0) - math.log(M+1)

    terms = conditional_lp - marginal_lp
    nce_part = _safe_mean_terms(terms)[1]

    # Calculate the score parts
    trace.compute_score_parts()
    prescore_function = sum(trace.nodes[l]["score_parts"][1] for l in observation_labels)
    grad_terms = (terms.detach() - control_variate) * prescore_function + terms

    surrogate_loss = _safe_mean_terms(grad_terms)[0]
    return (surrogate_loss, nce_part)


# This method is WIP
def differentiable_nce_proposal_eig(model, design, observation_labels, target_labels, 
                                    proposal, N=100, M=10, control_variate=0., **kwargs):

    # Take N samples of the model
    expanded_design = lexpand(design, N)  # N copies of the model
    trace = poutine.trace(proposal).get_trace(expanded_design)
    trace.compute_log_prob()
    proposal_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels)
    y_dict = {l: lexpand(trace.nodes[l]["value"], M+1) for l in observation_labels}
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    reexpanded_design = lexpand(design, M+1, N)  # sample M theta
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
    retrace.compute_log_prob()
    marginal_log_probs = torch.cat([sum(retrace.nodes[l]["log_prob"] for l in observation_labels)], dim=0)
    print(y_dict['y'][0,0,...])
    marginal_lp = marginal_log_probs.logsumexp(0) - math.log(M+1)
    print('marginal', marginal_lp[0,...])
    conditional_lp = marginal_log_probs[0, ...]
    print('cond', conditional_lp[0, ...])
    importance_weights = (conditional_lp - proposal_lp).exp()
    print('importance weight', importance_weights[0,...])
    print('mean weights', importance_weights.mean(0))
    # marginal_log_probs = sum(retrace.nodes[l]["log_prob"] for l in observation_labels)
    # marginal_lp = marginal_log_probs.logsumexp(0) - math.log(M)

    terms = conditional_lp - marginal_lp
    nce_part =  _safe_mean_terms(importance_weights * terms)[1]
    print('nce', nce_part)

    # Calculate the score parts
    grad_terms = (terms.detach() - control_variate) * importance_weights

    surrogate_loss = _safe_mean_terms(grad_terms)[0]
    return (surrogate_loss, nce_part)


def _differentiable_ace_eig_loss(model, guide, M, observation_labels, target_labels):

    def loss_fn(design, num_particles, control_variate=0., **kwargs):
        N = num_particles
        expanded_design = lexpand(design, N)

        # Sample from p(y, theta | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict_exp = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        trace.compute_log_prob()
        logw0 = sum(trace.nodes[l]["log_prob"] for l in target_labels)
        logw0 += sum(trace.nodes[l]["log_prob"] for l in observation_labels)
        
        reguide_trace = poutine.trace(pyro.condition(guide, data=theta_dict)).get_trace(
            y_dict, expanded_design, observation_labels, target_labels
        )
        reguide_trace.compute_log_prob()
        q_theta0_terms = sum(reguide_trace.nodes[l]["log_prob"] for l in target_labels)
        logw0 -= q_theta0_terms

        # Sample M times from q(theta | y, d) for each y
        reexpanded_design = lexpand(expanded_design, M)
        guide_trace = poutine.trace(guide).get_trace(
            y_dict, reexpanded_design, observation_labels, target_labels,
        )
        theta_y_dict = {l: guide_trace.nodes[l]["value"] for l in target_labels}
        theta_y_dict.update(y_dict_exp)
        guide_trace.compute_log_prob()

        # Re-run that through the model to compute the joint
        model_trace = poutine.trace(pyro.condition(model, data=theta_y_dict)).get_trace(reexpanded_design)
        model_trace.compute_log_prob()

        detached_theta_y_dict = {l: guide_trace.nodes[l]["value"].detach() for l in target_labels}
        detached_theta_y_dict.update(y_dict_exp)
        detached_model_trace = poutine.trace(pyro.condition(model, data=detached_theta_y_dict)
                                             ).get_trace(reexpanded_design)
        detached_model_trace.compute_log_prob()

        logwm = -sum(guide_trace.nodes[l]["log_prob"] for l in target_labels)
        logwm += sum(model_trace.nodes[l]["log_prob"] for l in target_labels)
        logwm_detached = logwm.detach().clone()
        logwm += sum(model_trace.nodes[l]["log_prob"] for l in observation_labels)
        logwm_detached += sum(detached_model_trace.nodes[l]["log_prob"] for l in observation_labels)

        logsumw_detached = torch.cat([lexpand(logw0, 1), logwm_detached]).logsumexp(0)
        terms = -logsumw_detached + math.log(M + 1)

        terms += sum(trace.nodes[l]["log_prob"] for l in observation_labels)
        eig_estimate = _safe_mean_terms(terms)[1]

        # Calculate the score parts
        trace.compute_score_parts()
        prescore_function = sum(trace.nodes[l]["score_parts"][1] for l in observation_labels)

        xi_grad_terms = (terms - control_variate).detach() * prescore_function + terms
        phi_grad_terms = (logw0 - logsumw_detached).detach().exp() * q_theta0_terms - \
                         ((logwm - logsumw_detached).detach().exp().pow(2) * logwm).sum(0)
        surrogate_loss = _safe_mean_terms(xi_grad_terms + phi_grad_terms)[0]

        return surrogate_loss, eig_estimate

    return loss_fn


# This is a WIP
def _saddle_marginal_loss(model, guide, observation_labels, target_labels):
    """Marginal loss: to evaluate directly use `marginal_eig` setting `num_steps=0`."""

    def loss_fn(design, num_particles, control_variate=0., **kwargs):
        expanded_design = lexpand(design, num_particles)

        # Sample from p(y | d)
        trace = poutine.trace(model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels}

        # Run through q(y | d)
        conditional_guide = pyro.condition(guide, data=y_dict)
        cond_trace = poutine.trace(conditional_guide).get_trace(
            expanded_design, observation_labels, target_labels)
        cond_trace.compute_log_prob()

        terms = -sum(cond_trace.nodes[l]["log_prob"] for l in observation_labels)

        trace.compute_log_prob()
        terms += sum(trace.nodes[l]["log_prob"] for l in observation_labels).detach()
        q_loss, eig_estimate = _safe_mean_terms(terms)

        # Calculate the score parts
        trace.compute_score_parts()
        prescore_function = sum(trace.nodes[l]["score_parts"][1] for l in observation_labels)
        grad_terms = (terms.detach() - control_variate) * prescore_function

        d_loss = _safe_mean_terms(grad_terms)[0]
        return d_loss, q_loss, eig_estimate

    return loss_fn


def marginal_gradient_eig(model, design, observation_labels, target_labels,
                          num_samples, num_steps, guide, optim,
                          final_design=None, final_num_samples=None, burn_in_steps=0):

    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]
    loss_fn = _saddle_marginal_loss(model, guide, observation_labels, target_labels)

    if final_design is None:
        final_design = design
    if final_num_samples is None:
        final_num_samples = num_samples

    params = None
    for step in range(num_steps):
        if params is not None:
            pyro.infer.util.zero_grads(params)
        with poutine.trace(param_only=True) as param_capture:
            d_loss, q_loss, eig_estimate = loss_fn(design, num_samples)
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        if torch.isnan(d_loss) or torch.isnan(q_loss):
            raise ArithmeticError("Encountered NaN loss in marginal_gradient_eig")
        q_loss.backward(retain_graph=True)
        optim(params)
        if step > burn_in_steps:
            (-d_loss).backward(retain_graph=True)
            optim(params)
        try:
            optim.step()
        except AttributeError:
            pass
        logging.debug("{} {} {}".format(step, pyro.param("xi"), pyro.param("xi").shape))

    _, _, eig_estimates = loss_fn(final_design, final_num_samples, evaluation=True)
    return eig_estimates
