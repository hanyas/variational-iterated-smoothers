from typing import Callable
from functools import partial

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import (
    Gaussian,
    AffineGaussian,
    GaussMarkov,
    LogPrior,
    LogTransition,
    LogObservation,
    Potential,
    LogMarginalNorm,
    LogConditionalNorm
)
from varsmooth.smoothers.utils import (
    statistical_expansion,
    merge_messages,
    log_to_std_form,
    std_to_log_form,
    line_search
)
from varsmooth.utils import (
    none_or_concat,
    none_or_shift,
    none_or_idx,
    symmetrize,
    eig
)

from varsmooth.smoothers.forward_markov import backward_log_message
from varsmooth.smoothers.forward_markov import forward_std_message
from varsmooth.smoothers.reverse_markov import forward_log_message
from varsmooth.smoothers.reverse_markov import backward_std_message

from varsmooth.smoothers.reverse_markov import dual_objective
from varsmooth.smoothers.reverse_markov import kl_between_reverse_gauss_markovs
from varsmooth.smoothers.forward_markov import kl_between_forward_gauss_markovs

_logdet = lambda x: jnp.linalg.slogdet(x)[1]


def log_two_filter_smoother(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    forward_reference: GaussMarkov,
    reverse_reference: GaussMarkov,
    temperature: float
) -> Gaussian:

    marginals = forward_std_message(forward_reference)

    log_prior, log_transition, log_observation = \
        statistical_expansion(
            observations,
            log_prior_fn,
            log_transition_fn,
            log_observation_fn,
            forward_reference.kernels,
            marginals,
        )

    damping = temperature / (1.0 + temperature)
    forward_posterior, _, _, backward_message, _ = backward_log_message(
        log_prior,
        log_transition,
        log_observation,
        forward_reference,
        damping,
    )

    reverse_posterior, _, forward_message, _, _ = forward_log_message(
        log_prior,
        log_transition,
        log_observation,
        reverse_reference,
        damping,
    )

    fwd_kl_div = kl_between_forward_gauss_markovs(
        forward_std_message(forward_posterior),
        forward_posterior,
        forward_reference
    )

    rvs_kl_div = kl_between_reverse_gauss_markovs(
        backward_std_message(reverse_posterior),
        reverse_posterior,
        reverse_reference
    )

    marginals = update_marginals(
        marginals,
        forward_message,
        backward_message,
        reverse_posterior.marginal,
        damping
    )
    return marginals


def update_marginals(
    marginals: Gaussian,
    forward_message: Potential,
    backward_message: LogConditionalNorm,
    boundary: Gaussian,
    damping: float
):
    log_marginals = jax.vmap(std_to_log_form)(marginals)
    log_messages = jax.vmap(merge_messages)(
        none_or_shift(forward_message, -1),
        backward_message,
    )
    log_boundary = std_to_log_form(boundary)

    # update all but last marginals
    potentials = Potential(
        R=(1.0 - damping) * log_messages.R + damping * log_marginals.R[:-1],
        r=(1.0 - damping) * log_messages.r + damping * log_marginals.r[:-1],
        rho=(1.0 - damping) * log_messages.rho + damping * log_marginals.rho[:-1]
    )

    # update last marginal
    potentials = none_or_concat(
        potentials,
        Potential(
            R=(1.0 - damping) * log_boundary.R + damping * log_marginals.R[-1],
            r=(1.0 - damping) * log_boundary.r + damping * log_marginals.r[-1],
            rho=(1.0 - damping) * log_boundary.rho + damping * log_marginals.rho[-1]
        ),
        position=-1
    )
    return jax.vmap(log_to_std_form)(potentials)


def iterated_log_two_filter_smoother(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    init_forward_posterior: GaussMarkov,
    init_reverse_posterior: GaussMarkov,
    kl_constraint: float,
    init_temperature: float,
    max_iter: int
):

    def _iteration(carry, args):
        reference_marginals, forward_reference, reverse_reference = carry
        i = args

        log_prior, log_transition, log_observation = \
            statistical_expansion(
                observations,
                log_prior_fn,
                log_transition_fn,
                log_observation_fn,
                forward_reference.kernels,
                reference_marginals,
            )

        def dual_fn(_temperature):
            _damping = _temperature / (1.0 + _temperature)
            return dual_objective(
                log_prior,
                log_transition,
                log_observation,
                reverse_reference,
                kl_constraint,
                _damping,
            )

        def dual_gd(_temperature):
            _damping = _temperature / (1.0 + _temperature)
            _posterior, _, _, _, _feasible_pass = forward_log_message(
                log_prior,
                log_transition,
                log_observation,
                reverse_reference,
                _damping,
            )

            def _feasible_constraint():
                _kl_div = kl_between_reverse_gauss_markovs(
                    marginals=backward_std_message(_posterior),
                    gauss_markov=_posterior,
                    ref_gauss_markov=reverse_reference
                )
                return kl_constraint - _kl_div

            def _not_feasible_constraint():
                return jnp.inf

            return jax.lax.cond(
                pred=jnp.all(_feasible_pass),
                true_fun=_feasible_constraint,
                false_fun=_not_feasible_constraint
            )

        temperature, dual_val, _, feasible = line_search(
            init_temperature, dual_fn, dual_gd, rtol=0.1 * kl_constraint
        )

        def _feasible_line_search():
            _damping = temperature / (1.0 + temperature)

            _forward_posterior, _, _, _backward_message, _ = backward_log_message(
                log_prior,
                log_transition,
                log_observation,
                forward_reference,
                _damping,
            )

            _reverse_posterior, _, _forward_message, _, _ = forward_log_message(
                log_prior,
                log_transition,
                log_observation,
                reverse_reference,
                _damping,
            )

            _fwd_kl_div = kl_between_forward_gauss_markovs(
                forward_std_message(_forward_posterior),
                _forward_posterior,
                forward_reference
            )

            _rvs_kl_div = kl_between_reverse_gauss_markovs(
                backward_std_message(_reverse_posterior),
                _reverse_posterior,
                reverse_reference
            )

            _marginals = update_marginals(
                reference_marginals,
                _forward_message,
                _backward_message,
                _reverse_posterior.marginal,
                _damping
            )

            jax.debug.print(
                "iter: {a}, damping: {b}, fwd_kl_div: {c}, rvs_kl_div: {d}, dual: {e}",
                a=i, b=_damping, c=_fwd_kl_div, d=_rvs_kl_div, e=dual_val
            )
            return _marginals, _forward_posterior, _reverse_posterior

        def _not_feasible_line_search():
            jax.debug.print(
                "iter: {a} not feasible, process might have converged", a=i
            )
            return reference_marginals, forward_reference, reverse_reference

        optimal_marginals, optimal_forward, optimal_reverse = jax.lax.cond(
            pred=feasible,
            true_fun=_feasible_line_search,
            false_fun=_not_feasible_line_search
        )
        return (optimal_marginals, optimal_forward, optimal_reverse), optimal_marginals

    init_marginals = forward_std_message(init_forward_posterior)
    (optimal_marginals, _, _), _ = jax.lax.scan(
        f=_iteration,
        init=(init_marginals, init_forward_posterior, init_reverse_posterior),
        xs=jnp.arange(max_iter)
    )
    return optimal_marginals
