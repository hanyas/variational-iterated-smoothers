from typing import Callable, Tuple
from functools import partial

import jax
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import (
    Gaussian,
    AffineGaussian,
    GaussMarkov,
    LogPrior,
    LogTransition,
    LogObservation,
    LogMarginalNorm,
    LogMessage,
    ValueFn,
)
from varsmooth.smoothers.utils import statistical_expansion, line_search
from varsmooth.smoothers.utils import kl_between_forward_gauss_markovs

from varsmooth.utils import (
    none_or_concat,
    none_or_shift,
    none_or_idx,
    symmetrize,
    logdet,
    bounded_while_loop,
)


# @jax.jit
def log_backward_message(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    nominal_posterior: GaussMarkov,
    damping: float,
) -> Tuple[GaussMarkov, LogMarginalNorm, ValueFn, LogMessage, bool]:

    def _backward_step(carry, args):
        R, r, rho = carry
        C11, C12, C21, C22, c1, c2, kappa, \
            L, l, nu, \
            F, d, Sigma = args

        G11 = (1.0 - damping) * (C11 + R) + damping * jsc.linalg.inv(Sigma)
        G22 = (1.0 - damping) * C22 + damping * F.T @ jsc.linalg.solve(Sigma, F)
        G12 = (1.0 - damping) * C12 + damping * jsc.linalg.solve(Sigma, F)
        g1 = (1.0 - damping) * (c1 + r) + damping * jsc.linalg.solve(Sigma, d)
        g2 = (1.0 - damping) * c2 - damping * F.T @ jsc.linalg.solve(Sigma, d)
        theta = (
            (1.0 - damping) * (kappa + rho)
            - 0.5 * damping * logdet(2 * jnp.pi * Sigma)
            - 0.5 * damping * d.T @ jsc.linalg.solve(Sigma, d)
        )

        G11 = symmetrize(G11)
        G22 = symmetrize(G22)

        def _feasible_backward_pass():
            S = G22 - G12.T @ jsc.linalg.solve(G11, G12)
            s = g2 + G12.T @ jsc.linalg.solve(G11, g1)
            xi = (
                theta
                + 0.5 * logdet(2 * jnp.pi * jsc.linalg.inv(G11))
                + 0.5 * g1.T @ jsc.linalg.solve(G11, g1)
            )

            F = jsc.linalg.solve(G11, G12)
            d = jsc.linalg.solve(G11, g1)
            Sigma = jsc.linalg.inv(G11)

            R = L + 1.0 / (1.0 - damping) * S
            r = l + 1.0 / (1.0 - damping) * s
            rho = nu + 1.0 / (1.0 - damping) * xi

            value_fn = ValueFn(R, r, rho)
            return value_fn, (
                value_fn,
                AffineGaussian(F, d, Sigma),
                LogMessage(S, s, xi),
                True  # feasible
            )

        def _not_feasible_backward_pass():
            S = jnp.zeros_like(G22)
            s = jnp.zeros_like(g2)
            xi = jnp.zeros_like(theta)

            R = jnp.zeros_like(L)
            r = jnp.zeros_like(l)
            rho = jnp.zeros_like(nu)

            value_fn = ValueFn(R, r, rho)
            return value_fn, (
                value_fn,
                AffineGaussian(F, d, Sigma),
                LogMessage(S, s, xi),
                False   # Not feasible
            )

        return jax.lax.cond(
            pred=jnp.all(jnp.linalg.eigvalsh(G11) > 1e-8),
            true_fun=_feasible_backward_pass,
            false_fun=_not_feasible_backward_pass,
        )

    last_log_obs = none_or_idx(log_observation, -1)
    last_value_fn = ValueFn(
        R=last_log_obs.L,
        r=last_log_obs.l,
        rho=last_log_obs.nu
    )

    log_aux_obs = none_or_concat(
        none_or_shift(log_observation, -1),
        LogObservation(log_prior.L, log_prior.l, log_prior.nu),
        1
    )

    nominal_marginal, nominal_kernels = nominal_posterior

    first_value_fn, (value_fns, kernels, log_bwd_msgs, feasible_pass) = jax.lax.scan(
        f=_backward_step,
        init=last_value_fn,
        xs=(*log_transition, *log_aux_obs, *nominal_kernels),
        reverse=True,
    )
    value_fns = none_or_concat(value_fns, last_value_fn, -1)

    R, r, rho = first_value_fn

    m, P = nominal_marginal
    inv_P = jsc.linalg.inv(P)

    J11 = (1.0 - damping) * R + damping * inv_P
    J12 = damping * inv_P
    J22 = damping * inv_P
    j1 = (1.0 - damping) * r
    j2 = jnp.zeros_like(j1)
    tau = (
        (1.0 - damping) * rho
        - 0.5 * damping * logdet(2 * jnp.pi * P)
    )

    J11 = symmetrize(J11)
    J22 = symmetrize(J22)

    def _feasible_marginal():
        # init marginal
        _m = jsc.linalg.solve(J11, j1 + J12 @ m)
        _P = jsc.linalg.inv(J11)

        # log normalizer
        U = J22 - J12.T @ jsc.linalg.solve(J11, J12)
        u = j2 - J12.T @ jsc.linalg.solve(J11, j1)
        eta = (
            tau
            + 0.5 * logdet(2 * jnp.pi * jsc.linalg.inv(J11))
            + 0.5 * j1.T @ jsc.linalg.solve(J11, j1)
        )
        return Gaussian(_m, _P), LogMarginalNorm(U, u, eta)

    def _not_feasible_marginal():
        _m = jnp.zeros_like(nominal_marginal.mean)
        _P = jnp.zeros_like(nominal_marginal.cov)
        marginal = Gaussian(_m, _P)

        U = jnp.zeros_like(J22)
        u = jnp.zeros_like(j2)
        eta = jnp.zeros_like(tau)
        return Gaussian(_m, _P), LogMarginalNorm(U, u, eta)

    marginal, log_marg_norm = jax.lax.cond(
        pred=jnp.all(feasible_pass),
        true_fun=_feasible_marginal,
        false_fun=_not_feasible_marginal,
    )
    return (
        GaussMarkov(marginal, kernels),
        log_marg_norm,
        value_fns,
        log_bwd_msgs,
        feasible_pass
    )


# @jax.jit
def std_forward_message(posterior: GaussMarkov) -> Gaussian:
    init_marginal, kernels = posterior

    def _forward_step(carry, args):
        q = carry
        kernel = args

        m, P = q
        F, d, Sigma = kernel

        qn = Gaussian(
            mean=F @ m + d,
            cov=F @ P @ F.T + Sigma
        )
        return qn, qn

    _, marginals = jax.lax.scan(_forward_step, init_marginal, kernels)
    return none_or_concat(marginals, init_marginal, position=1)


def forward_markov_smoother(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    reference_posterior: GaussMarkov,
    temperature: float
) -> GaussMarkov:

    marginals = std_forward_message(reference_posterior)

    log_prior, log_transition, log_observation = \
        statistical_expansion(
            observations,
            log_prior_fn,
            log_transition_fn,
            log_observation_fn,
            reference_posterior.kernels,
            marginals,
        )

    damping = temperature / (1.0 + temperature)
    posterior, _, _, _, _ = log_backward_message(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        damping,
    )
    return posterior


def dual_objective(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    reference_posterior: GaussMarkov,
    kl_constraint: float,
    damping: float,
):
    posterior, log_norm, _, _, feasible = log_backward_message(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        damping,
    )

    def _feasible_objective():
        U, u, eta = log_norm
        m, _ = reference_posterior.marginal

        dual_value = damping * kl_constraint
        dual_value += - 0.5 * m.T @ U @ m + m.T @ u + eta
        return dual_value / (1.0 - damping)

    def _not_feasible_objective():
        return jnp.inf

    return jax.lax.cond(
        pred=jnp.all(feasible),
        true_fun=_feasible_objective,
        false_fun=_not_feasible_objective
    )


def vanilla_objective(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    reference_posterior: GaussMarkov,
):
    _, log_norm, _, _, _ = log_backward_message(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        0.0,
    )

    U, u, eta = log_norm
    m, _ = reference_posterior.marginal
    return - 0.5 * m.T @ U @ m + m.T @ u + eta


@partial(jax.jit, static_argnames=[
    'log_prior_fn',
    'log_transition_fn',
    'log_observation_fn',
    'kl_constraint',
    'init_temperature',
    'min_temperature',
    'max_iterations'
])
def iterated_forward_markov_smoother(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    init_posterior: GaussMarkov,
    kl_constraint: float,
    init_temperature: float = 1e12,
    min_temperature: float = 1e-12,
    max_iterations: int = 1000,
):
    """
    Iterated forward Markov smoother with early stopping based on temperature.

    This function performs variational inference by iteratively updating the posterior
    until convergence. The iterations stop when either:
    1. Maximum iterations are reached, or
    2. Temperature drops below min_temperature (indicating convergence)

    Args:
        observations: Array of observations
        log_prior_fn: Function to compute log prior
        log_transition_fn: Function to compute log transition
        log_observation_fn: Function to compute log observation likelihood
        init_posterior: Initial posterior estimate
        kl_constraint: KL divergence constraint for the optimization
        init_temperature: Initial temperature for line search
        min_temperature: Minimum temperature threshold for early stopping
        max_iterations: Maximum number of iterations

    Returns:
        Optimal posterior after convergence
    """

    def single_iteration(reference, iteration_idx):
        # Step 1: Compute marginals and statistical expansion
        marginals = std_forward_message(reference)
        log_prior, log_transition, log_observation = statistical_expansion(
            observations,
            log_prior_fn,
            log_transition_fn,
            log_observation_fn,
            reference.kernels,
            marginals,
        )

        # Step 2: Define dual objective function for line search
        def dual_objective_fn(temperature):
            """Dual objective function for temperature optimization."""
            damping = temperature / (1.0 + temperature)
            return dual_objective(
                log_prior,
                log_transition,
                log_observation,
                reference,
                kl_constraint,
                damping,
            )

        # Step 3: Define gradient function for line search
        def dual_gradient_fn(temperature):
            """Gradient of dual objective with respect to temperature."""
            damping = temperature / (1.0 + temperature)
            posterior, _, _, _, feasible_pass = log_backward_message(
                log_prior,
                log_transition,
                log_observation,
                reference,
                damping,
            )

            def compute_gradient():
                """Compute gradient when forward pass is feasible."""
                kl_div = kl_between_forward_gauss_markovs(
                    marginals=std_forward_message(posterior),
                    gauss_markov=posterior,
                    ref_gauss_markov=reference
                )
                return kl_constraint - kl_div

            def inf_gradient():
                """Return infinity when forward pass is not feasible."""
                return jnp.inf

            return jax.lax.cond(
                pred=jnp.all(feasible_pass),
                true_fun=lambda _: compute_gradient(),
                false_fun=lambda _: inf_gradient(),
                operand=None
            )

        # Step 4: Perform line search to find optimal temperature
        temperature, dual_value, _, line_search_feasible = line_search(
            init_temperature,
            dual_objective_fn,
            dual_gradient_fn,
            rtol=0.1 * kl_constraint
        )

        # Step 5: Apply the optimal temperature to get final posterior
        def apply_optimal_solution():
            """Apply the optimal temperature to compute final posterior."""
            damping = temperature / (1.0 + temperature)
            posterior, _, _, _, _ = log_backward_message(
                log_prior,
                log_transition,
                log_observation,
                reference,
                damping,
            )

            # Compute KL divergence for logging
            kl_div = kl_between_forward_gauss_markovs(
                marginals=std_forward_message(posterior),
                gauss_markov=posterior,
                ref_gauss_markov=reference
            )

            # Compute objective value for logging
            obj_value = vanilla_objective(
                log_prior,
                log_transition,
                log_observation,
                reference
            )

            # Log progress
            jax.debug.print(
                "iter: {iter}, damping: {damp}, kl_div: {kl}, dual: {dual}, val: {val}",
                iter=iteration_idx,
                damp=damping,
                kl=kl_div,
                dual=dual_value,
                val=obj_value
            )

            return posterior

        def use_reference():
            """Use reference posterior when line search fails."""
            jax.debug.print(
                "iter: {iter} not feasible, process might have converged",
                iter=iteration_idx
            )
            return reference

        # Choose between optimal solution and reference based on feasibility
        posterior = jax.lax.cond(
            pred=line_search_feasible,
            true_fun=lambda _: apply_optimal_solution(),
            false_fun=lambda _: use_reference(),
            operand=None
        )

        return posterior, temperature

    def iteration_body(carry):
        current_posterior, iteration_count, _ = carry
        next_posterior, next_temperature = single_iteration(current_posterior, iteration_count)
        return next_posterior, iteration_count + 1, next_temperature

    def iteration_condition(carry):
        _, iteration_count, next_temperature = carry
        # Continue if: not reached max iterations AND temperature is above minimum
        return jnp.logical_and(iteration_count < max_iterations, next_temperature > min_temperature)

    # Run the iterative optimization
    optimal_posterior, _, _ = bounded_while_loop(
        cond_fun=iteration_condition,
        body_fun=iteration_body,
        init_val=(init_posterior, 0, init_temperature),
        maxiter=max_iterations,
    )

    return optimal_posterior


@partial(jax.jit, static_argnames=[
    'log_prior_fn',
    'log_transition_fn',
    'log_observation_fn',
    'max_iterations'
])
def undamped_iterated_forward_markov_smoother(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    init_posterior: GaussMarkov,
    max_iterations: int
):

    def single_iteration(reference, iteration_idx):

        marginals = std_forward_message(reference)
        log_prior, log_transition, log_observation = \
            statistical_expansion(
                observations,
                log_prior_fn,
                log_transition_fn,
                log_observation_fn,
                reference.kernels,
                marginals,
            )

        optimal_posterior, _, _, _, _ = log_backward_message(
            log_prior,
            log_transition,
            log_observation,
            reference,
            0.0,
        )

        kl_div = kl_between_forward_gauss_markovs(
            marginals=std_forward_message(optimal_posterior),
            gauss_markov=optimal_posterior,
            ref_gauss_markov=reference
        )

        obj_val = vanilla_objective(
            log_prior,
            log_transition,
            log_observation,
            optimal_posterior
        )
        jax.debug.print(
            "iter: {a}, damping: {b}, kl_div: {c} val: {v}",
            a=iteration_idx, b=0.0, c=kl_div, v=obj_val
        )

        return optimal_posterior, optimal_posterior

    optimal_posterior, _ = jax.lax.scan(
        single_iteration, init_posterior, xs=jnp.arange(max_iterations)
    )
    return optimal_posterior
