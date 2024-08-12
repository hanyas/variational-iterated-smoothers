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
    kl_between_marginals,
    statistical_expansion
)
from varsmooth.utils import (
    none_or_concat,
    none_or_shift,
    symmetrize,
    eig
)

from varsmooth.smoothers.utils import line_search

_logdet = lambda x: jnp.linalg.slogdet(x)[1]


# @jax.jit
def forward_log_message(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    nominal_posterior: GaussMarkov,
    damping: float,
) -> (GaussMarkov, LogMarginalNorm, Potential, LogConditionalNorm):

    def _forward(carry, args):
        R, r, rho = carry
        C11, C12, C21, C22, c1, c2, kappa, \
            L, l, nu, \
            F, d, Sigma = args

        G11 = (1.0 - damping) * C11 + damping * F.T @ jsc.linalg.solve(Sigma, F)
        G22 = (1.0 - damping) * (C22 + R) + damping * jsc.linalg.inv(Sigma)
        G21 = (1.0 - damping) * C21 + damping * jsc.linalg.solve(Sigma, F)
        g1 = (1.0 - damping) * c1 - damping * F.T @ jsc.linalg.solve(Sigma, d)
        g2 = (1.0 - damping) * (c2 + r) + damping * jsc.linalg.solve(Sigma, d)
        theta = (
            (1.0 - damping) * (kappa + rho)
            - 0.5 * damping * _logdet(2 * jnp.pi * Sigma)
            - 0.5 * damping * d.T @ jsc.linalg.solve(Sigma, d)
        )

        G11 = symmetrize(G11)
        G22 = symmetrize(G22)

        def _feasible_forward_pass():
            S = G11 - G21.T @ jsc.linalg.solve(G22, G21)
            s = g1 + G21.T @ jsc.linalg.solve(G22, g2)
            xi = (
                theta
                + 0.5 * _logdet(2 * jnp.pi * jsc.linalg.inv(G22))
                + 0.5 * g2.T @ jsc.linalg.solve(G22, g2)
            )

            F = jsc.linalg.solve(G22, G21)
            d = jsc.linalg.solve(G22, g2)
            Sigma = jsc.linalg.inv(G22)

            R = L + 1.0 / (1.0 - damping) * S
            r = l + 1.0 / (1.0 - damping) * s
            rho = nu + 1.0 / (1.0 - damping) * xi

            potential = Potential(R, r, rho)
            return potential, (
                potential,
                AffineGaussian(F, d, Sigma),
                LogConditionalNorm(S, s, xi),
                True  # feasible
            )

        def _not_feasible_forward_pass():
            S = jnp.zeros_like(G11)
            s = jnp.zeros_like(g1)
            xi = jnp.zeros_like(theta)

            R = jnp.zeros_like(L)
            r = jnp.zeros_like(l)
            rho = jnp.zeros_like(nu)

            potential = Potential(R, r, rho)
            return potential, (
                potential,
                AffineGaussian(F, d, Sigma),
                LogConditionalNorm(S, s, xi),
                False   # Not feasible
            )

        return jax.lax.cond(
            pred=jnp.all(eig(G22)[0] > 1e-8),
            true_fun=_feasible_forward_pass,
            false_fun=_not_feasible_forward_pass,
        )

    first_potential = Potential(
        R=log_prior.L,
        r=log_prior.l,
        rho=log_prior.nu
    )

    nominal_marginal, nominal_kernels = nominal_posterior

    last_potential, (potentials, kernels, log_cond_norms, feasible_pass) = jax.lax.scan(
        f=_forward,
        init=first_potential,
        xs=(*log_transition, *log_observation, *nominal_kernels),
    )
    potentials = none_or_concat(potentials, first_potential, 1)

    R, r, rho = last_potential

    m, P = nominal_marginal
    inv_P = jsc.linalg.inv(P)

    J11 = (1.0 - damping) * R + damping * inv_P
    J12 = damping * inv_P
    J22 = damping * inv_P
    j1 = (1.0 - damping) * r
    j2 = jnp.zeros_like(j1)
    tau = (
        (1.0 - damping) * rho
        - 0.5 * damping * _logdet(2 * jnp.pi * P)
    )

    J11 = symmetrize(J11)
    J22 = symmetrize(J22)

    def _feasible_marginal():
        # init marginal
        _m = jsc.linalg.solve(J11, j1 + damping * inv_P @ m)
        _P = jsc.linalg.inv(J11)

        # log normalizer
        U = J22 - J12.T @ jsc.linalg.solve(J11, J12)
        u = j2 + J12.T @ jsc.linalg.solve(J11, j1)
        eta = (
            tau
            + 0.5 * _logdet(2 * jnp.pi * jsc.linalg.inv(J11))
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
        potentials,
        log_cond_norms,
        feasible_pass
    )


# @jax.jit
def backward_std_message(posterior: GaussMarkov) -> Gaussian:
    last_marginal, kernels = posterior

    def _backward(carry, args):
        q = carry
        kernel = args

        m, P = q
        F, d, Sigma = kernel

        qn = Gaussian(
            mean=F @ m + d,
            cov=F @ P @ F.T + Sigma
        )
        return qn, qn

    _, marginals = jax.lax.scan(_backward, last_marginal, kernels, reverse=True)
    return none_or_concat(marginals, last_marginal, position=-1)


def reverse_markov_smoother(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    reference_posterior: GaussMarkov,
    temperature: float
) -> GaussMarkov:

    marginals = backward_std_message(reference_posterior)

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
    posterior, _, _, _, _ = forward_log_message(
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
    posterior, lognorm, _, _, feasible = forward_log_message(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        damping,
    )

    def _feasible_objective():
        U, u, eta = lognorm
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
    _, lognorm, _, _, _ = forward_log_message(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        0.0,
    )

    U, u, eta = lognorm
    m, _ = reference_posterior.marginal
    return - 0.5 * m.T @ U @ m + m.T @ u + eta


@partial(jax.jit, static_argnums=(1, 2, 3, 5, 6, 7))
def iterated_reverse_markov_smoother(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    init_posterior: GaussMarkov,
    kl_constraint: float,
    init_temperature: float,
    max_iter: int
):
    optimal_posterior = init_posterior

    def _iteration(carry, args):
        reference = carry
        i = args

        marginals = backward_std_message(reference)
        log_prior, log_transition, log_observation = \
            statistical_expansion(
                observations,
                log_prior_fn,
                log_transition_fn,
                log_observation_fn,
                reference.kernels,
                marginals,
            )

        def dual_fn(_temperature):
            _damping = _temperature / (1.0 + _temperature)
            return dual_objective(
                log_prior,
                log_transition,
                log_observation,
                reference,
                kl_constraint,
                _damping,
            )

        def dual_gd(_temperature):
            _damping = _temperature / (1.0 + _temperature)
            _posterior, _, _, _, _feasible_pass = forward_log_message(
                log_prior,
                log_transition,
                log_observation,
                reference,
                _damping,
            )

            def _feasible_constraint():
                _kl_div = kl_between_reverse_gauss_markovs(
                    marginals=backward_std_message(_posterior),
                    gauss_markov=_posterior,
                    ref_gauss_markov=reference
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
            _posterior, _, _, _, _ = forward_log_message(
                log_prior,
                log_transition,
                log_observation,
                reference,
                _damping,
            )

            _kl_div = kl_between_reverse_gauss_markovs(
                marginals=backward_std_message(_posterior),
                gauss_markov=_posterior,
                ref_gauss_markov=reference
            )
            jax.debug.print(
                "iter: {a}, damping: {b}, kl_div: {c}, dual: {d}",
                a=i, b=_damping, c=_kl_div, d=dual_val
            )
            return _posterior

        def _not_feasible_line_search():
            jax.debug.print(
                "iter: {a} not feasible, process might have converged", a=i
            )
            return reference

        optimal_posterior = jax.lax.cond(
            pred=feasible,
            true_fun=_feasible_line_search,
            false_fun=_not_feasible_line_search
        )
        return optimal_posterior, optimal_posterior

    optimal_posterior, _ = jax.lax.scan(
        _iteration, init_posterior, xs=jnp.arange(max_iter)
    )
    return optimal_posterior


def kl_between_reverse_gauss_markovs(
    marginals, gauss_markov, ref_gauss_markov
):
    dim = gauss_markov.marginal.mean.shape[0]

    def body(carry, args):
        kl_value = carry
        m, P, F, d, Sigma, \
            ref_F, ref_d, ref_Sigma = args

        inv_ref_Sigma = jsc.linalg.inv(ref_Sigma)

        diff_F = (ref_F - F).T @ jsc.linalg.solve(ref_Sigma, ref_F - F)
        diff_d = (ref_d - d).T @ jsc.linalg.solve(ref_Sigma, ref_d - d)
        diff_cross = (ref_F - F).T @ jsc.linalg.solve(ref_Sigma, ref_d - d)

        kl_value += (
            0.5 * jnp.trace(diff_F @ P)
            + 0.5 * m.T @ diff_F @ m
            + m.T @ diff_cross
            + 0.5 * diff_d
            + 0.5 * jnp.trace(jsc.linalg.solve(ref_Sigma, Sigma))
            - 0.5 * dim
            + 0.5 * _logdet(ref_Sigma) - 0.5 * _logdet(Sigma)
        )
        return kl_value, kl_value

    init_kl_value = kl_between_marginals(
        gauss_markov.marginal, ref_gauss_markov.marginal
    )

    kl_value, _ = jax.lax.scan(
        f=body,
        init=init_kl_value,
        xs=(
            *none_or_shift(marginals, 1),
            *gauss_markov.kernels,
            *ref_gauss_markov.kernels
        ),
        reverse=True,
    )
    return kl_value
