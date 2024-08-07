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
    LogMarginalNorm
)
from varsmooth.utils import (
    none_or_concat,
    none_or_shift,
    none_or_idx,
    symmetrize
)

import jaxopt

_logdet = lambda x: jnp.linalg.slogdet(x)[1]


def statistical_expansion(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    posterior_marginals: Gaussian
) -> (LogPrior, LogTransition, LogObservation):

    init_marginal = none_or_idx(posterior_marginals, 0)
    prev_marginals = none_or_shift(posterior_marginals, -1)
    next_marginals = none_or_shift(posterior_marginals, 1)

    log_prior = log_prior_fn(init_marginal)
    log_transition = log_transition_fn(prev_marginals)
    log_observation = log_observation_fn(observations, next_marginals)
    return log_prior, log_transition, log_observation


def backward_pass(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    nominal_posterior: GaussMarkov,
    damping: float,
) -> (GaussMarkov, LogMarginalNorm):

    def _backward(carry, args):
        R, r, rho = carry
        C11, C12, C21, C22, c1, c2, kappa, \
            L, l, nu, \
            F, d, Sigma = args

        inv_Sigma = jsc.linalg.inv(Sigma)

        G11 = (1.0 - damping) * (C11 + R) + damping * jsc.linalg.inv(Sigma)
        G22 = (1.0 - damping) * C22 + damping * F.T @ jsc.linalg.solve(Sigma, F)
        G12 = (1.0 - damping) * C12 + damping * jsc.linalg.solve(Sigma, F)
        g1 = (1.0 - damping) * (c1 + r) + damping * jsc.linalg.solve(Sigma, d)
        g2 = (1.0 - damping) * c2 - damping * F.T @ jsc.linalg.solve(Sigma, d)
        theta = (
            (1.0 - damping) * (kappa + rho)
            - 0.5 * damping * _logdet(2 * jnp.pi * Sigma)
            - 0.5 * damping * d.T @ jsc.linalg.solve(Sigma, d)
        )

        G11 = symmetrize(G11)
        G22 = symmetrize(G22)

        S = G22 - G12.T @ jsc.linalg.solve(G11, G12)
        s = g2 + G12.T @ jsc.linalg.solve(G11, g1)
        xi = (
            theta
            + 0.5 * _logdet(2 * jnp.pi * jsc.linalg.inv(G11))
            + 0.5 * g1.T @ jsc.linalg.solve(G11, g1)
        )

        R = L + 1.0 / (1.0 - damping) * S
        r = l + 1.0 / (1.0 - damping) * s
        rho = nu + 1.0 / (1.0 - damping) * xi

        F = jsc.linalg.solve(G11, G12)
        d = jsc.linalg.solve(G11, g1)
        Sigma = jsc.linalg.inv(G11)

        return Potential(R, r, rho), AffineGaussian(F, d, Sigma)

    last_log_observation = none_or_idx(log_observation, -1)
    last_potential = Potential(
        R=last_log_observation.L,
        r=last_log_observation.l,
        rho=last_log_observation.nu
    )

    _log_aux_observation = none_or_concat(
        none_or_shift(log_observation, -1),
        LogObservation(log_prior.L, log_prior.l, log_prior.nu),
        1
    )

    nominal_marginal, nominal_kernels = nominal_posterior

    first_potential, kernels = jax.lax.scan(
        f=_backward,
        init=last_potential,
        xs=(*log_transition, *_log_aux_observation, *nominal_kernels),
        reverse=True,
    )

    # get initial marginal
    R, r, rho = first_potential

    m, P = nominal_marginal
    inv_P = jsc.linalg.inv(P)

    J11 = (1.0 - damping) * R + damping * inv_P
    J12 = damping * inv_P
    J22 = damping * inv_P
    j1 = (1.0 - damping) * r
    j2 = jnp.zeros((1,))
    tau = (
        (1.0 - damping) * rho
        - 0.5 * damping * _logdet(2 * jnp.pi * P)
    )

    J11 = symmetrize(J11)
    J22 = symmetrize(J22)

    m = jsc.linalg.solve(J11, j1)
    P = jsc.linalg.inv(J11)
    marginal = Gaussian(m, P)

    # get log normalizer
    U = J22 - J12.T @ jsc.linalg.solve(J11, J12)
    u = j2 + J12.T @ jsc.linalg.solve(J11, j1)
    eta = (
        tau
        + 0.5 * _logdet(2 * jnp.pi * jsc.linalg.inv(J11))
        + 0.5 * j1.T @ jsc.linalg.solve(J11, j1)
    )

    return GaussMarkov(marginal, kernels), LogMarginalNorm(U, u, eta)


def forward_pass(posterior: GaussMarkov) -> Gaussian:
    init_marginal, kernels = posterior

    def _forward(carry, args):
        q = carry
        kernel = args

        m, P = q
        F, d, Sigma = kernel

        m = F @ m + d
        P = F @ P @ F.T + Sigma
        q = Gaussian(m, P)
        return q, q

    _, marginals = jax.lax.scan(_forward, init_marginal, kernels)
    return none_or_concat(marginals, init_marginal, position=1)


def forward_markov_smoother(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    nominal_posterior: GaussMarkov,
    temperature: float
) -> GaussMarkov:

    marginals = forward_pass(nominal_posterior)

    log_prior, log_transition, log_observation = \
        statistical_expansion(
            observations,
            log_prior_fn,
            log_transition_fn,
            log_observation_fn,
            marginals,
        )

    damping = temperature / (1.0 + temperature)
    posterior, _ = backward_pass(
        log_prior,
        log_transition,
        log_observation,
        nominal_posterior,
        damping,
    )
    return posterior


def dual_objective(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    reference_posterior: GaussMarkov,
    kl_constraint: float,
    temperature: float,
):
    damping = temperature / (1.0 + temperature)
    posterior, lognorm = backward_pass(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        damping,
    )

    U, u, eta = lognorm
    m, _ = reference_posterior.marginal

    dual_value = damping * kl_constraint
    dual_value += - 0.5 * m.T @ U @ m + m.T @ u + eta
    return dual_value / (1.0 - damping)


def vanilla_objective(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    reference_posterior: GaussMarkov,
):
    _, lognorm = backward_pass(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        0.0,
    )

    U, u, eta = lognorm
    m, _ = reference_posterior.marginal
    return - 0.5 * m.T @ U @ m + m.T @ u + eta


def optimize_temperature(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    reference_posterior: GaussMarkov,
    kl_constraint: float,
    init_temperature: float,
) -> (float, float):

    def dual_fn(temperature):
        _temperature = jnp.squeeze(temperature)
        return dual_objective(
            log_prior,
            log_transition,
            log_observation,
            reference_posterior,
            kl_constraint,
            _temperature,
        )

    dual_opt = jaxopt.LBFGSB(
        fun=dual_fn,
        tol=1e-3,
        maxiter=500,
        jit=True,
    )

    params, opt_state = dual_opt.run(
        init_params=jnp.atleast_1d(init_temperature),
        bounds=(1e-16, 1e16)
    )

    dual_value = opt_state.value
    opt_temperature = jnp.squeeze(params)
    return opt_temperature, dual_value


@partial(jax.jit, static_argnums=(1, 2, 3, 5, 6, 7))
def iterated_forward_markov_smoother(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    init_posterior: GaussMarkov,
    kl_constraint: float,
    init_temperature: float,
    nb_iter: int
):

    def _iteration(carry, args):
        reference = carry
        iter = args

        marginals = forward_pass(reference)
        log_prior, log_transition, log_observation = \
            statistical_expansion(
                observations,
                log_prior_fn,
                log_transition_fn,
                log_observation_fn,
                marginals,
            )

        objective = vanilla_objective(
            log_prior,
            log_transition,
            log_observation,
            reference,
        )

        temperature, _ = optimize_temperature(
            log_prior,
            log_transition,
            log_observation,
            reference,
            kl_constraint,
            init_temperature
        )

        damping = temperature / (1.0 + temperature)
        posterior, _ = backward_pass(
            log_prior,
            log_transition,
            log_observation,
            reference,
            damping,
        )

        kl_div = kl_between_gauss_markovs(
            marginals=forward_pass(posterior),
            gauss_markov=posterior,
            ref_gauss_markov=reference
        )
        jax.debug.print(
            "iter: {a}, damping: {b}, kl_div: {c}",
            a=iter, b=damping, c=kl_div
        )

        return posterior, posterior

    posterior, _ = jax.lax.scan(
        f=_iteration, init=init_posterior, xs=jnp.arange(nb_iter)
    )
    return posterior


def kl_between_marginals(p, q):
    dim = p.mean.shape[0]
    return 0.5 * (
        jnp.trace(jsc.linalg.inv(q.cov) @ p.cov) - dim
        + (q.mean - p.mean).T @ jsc.linalg.solve(q.cov, q.mean - p.mean)
        + _logdet(q.cov) - _logdet(p.cov)
    )


def kl_between_gauss_markovs(
    marginals, gauss_markov, ref_gauss_markov
):
    dim = gauss_markov.marginal.mean.shape[0]

    def body(carry, args):
        kl_value = carry
        m, P, F, d, Sigma, \
            ref_F, ref_d, ref_Sigma = args

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
            *none_or_shift(marginals, -1),
            *gauss_markov.kernels,
            *ref_gauss_markov.kernels
        )
    )
    return kl_value
