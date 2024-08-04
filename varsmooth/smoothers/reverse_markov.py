from typing import Callable
from functools import partial

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import (
    Gaussian,
    AffineGaussian,
    AdditiveGaussianModel,
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
    symmetrize
)

import jaxopt

_logdet = lambda x: jnp.linalg.slogdet(x)[1]


@partial(jax.vmap, in_axes=(None, 0, None))
def get_linear_model(
    f: AdditiveGaussianModel,
    q: Gaussian,
    method: Callable
) -> AffineGaussian:
    return AffineGaussian(*method(f, q))


def get_log_prior(
    prior_dist: Gaussian,
):
    mu, Lambda = prior_dist
    inv_Lambda = jsc.linalg.inv(Lambda)
    return LogPrior(
        L=inv_Lambda,
        l=inv_Lambda @ mu,
        nu=(
            - 0.5 * _logdet(2 * jnp.pi * Lambda)
            - 0.5 * mu.T @ inv_Lambda @ mu
        )
    )


@partial(jax.vmap, in_axes=(None, 0, None))
def get_log_transition(
    f: AdditiveGaussianModel,
    q: Gaussian,
    method: Callable
) -> LogTransition:

    A, b, Omega = method(f, q)
    inv_Omega = jsc.linalg.inv(Omega)
    return LogTransition(
        C11=inv_Omega,
        C12=inv_Omega @ A,
        C21=A.T @ inv_Omega,
        C22=A.T @ inv_Omega @ A,
        c1=inv_Omega @ b,
        c2=-A.T @ inv_Omega @ b,
        kappa=(
            - 0.5 * _logdet(2 * jnp.pi * Omega)
            - 0.5 * b.T @ inv_Omega @ b
        ),
    )


@partial(jax.vmap, in_axes=(0, None, 0, None))
def get_log_observation(
    y: jnp.ndarray,
    h: AdditiveGaussianModel,
    q: Gaussian,
    method: Callable
) -> LogObservation:

    H, e, Delta = method(h, q)
    inv_Delta = jsc.linalg.inv(Delta)
    return LogObservation(
        L=H.T @ inv_Delta @ H,
        l=H.T @ inv_Delta @ (y - e),
        nu=(
            - 0.5 * _logdet(2 * jnp.pi * Delta)
            - 0.5 * (y - e).T @ inv_Delta @ (y - e)
        )
    )


def statistical_expansion(
    observations: jnp.ndarray,
    prior_dist: Gaussian,
    transition_model: AdditiveGaussianModel,
    observation_model: AdditiveGaussianModel,
    approximation_method: Callable,
    posterior_marginals: Gaussian
):
    log_prior = get_log_prior(prior_dist)

    prev_marginals = none_or_shift(posterior_marginals, -1)
    log_transition = get_log_transition(
        transition_model, prev_marginals, approximation_method
    )

    next_marginals = none_or_shift(posterior_marginals, 1)
    log_observation = get_log_observation(
        observations, observation_model, next_marginals, approximation_method
    )
    return log_prior, log_transition, log_observation


def forward_pass(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    nominal_posterior: GaussMarkov,
    damping: float,
) -> (GaussMarkov, LogMarginalNorm):

    def _forward(carry, args):
        R, r, rho = carry
        C11, C12, C21, C22, c1, c2, kappa, \
            L, l, nu, \
            F, d, Sigma = args

        inv_Sigma = jsc.linalg.inv(Sigma)

        G11 = (1.0 - damping) * C11 + damping * F.T @ inv_Sigma @ F
        G22 = (1.0 - damping) * (C22 + R) + damping * inv_Sigma
        G21 = (1.0 - damping) * C21 + damping * inv_Sigma @ F
        g1 = (1.0 - damping) * c1 - damping * F.T @ inv_Sigma @ d
        g2 = (1.0 - damping) * (c2 + r) + damping * inv_Sigma @ d
        theta = (
            (1.0 - damping) * (kappa + rho)
            - 0.5 * damping * _logdet(2 * jnp.pi * Sigma)
            - 0.5 * damping * d.T @ inv_Sigma @ d
        )

        G11 = symmetrize(G11)
        G22 = symmetrize(G22)

        inv_G22 = jsc.linalg.inv(G22)
        S = G11 - G21.T @ inv_G22 @ G21
        s = g1 + G21.T @ inv_G22 @ g2
        xi = (
            theta
            + 0.5 * _logdet(2 * jnp.pi * inv_G22)
            + 0.5 * g2.T @ inv_G22 @ g2
        )

        R = L + 1.0 / (1.0 - damping) * S
        r = l + 1.0 / (1.0 - damping) * s
        rho = nu + 1.0 / (1.0 - damping) * xi

        inv_G12 = jsc.linalg.inv(G21.T)

        F = inv_G22 @ G21
        d = inv_G22 @ g2
        Sigma = inv_G22

        return Potential(R, r, rho), AffineGaussian(F, d, Sigma)

    first_potential = Potential(
        R=log_prior.L,
        r=log_prior.l,
        rho=log_prior.nu
    )

    nominal_marginal, nominal_kernels = nominal_posterior

    last_potential, kernels = jax.lax.scan(
        f=_forward,
        init=first_potential,
        xs=(*log_transition, *log_observation, *nominal_kernels),
    )

    # get last marginal
    R, r, rho = last_potential

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

    inv_J11 = jsc.linalg.inv(J11)

    m = inv_J11 @ j1
    P = inv_J11
    marginal = Gaussian(m, P)

    # get log normalizer
    U = J22 - J12.T @ inv_J11 @ J12
    u = j2 + J12.T @ inv_J11 @ j1
    eta = (
        tau
        + 0.5 * _logdet(2 * jnp.pi * inv_J11)
        + 0.5 * j1.T @ inv_J11 @ j1
    )

    return GaussMarkov(marginal, kernels), LogMarginalNorm(U, u, eta)


def backward_pass(posterior: GaussMarkov) -> Gaussian:
    last_marginal, kernels = posterior

    def _backward(carry, args):
        q = carry
        kernel = args

        m, P = q
        F, d, Sigma = kernel

        m = F @ m + d
        P = F @ P @ F.T + Sigma
        q = Gaussian(m, P)
        return q, q

    _, marginals = jax.lax.scan(_backward, last_marginal, kernels, reverse=True)
    return none_or_concat(marginals, last_marginal, position=-1)


def reverse_markov_smoother(
    observations: jnp.ndarray,
    prior_dist: Gaussian,
    transition_model: AdditiveGaussianModel,
    observation_model: AdditiveGaussianModel,
    approximation_method: Callable,
    nominal_posterior: GaussMarkov,
    temperature: float
) -> GaussMarkov:

    marginals = backward_pass(nominal_posterior)

    log_prior, log_transition, log_observation = \
        statistical_expansion(
            observations,
            prior_dist,
            transition_model,
            observation_model,
            approximation_method,
            marginals,
        )

    damping = temperature / (1.0 + temperature)
    posterior, _ = forward_pass(
        log_prior,
        log_transition,
        log_observation,
        nominal_posterior,
        damping,
    )
    return posterior


def optimize_step(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    reference_posterior: GaussMarkov,
    kl_constraint: float,
    init_temperature: float,
) -> float:

    def dual_objective(temperature):
        damping = temperature / (1.0 + temperature)
        posterior, lognorm = forward_pass(
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

    dual_opt = jaxopt.ScipyBoundedMinimize(
        fun=dual_objective,
        method="L-BFGS-B",
        tol=1e-3,
        maxiter=500,
        jit=True,
    )

    opt_temperature = dual_opt.run(init_temperature, bounds=(1e-16, 1e16)).params
    opt_damping = opt_temperature / (1.0 + opt_temperature)
    return opt_damping


def iterated_reverse_markov_smoother(
    observations: jnp.ndarray,
    prior_dist: Gaussian,
    transition_model: AdditiveGaussianModel,
    observation_model: AdditiveGaussianModel,
    approximation_method: Callable,
    initial_posterior: GaussMarkov,
    kl_constraint: float,
    init_temperature: float,
    nb_iter: int
):
    optimal_posterior = initial_posterior

    for i in range(nb_iter):
        reference = optimal_posterior

        marginals = backward_pass(reference)
        log_prior, log_transition, log_observation = \
            statistical_expansion(
                observations,
                prior_dist,
                transition_model,
                observation_model,
                approximation_method,
                marginals,
            )

        optimal_damping = optimize_step(
            log_prior,
            log_transition,
            log_observation,
            reference,
            kl_constraint,
            init_temperature
        )

        optimal_posterior, _ = forward_pass(
            log_prior,
            log_transition,
            log_observation,
            reference,
            optimal_damping,
        )

        kl_div = kl_between_gauss_markovs(
            marginals=backward_pass(optimal_posterior),
            gauss_markov=optimal_posterior,
            ref_gauss_markov=reference
        )
        print(f"iter: {i:d}, damping: {optimal_damping:.3f}, kl_div: {kl_div:.3f}")

    return optimal_posterior


def kl_between_marginals(p, q):
    dim = p.mean.shape[0]
    return 0.5 * (
        jnp.trace(jsc.linalg.inv(q.cov) @ p.cov) - dim
        + (q.mean - p.mean).T @ jsc.linalg.inv(q.cov) @ (q.mean - p.mean)
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

        inv_ref_Sigma = jsc.linalg.inv(ref_Sigma)

        diff_F = (ref_F - F).T @ inv_ref_Sigma @ (ref_F - F)
        diff_d = (ref_d - d).T @ inv_ref_Sigma @ (ref_d - d)
        diff_cross = (ref_F - F).T @ inv_ref_Sigma @ (ref_d - d)

        kl_value += (
            0.5 * jnp.trace(diff_F @ P)
            + 0.5 * m.T @ diff_F @ m
            + m.T @ diff_cross
            + 0.5 * diff_d
            + 0.5 * jnp.trace(inv_ref_Sigma @ Sigma)
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
