from typing import Callable
from functools import partial

import jax
from jax import numpy as jnp

from varsmooth.objects import (
    Gaussian,
    AffineGaussian,
    AdditiveGaussianModel,
    GaussMarkov,
    LogPrior,
    LogTransition,
    LogObservation,
    Potential,
)
from varsmooth.utils import (
    none_or_concat,
    none_or_shift,
    none_or_idx,
    symmetrize
)

from varsmooth.linearization import cubature


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
    inv_Lambda = jnp.linalg.inv(Lambda)
    return LogPrior(
        L=inv_Lambda,
        l=inv_Lambda @ mu,
        eta=jnp.zeros((1,))
    )


@partial(jax.vmap, in_axes=(None, 0, None))
def get_log_transition(
    f: AdditiveGaussianModel,
    q: Gaussian,
    method: Callable
) -> LogTransition:

    A, b, Omega = method(f, q)
    inv_Omega = jnp.linalg.inv(Omega)
    return LogTransition(
        C11=inv_Omega,
        C12=inv_Omega @ A,
        C21=A.T @ inv_Omega,
        C22=A.T @ inv_Omega @ A,
        c1=inv_Omega @ b,
        c2=-A.T @ inv_Omega @ b,
        kappa=jnp.zeros((1,)),
    )


@partial(jax.vmap, in_axes=(0, None, 0, None))
def get_log_observation(
    y: jnp.ndarray,
    h: AdditiveGaussianModel,
    q: Gaussian,
    method: Callable
) -> LogObservation:

    H, e, Delta = method(h, q)
    inv_Delta = jnp.linalg.inv(Delta)
    return LogObservation(
        L=H.T @ inv_Delta @ H,
        l=H.T @ inv_Delta @ (y - e),
        eta=jnp.zeros((1,)),
    )


def statistical_expansion(
    observations: jnp.ndarray,
    prior_dist: Gaussian,
    transition_model: AdditiveGaussianModel,
    observation_model: AdditiveGaussianModel,
    integration_method: Callable,
    posterior_marginals: Gaussian
):

    log_prior = get_log_prior(prior_dist)

    prev_marginals = none_or_shift(posterior_marginals, -1)
    log_transition = get_log_transition(
        transition_model, prev_marginals, integration_method
    )

    next_marginals = none_or_shift(posterior_marginals, 1)
    log_observation = get_log_observation(
        observations, observation_model, next_marginals, integration_method
    )
    return log_prior, log_transition, log_observation


def backward_pass(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    nominal_posterior: GaussMarkov,
    beta: float,
) -> (GaussMarkov, jnp.ndarray):

    def _backward(carry, args):
        R, r, rho = carry
        C11, C12, C21, C22, c1, c2, kappa, \
            L, l, eta, \
            F, d, Sigma = args

        inv_Sigma = jnp.linalg.inv(Sigma)

        G11 = (1 - beta) * (C11 + R) + beta * inv_Sigma
        G22 = (1 - beta) * C22 + beta * F.T @ inv_Sigma @ F
        G12 = (1 - beta) * C12 + beta * inv_Sigma @ F
        g1 = (1 - beta) * (c1 + r) + beta * inv_Sigma @ d
        g2 = (1 - beta) * c2 - beta * F.T @ inv_Sigma @ d

        G11 = symmetrize(G11)
        G22 = symmetrize(G22)

        inv_G11 = jnp.linalg.inv(G11)
        S = G22 - G12.T @ inv_G11 @ G12
        s = g2 + G12.T @ inv_G11 @ g1

        R = L + 1 / (1 - beta) * S
        r = l + 1 / (1 - beta) * s
        rho = jnp.zeros((1, ))

        inv_G21 = jnp.linalg.inv(G12.T)

        F = inv_G21 @ (G22 - S)
        d = - inv_G21 @ (g2 - s)
        Sigma = inv_G21 @ (G22 - S) @ inv_G21.T

        return Potential(R, r, rho), AffineGaussian(F, d, Sigma)

    last_log_observation = none_or_idx(log_observation, -1)
    last_potential = Potential(
        R=last_log_observation.L,
        r=last_log_observation.l,
        rho=last_log_observation.eta
    )

    _log_aux_observation = none_or_concat(
        none_or_shift(log_observation, -1),
        LogObservation(log_prior.L, log_prior.l, log_prior.eta),
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
    R, r, _ = first_potential

    m, P = nominal_marginal
    inv_P = jnp.linalg.inv(P)

    inv_aux = jnp.linalg.inv(beta * inv_P + (1 - beta) * R)
    U = beta * inv_P - beta**2 * inv_P @ inv_aux @ inv_P
    u = beta * (1 - beta) * inv_P @ inv_aux @ r

    m = m - 1 / beta * P @ (U @ m - u)
    P = 1 / beta * P - 1 / beta**2 * P @ U @ P
    marginal = Gaussian(m, P)

    return GaussMarkov(marginal, kernels)


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
    return none_or_concat(marginals, init_marginal, 1)


def forward_markov_smoother(
    observations: jnp.ndarray,
    prior_dist: Gaussian,
    transition_model: AdditiveGaussianModel,
    observation_model: AdditiveGaussianModel,
    nominal_posterior: GaussMarkov,
    beta: float
) -> GaussMarkov:

    marginals = forward_pass(nominal_posterior)

    log_prior, log_transition, log_observation = \
        statistical_expansion(
            observations,
            prior_dist,
            transition_model,
            observation_model,
            cubature,
            marginals,
        )

    posterior = backward_pass(
        log_prior,
        log_transition,
        log_observation,
        nominal_posterior,
        beta,
    )

    return posterior
