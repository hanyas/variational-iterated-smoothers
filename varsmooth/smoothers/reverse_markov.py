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


def forward_pass(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    nominal_posterior: GaussMarkov,
    beta: float,
) -> (GaussMarkov, jnp.ndarray):

    def _forward(carry, args):
        R, r, rho = carry
        C11, C12, C21, C22, c1, c2, kappa, \
            L, l, eta, \
            F, d, Sigma = args

        inv_Sigma = jnp.linalg.inv(Sigma)

        G11 = (1 - beta) * C11 + beta * F.T @ inv_Sigma @ F
        G22 = (1 - beta) * (C22 + R) + beta * inv_Sigma
        G21 = (1 - beta) * C21 + beta * inv_Sigma @ F
        g1 = (1 - beta) * c1 - beta * F.T @ inv_Sigma @ d
        g2 = (1 - beta) * (c2 + r) + beta * inv_Sigma @ d

        G11 = symmetrize(G11)
        G22 = symmetrize(G22)

        inv_G22 = jnp.linalg.inv(G22)
        S = G11 - G21.T @ inv_G22 @ G21
        s = g1 + G21.T @ inv_G22 @ g2

        R = L + 1 / (1 - beta) * S
        r = l + 1 / (1 - beta) * s
        rho = jnp.zeros((1, ))

        inv_G12 = jnp.linalg.inv(G21.T)

        F = inv_G12 @ (G11 - S)
        d = - inv_G12 @ (g1 - s)
        Sigma = inv_G12 @ (G11 - S) @ inv_G12.T

        return Potential(R, r, rho), AffineGaussian(F, d, Sigma)

    first_potential = Potential(
        R=log_prior.L,
        r=log_prior.l,
        rho=log_prior.eta
    )

    nominal_marginal, nominal_kernels = nominal_posterior

    last_potential, kernels = jax.lax.scan(
        f=_forward,
        init=first_potential,
        xs=(*log_transition, *log_observation, *nominal_kernels),
    )

    # get last marginal
    R, r, _ = last_potential

    m, P = nominal_marginal
    inv_P = jnp.linalg.inv(P)

    inv_aux = jnp.linalg.inv(beta * inv_P + (1 - beta) * R)
    U = beta * inv_P - beta**2 * inv_P @ inv_aux @ inv_P
    u = beta * (1 - beta) * inv_P @ inv_aux @ r

    m = m - 1 / beta * P @ (U @ m - u)
    P = 1 / beta * P - 1 / beta**2 * P @ U @ P
    marginal = Gaussian(m, P)

    return GaussMarkov(marginal, kernels)


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
    nominal_posterior: GaussMarkov,
    beta: float
) -> GaussMarkov:

    marginals = backward_pass(nominal_posterior)

    log_prior, log_transition, log_observation = \
        statistical_expansion(
            observations,
            prior_dist,
            transition_model,
            observation_model,
            cubature,
            marginals,
        )

    posterior = forward_pass(
        log_prior,
        log_transition,
        log_observation,
        nominal_posterior,
        beta,
    )

    return posterior
