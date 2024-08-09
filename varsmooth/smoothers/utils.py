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
    LogConditionalNorm
)
from varsmooth.utils import none_or_idx, none_or_shift

_logdet = lambda x: jnp.linalg.slogdet(x)[1]


def kl_between_marginals(p, q):
    dim = p.mean.shape[0]
    return 0.5 * (
        jnp.trace(jsc.linalg.inv(q.cov) @ p.cov) - dim
        + (q.mean - p.mean).T @ jsc.linalg.solve(q.cov, q.mean - p.mean)
        + _logdet(q.cov) - _logdet(p.cov)
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def statistical_expansion(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    posterior_kernels: AffineGaussian,
    posterior_marginals: Gaussian
) -> (LogPrior, LogTransition, LogObservation):

    init_marginal = none_or_idx(posterior_marginals, 0)
    prev_marginals = none_or_shift(posterior_marginals, -1)
    next_marginals = none_or_shift(posterior_marginals, 1)

    log_prior = log_prior_fn(init_marginal)
    log_transition = log_transition_fn(prev_marginals, posterior_kernels)
    log_observation = log_observation_fn(observations, next_marginals)
    return log_prior, log_transition, log_observation


def initialize_reverse_with_forward(
    forward_markov: GaussMarkov
):
    from varsmooth.smoothers.forward_markov import forward_std_message

    forward_marginals = forward_std_message(forward_markov)

    Fs = jnp.zeros_like(forward_markov.kernels.F)
    ds = jnp.zeros_like(forward_markov.kernels.d)
    Sigmas = jnp.zeros_like(forward_markov.kernels.Sigma)

    nb_steps = forward_markov.kernels.F.shape[0]
    for k in range(nb_steps):

        marginal = none_or_idx(forward_marginals, k)  # marginal at k
        kernel = none_or_idx(forward_markov.kernels, k)  # kernel k+1 | k
        next_marginal = none_or_idx(forward_marginals, k+1)  # marginal at k+1

        _kernel = get_reverse_kernel(marginal, kernel, next_marginal)

        Fs = Fs.at[k].set(_kernel.F)
        ds = ds.at[k].set(_kernel.d)
        Sigmas = Sigmas.at[k].set(_kernel.Sigma)

    reverse_markov = GaussMarkov(
        Gaussian(
            mean=forward_marginals.mean[-1],
            cov=forward_marginals.cov[-1],
        ),
        kernels=AffineGaussian(
            F=Fs, d=ds, Sigma=Sigmas
        )
    )
    return reverse_markov


def get_pairwise_marginal(
    marginal: Gaussian,
    kernel: AffineGaussian
):
    m, P = marginal
    F, d, Sigma = kernel

    q = Gaussian(
        mean=jnp.hstack((F @ m + d, m)),
        cov=jnp.vstack((
            jnp.hstack((F @ P @ F.T + Sigma, F @ P)),
            jnp.hstack((P.T @ F.T, P))
        ))
    )
    return q


def get_conditional(
    marginal: Gaussian,
    pairwise: Gaussian
):
    dim = marginal.mean.shape[0]

    a = pairwise.mean[:dim]
    b = pairwise.mean[dim:]

    A = pairwise.cov[:dim, :dim]
    B = pairwise.cov[dim:, dim:]
    C = pairwise.cov[:dim, dim:]

    return AffineGaussian(
        F=jsc.linalg.solve(A, C).T,
        d=b - C.T @ jsc.linalg.solve(A, a),
        Sigma=B - C.T @ jsc.linalg.solve(A, C)
    )


def get_reverse_kernel(
    marginal: Gaussian,
    kernel: AffineGaussian,
    next_marginal: Gaussian
):
    pairwise = get_pairwise_marginal(marginal, kernel)
    return get_conditional(next_marginal, pairwise)


def merge_messages(
    fwd_message: Potential,
    bwd_message: LogConditionalNorm,
) -> Potential:
    return Potential(
        R=fwd_message.R + bwd_message.S,
        r=fwd_message.r + bwd_message.s,
        rho=fwd_message.rho + bwd_message.xi,
    )


def log_to_std_form(
    potential: Potential
) -> Gaussian:
    return Gaussian(
        mean=jsc.linalg.inv(potential.R) @ potential.r,
        cov=jsc.linalg.inv(potential.R)
    )
