from typing import Callable

from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import (
    Gaussian,
    AffineGaussian,
    GaussMarkov,
    LogPrior,
    LogTransition,
    LogObservation,
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
    forward_markov: GaussMarkov,
    forward_marginals: Gaussian,
):

    rvs_Fs = jnp.zeros_like(forward_markov.kernels.F)
    rvs_ds = jnp.zeros_like(forward_markov.kernels.d)
    rvs_Sigmas = jnp.zeros_like(forward_markov.kernels.Sigma)

    nb_steps = rvs_Fs.shape[0]
    for k in range(nb_steps):
        m = forward_marginals.mean[k]
        P = forward_marginals.cov[k]

        fwd_F = forward_markov.kernels.F[k]
        fwd_d = forward_markov.kernels.d[k]
        fwd_Sigma = forward_markov.kernels.Sigma[k]

        a = m
        b = fwd_d + fwd_F @ m

        A = P
        C = P.T @ fwd_F.T
        B = fwd_F @ P @ fwd_F.T + fwd_Sigma

        rvs_F = jsc.linalg.solve(B, C.T).T
        rvs_d = a - C @ jsc.linalg.solve(B, b)
        rvs_Sigma = A - C @ jsc.linalg.solve(B, C.T)

        rvs_Fs = rvs_Fs.at[k].set(rvs_F)
        rvs_ds = rvs_ds.at[k].set(rvs_d)
        rvs_Sigmas = rvs_Sigmas.at[k].set(rvs_Sigma)

    m = forward_marginals.mean[-1]
    P = forward_marginals.cov[-1]

    reverse_markov = GaussMarkov(
        marginal=Gaussian(m, P),
        kernels=AffineGaussian(
            F=rvs_Fs, d=rvs_ds, Sigma=rvs_Sigmas
        )
    )
    return reverse_markov
