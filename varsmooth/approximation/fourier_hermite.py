from typing import Callable
from functools import partial

import jax
from jax import numpy as jnp

from varsmooth.objects import (
    Gaussian,
    GaussMarkov,
    AdditiveGaussianModel,
    LogPrior,
    LogTransition,
    LogObservation,
)


def get_log_prior(
    prior_dist: Gaussian,
    q: Gaussian,
    method: Callable,
):
    L, l, nu = method(prior_dist.logpdf, q)
    return LogPrior(L, l, nu)


@partial(jax.vmap, in_axes=(None, 0, 0, None))
def get_log_transition(
    f: AdditiveGaussianModel,
    marginal: Gaussian,
    kernel: GaussMarkov,
    method: Callable
) -> LogTransition:

    dim = marginal.mean.shape[0]

    m, P = marginal
    F, d, Sigma = kernel

    q = Gaussian(
        mean=jnp.hstack((F @ m + d, m)),
        cov=jnp.vstack((
            jnp.hstack((F @ P @ F.T + Sigma, F @ P)),
            jnp.hstack((P.T @ F.T, P))
        ))
    )

    logpdf = lambda z: f.logpdf(z[:dim], z[dim:])
    C, c, kappa = method(logpdf, q)
    return LogTransition(
        C11=C[:dim, :dim],
        C12=-C[:dim, dim:],
        C21=-C[dim:, :dim],
        C22=C[dim:, dim:],
        c1=c[:dim],
        c2=c[dim:],
        kappa=kappa
    )


@partial(jax.vmap, in_axes=(0, None, 0, None))
def get_log_observation(
    y: jnp.ndarray,
    h: AdditiveGaussianModel,
    q: Gaussian,
    method: Callable
) -> LogObservation:

    logpdf = lambda x: h.logpdf(y, x)
    L, l, nu = method(logpdf, q)
    return LogObservation(L, l, nu)
