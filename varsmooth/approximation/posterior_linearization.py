from typing import Callable
from functools import partial

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import (
    Gaussian,
    AdditiveGaussianModel,
    LogPrior,
    LogTransition,
    LogObservation,
)

_logdet = lambda x: jnp.linalg.slogdet(x)[1]


def get_log_prior(
    prior_dist: Gaussian,
    q: Gaussian,
    method: Callable,
):
    mu, Lambda = prior_dist
    return LogPrior(
        L=jsc.linalg.inv(Lambda),
        l=jsc.linalg.solve(Lambda, mu),
        nu=(
            - 0.5 * _logdet(2 * jnp.pi * Lambda)
            - 0.5 * mu.T @ jsc.linalg.solve(Lambda, mu)
        )
    )


@partial(jax.vmap, in_axes=(None, 0, None))
def get_log_transition(
    f: AdditiveGaussianModel,
    q: Gaussian,
    method: Callable
) -> LogTransition:

    A, b, Omega = method(f, q)
    return LogTransition(
        C11=jsc.linalg.inv(Omega),
        C12=jsc.linalg.solve(Omega, A),
        C21=jsc.linalg.solve(Omega, A).T,
        C22=A.T @ jsc.linalg.solve(Omega, A),
        c1=jsc.linalg.solve(Omega, b),
        c2=-A.T @ jsc.linalg.solve(Omega, b),
        kappa=(
            - 0.5 * _logdet(2 * jnp.pi * Omega)
            - 0.5 * b.T @ jsc.linalg.solve(Omega, b)
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
    return LogObservation(
        L=H.T @ jsc.linalg.solve(Delta, H),
        l=H.T @ jsc.linalg.solve(Delta, y - e),
        nu=(
            - 0.5 * _logdet(2 * jnp.pi * Delta)
            - 0.5 * (y - e).T @ jsc.linalg.solve(Delta, y - e)
        )
    )
