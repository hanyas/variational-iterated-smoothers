from functools import partial
from typing import Callable

import jax
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import Gaussian
from varsmooth.objects import LogLikelihood
from varsmooth.objects import LogPrior
from varsmooth.objects import LogTransition
from varsmooth.utils import logdet


def get_log_prior(
    prior_dist: Gaussian,
    q: Gaussian,
    method: Callable,
):
    """Write a Gaussian prior over x_0 as its exact quadratic log-form.

    The prior N(mu, Lambda) is already log-quadratic, so no linearization is
    performed; q and method are accepted for a uniform interface and ignored.

    Args:
        prior_dist: Gaussian
            Prior over x_0 with mean mu and covariance Lambda.
        q: Gaussian
            Expansion point; unused, since the prior is exactly quadratic.
        method: Callable
            Linearization routine; unused for the same reason.

    Returns:
        LogPrior
            The prior as the quadratic log-form -0.5 x^T L x + l^T x + nu, with
            precision L = Lambda^-1.
    """
    mu, Lambda = prior_dist
    return LogPrior(
        L=jsc.linalg.inv(Lambda),
        l=jsc.linalg.solve(Lambda, mu),
        nu=(-0.5 * logdet(2 * jnp.pi * Lambda) - 0.5 * mu.T @ jsc.linalg.solve(Lambda, mu)),
    )


@partial(jax.vmap, in_axes=(None, 0, None))
def get_log_transition(f: AdditiveGaussianModel, q: Gaussian, method: Callable) -> LogTransition:
    """Statistically linearize the transition and write it as a log-transition.

    Linearizes f around q into an affine-Gaussian x_{k+1} | x_k = N(A x_k + b,
    Omega), then stores the joint log-quadratic over z = (x_{k+1}, x_k). Applied
    per transition via jax.vmap over the (T,) batch of expansion points q.

    Args:
        f: AdditiveGaussianModel
            Transition model to linearize.
        q: Gaussian
            Expansion point at which the linearization is taken.
        method: Callable
            Statistical-linearization routine (model, q) -> (A, b, Omega).

    Returns:
        LogTransition
            The linearized transition as a quadratic log-potential over
            z = (x_{k+1}, x_k).
    """

    A, b, Omega = method(f, q)
    return LogTransition(
        C11=jsc.linalg.inv(Omega),
        C12=jsc.linalg.solve(Omega, A),
        C21=jsc.linalg.solve(Omega, A).T,
        C22=A.T @ jsc.linalg.solve(Omega, A),
        c1=jsc.linalg.solve(Omega, b),
        c2=-A.T @ jsc.linalg.solve(Omega, b),
        kappa=(-0.5 * logdet(2 * jnp.pi * Omega) - 0.5 * b.T @ jsc.linalg.solve(Omega, b)),
    )


@partial(jax.vmap, in_axes=(0, None, 0, None))
def get_log_likelihood(y: Array, h: AdditiveGaussianModel, q: Gaussian, method: Callable) -> LogLikelihood:
    """Statistically linearize the observation and write it as a log-likelihood.

    Linearizes h around q into an affine-Gaussian y | x = N(H x + e, Delta),
    then stores the observation log-likelihood as a quadratic in x. Applied per
    observation via jax.vmap over the (T,) batches of y and q.

    Args:
        y: Array
            Observation of shape (dy,).
        h: AdditiveGaussianModel
            Observation model to linearize.
        q: Gaussian
            Expansion point at which the linearization is taken.
        method: Callable
            Statistical-linearization routine (model, q) -> (H, e, Delta).

    Returns:
        LogLikelihood
            The linearized observation as a quadratic log-potential in x.
    """

    H, e, Delta = method(h, q)
    return LogLikelihood(
        L=H.T @ jsc.linalg.solve(Delta, H),
        l=H.T @ jsc.linalg.solve(Delta, y - e),
        nu=(-0.5 * logdet(2 * jnp.pi * Delta) - 0.5 * (y - e).T @ jsc.linalg.solve(Delta, y - e)),
    )
