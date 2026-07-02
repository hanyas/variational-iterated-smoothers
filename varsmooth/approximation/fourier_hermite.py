from functools import partial
from typing import Callable

import jax
from jax import Array
from jax import numpy as jnp

from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import GaussMarkov
from varsmooth.objects import Gaussian
from varsmooth.objects import LogLikelihood
from varsmooth.objects import LogPrior
from varsmooth.objects import LogTransition


def get_log_prior(
    prior_dist: Gaussian,
    q: Gaussian,
    method: Callable,
):
    """Quadratize the log-prior over x_0 under q via Fourier-Hermite.

    Args:
        prior_dist: Gaussian
            Prior over x_0 whose log-density is quadratized.
        q: Gaussian
            Expansion point under which the quadratization is taken.
        method: Callable
            Quadratization routine (logpdf, q) -> (M, v, c).

    Returns:
        LogPrior
            The quadratized log-prior -0.5 x^T L x + l^T x + nu.
    """
    L, l, nu = method(prior_dist.log_prob, q)
    return LogPrior(L, l, nu)


@partial(jax.vmap, in_axes=(None, 0, 0, None))
def get_log_transition(
    f: AdditiveGaussianModel, marginal: Gaussian, kernel: GaussMarkov, method: Callable
) -> LogTransition:
    """Quadratize the log-transition under the joint smoothing posterior.

    Forms the joint expansion point over z = (x_{k+1}, x_k) implied by the
    marginal over x_k and the affine-Gaussian kernel, then quadratizes the
    transition log-density f.log_prob(x_{k+1}, x_k) and slices the result into
    the LogTransition blocks. Applied per transition via jax.vmap over the (T,)
    batches of marginal and kernel.

    Args:
        f: AdditiveGaussianModel
            Transition model whose log-density is quadratized.
        marginal: Gaussian
            Marginal over x_k, with mean m and covariance P.
        kernel: AffineGaussian
            Current affine-Gaussian kernel (F, d, Sigma) mapping x_k to
            x_{k+1}; combined with marginal to build the joint expansion point.
        method: Callable
            Quadratization routine (logpdf, q) -> (C, c, kappa).

    Returns:
        LogTransition
            The quadratized transition as a log-potential over
            z = (x_{k+1}, x_k).
    """

    dim = marginal.mean.shape[0]

    m, P = marginal
    F, d, Sigma = kernel

    q = Gaussian(
        mean=jnp.hstack((F @ m + d, m)),
        cov=jnp.vstack((jnp.hstack((F @ P @ F.T + Sigma, F @ P)), jnp.hstack((P.T @ F.T, P)))),
    )

    logpdf = lambda z: f.log_prob(z[:dim], z[dim:])
    C, c, kappa = method(logpdf, q)
    return LogTransition(
        C11=C[:dim, :dim],
        C12=-C[:dim, dim:],
        C21=-C[dim:, :dim],
        C22=C[dim:, dim:],
        c1=c[:dim],
        c2=c[dim:],
        kappa=kappa,
    )


@partial(jax.vmap, in_axes=(0, None, 0, None))
def get_log_likelihood(y: Array, h: AdditiveGaussianModel, q: Gaussian, method: Callable) -> LogLikelihood:
    """Quadratize the log-likelihood under q via Fourier-Hermite.

    Quadratizes x -> h.log_prob(y, x) under q. Applied per observation via
    jax.vmap over the (T,) batches of y and q.

    Args:
        y: Array
            Observation of shape (dy,).
        h: AdditiveGaussianModel
            Observation model whose log-likelihood is quadratized.
        q: Gaussian
            Expansion point under which the quadratization is taken.
        method: Callable
            Quadratization routine (logpdf, q) -> (M, v, c).

    Returns:
        LogLikelihood
            The quadratized observation as a quadratic log-potential in x.
    """

    logpdf = lambda x: h.log_prob(y, x)
    L, l, nu = method(logpdf, q)
    return LogLikelihood(L, l, nu)
