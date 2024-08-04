from typing import NamedTuple, Callable

import jax.numpy as jnp


class Gaussian(NamedTuple):
    mean: jnp.ndarray
    cov: jnp.ndarray


class AdditiveGaussianModel(NamedTuple):
    fun: Callable
    noise: Gaussian


class ConditionalMomentsModel(NamedTuple):
    mean_fn: Callable
    cov_fn: Callable


class AffineGaussian(NamedTuple):
    F: jnp.ndarray
    d: jnp.ndarray
    Sigma: jnp.ndarray

    def logpdf(self, y, x):
        err = y - self.F @ x - self.d
        return - 0.5 * err.T @ jnp.linalg.inv(self.Sigma) @ err\
            - 0.5 * jnp.linalg.slogdet(2.0 * jnp.pi * self.Sigma)[1]


class GaussMarkov(NamedTuple):
    marginal: Gaussian
    kernels: AffineGaussian


class LogMarginalNorm(NamedTuple):
    U: jnp.ndarray
    u: jnp.ndarray
    eta: jnp.ndarray


class Potential(NamedTuple):
    R: jnp.ndarray
    r: jnp.ndarray
    rho: jnp.ndarray


class LogPrior(NamedTuple):
    L: jnp.ndarray
    l: jnp.ndarray
    nu: jnp.ndarray


class LogTransition(NamedTuple):
    C11: jnp.ndarray
    C12: jnp.ndarray
    C21: jnp.ndarray
    C22: jnp.ndarray
    c1: jnp.ndarray
    c2: jnp.ndarray
    kappa: jnp.ndarray


class LogObservation(NamedTuple):
    L: jnp.ndarray
    l: jnp.ndarray
    nu: jnp.ndarray
