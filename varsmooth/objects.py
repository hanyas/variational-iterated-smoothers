from typing import NamedTuple, Callable

import jax.numpy as jnp
import jax.scipy as jsc


class Gaussian(NamedTuple):
    mean: jnp.ndarray
    cov: jnp.ndarray

    def logpdf(self, x):
        diff = x - self.mean
        return (
            - 0.5 * diff.T @ jsc.linalg.solve(self.cov, diff)
            - 0.5 * jnp.linalg.slogdet(2.0 * jnp.pi * self.cov)[1]
        )


class AdditiveGaussianModel(NamedTuple):
    fun: Callable
    noise: Gaussian

    def logpdf(self, y, x):
        diff = y - self.fun(x)
        return (
            - 0.5 * diff.T @ jsc.linalg.solve(self.noise.cov, diff)
            - 0.5 * jnp.linalg.slogdet(2.0 * jnp.pi * self.noise.cov)[1]
        )


class ConditionalMomentsModel(NamedTuple):
    mean_fn: Callable
    cov_fn: Callable


class AffineGaussian(NamedTuple):
    F: jnp.ndarray
    d: jnp.ndarray
    Sigma: jnp.ndarray

    def logpdf(self, y, x):
        diff = y - self.F @ x - self.d
        return (
            - 0.5 * diff.T @ jsc.linalg.solve(self.Sigma, diff)
            - 0.5 * jnp.linalg.slogdet(2.0 * jnp.pi * self.Sigma)[1]
        )


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
