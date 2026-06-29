from typing import Callable, NamedTuple

from jax import Array
from jax import random
import jax.numpy as jnp
import jax.scipy as jsc

from varsmooth.utils import logdet


def quad_predict(M: Array, v: Array, c: Array, x: Array) -> Array:
    """Evaluate the log-quadratic form ``-0.5 xᵀ M x + vᵀ x + c`` at ``x``."""
    return -0.5 * jnp.dot(x, jnp.dot(M, x)) + v @ x + c


class Gaussian(NamedTuple):
    mean: Array
    cov: Array

    def sample(self, key: Array) -> Array:
        return random.multivariate_normal(key, self.mean, self.cov)

    def log_prob(self, x: Array) -> Array:
        diff = x - self.mean
        return -0.5 * diff.T @ jsc.linalg.solve(self.cov, diff) - 0.5 * logdet(2 * jnp.pi * self.cov)

    def sample_and_log_prob(self, key: Array) -> tuple[Array, Array]:
        sample = self.sample(key)
        log_prob = self.log_prob(sample)
        return sample, log_prob


class AffineGaussian(NamedTuple):
    F: Array
    d: Array
    Sigma: Array

    def sample(self, key: Array, x: Array) -> Array:
        return random.multivariate_normal(key, self.F @ x + self.d, self.Sigma)

    def log_prob(self, y: Array, x: Array) -> Array:
        diff = y - self.F @ x - self.d
        return -0.5 * diff.T @ jsc.linalg.solve(self.Sigma, diff) - 0.5 * logdet(2.0 * jnp.pi * self.Sigma)

    def sample_and_log_prob(self, key: Array, x: Array) -> tuple[Array, Array]:
        sample = self.sample(key, x)
        log_prob = self.log_prob(sample, x)
        return sample, log_prob


class GaussMarkov(NamedTuple):
    marginal: Gaussian
    kernels: AffineGaussian


class AdditiveGaussianModel(NamedTuple):
    fun: Callable
    noise: Gaussian

    def log_prob(self, y, x):
        diff = y - self.fun(x)
        return -0.5 * diff.T @ jsc.linalg.solve(self.noise.cov, diff) - 0.5 * logdet(2 * jnp.pi * self.noise.cov)


class ConditionalMomentsModel(NamedTuple):
    mean_fn: Callable
    cov_fn: Callable


class LogMessage(NamedTuple):
    S: Array
    s: Array
    xi: Array

    def predict(self, x: Array) -> Array:
        return quad_predict(self.S, self.s, self.xi, x)


class LogMarginalNorm(NamedTuple):
    U: Array
    u: Array
    eta: Array

    def predict(self, x: Array) -> Array:
        return quad_predict(self.U, self.u, self.eta, x)


class ValueFn(NamedTuple):
    R: Array
    r: Array
    rho: Array

    def predict(self, x: Array) -> Array:
        return quad_predict(self.R, self.r, self.rho, x)


class LogPrior(NamedTuple):
    L: Array
    l: Array
    nu: Array


class LogTransition(NamedTuple):
    C11: Array
    C12: Array
    C21: Array
    C22: Array
    c1: Array
    c2: Array
    kappa: Array


class LogObservation(NamedTuple):
    L: Array
    l: Array
    nu: Array

    def predict(self, x: Array) -> Array:
        return quad_predict(self.L, self.l, self.nu, x)
