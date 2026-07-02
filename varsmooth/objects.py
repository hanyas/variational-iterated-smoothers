"""Core data structures for variational Gauss-Markov smoothing.

Shape conventions used throughout varsmooth:

    dx      state dimension
    dy      observation dimension
    T       number of transitions (equivalently, of observations)

A trajectory therefore has T + 1 marginals x_0, ..., x_T linked by T
Gauss-Markov kernels and explained by T observations y_1, ..., y_T. Batched
(scanned) quantities carry a leading axis of size T or T + 1 accordingly.
Other modules refer back to these conventions instead of re-stating them.

Quadratic log-potentials follow the notation of the paper "Recursive
Entropic Variational Inference for Nonlinear State-Space Models"
(arXiv:2511.15409). Every
single-variable potential below stores the log-quadratic form

    -0.5 x^T M x + v^T x + c

with the paper's symbols mapped onto the NamedTuple fields as follows:

    NamedTuple        M    v    c       Role
    LogPrior          L    l    nu      log prior over x_0
    LogLikelihood     L    l    nu      log-likelihood of y_k
    ValueFn           R    r    rho     backward/forward value function
    LogMessage        S    s    xi      eliminated-variable message
    LogNormalizer     U    u    eta     marginal log-normalizer

LogTransition is the two-variable analogue over z = (x_{k+1}, x_k): it stores
-0.5 z^T C z + [c1, c2]^T z + kappa with the block structure

    C = [[C11, -C12], [-C21, C22]].
"""

from typing import Callable, NamedTuple

from jax import Array
from jax import random
import jax.numpy as jnp
import jax.scipy as jsc

from varsmooth.utils import logdet


def quad_predict(M: Array, v: Array, c: Array, x: Array) -> Array:
    """Evaluate the log-quadratic form -0.5 x^T M x + v^T x + c at x.

    Args:
        M: Array
            Quadratic-form matrix of shape (dx, dx).
        v: Array
            Linear coefficient vector of shape (dx,).
        c: Array
            Scalar constant offset.
        x: Array
            Point of shape (dx,) at which to evaluate.

    Returns:
        Array
            The scalar value -0.5 x^T M x + v^T x + c.
    """
    return -0.5 * jnp.dot(x, jnp.dot(M, x)) + v @ x + c


class Gaussian(NamedTuple):
    """A multivariate Gaussian in moment form.

    Attributes:
        mean: Array
            Mean vector of shape (dx,).
        cov: Array
            Covariance matrix of shape (dx, dx).
    """

    mean: Array
    cov: Array

    def sample(self, key: Array) -> Array:
        """Draw a single sample from the Gaussian given a PRNG key."""
        return random.multivariate_normal(key, self.mean, self.cov)

    def log_prob(self, x: Array) -> Array:
        """Return the log density log N(x; mean, cov) evaluated at x."""
        diff = x - self.mean
        return -0.5 * diff.T @ jsc.linalg.solve(self.cov, diff) - 0.5 * logdet(2 * jnp.pi * self.cov)

    def sample_and_log_prob(self, key: Array) -> tuple[Array, Array]:
        """Draw a sample and return it together with its log density."""
        sample = self.sample(key)
        log_prob = self.log_prob(sample)
        return sample, log_prob


class AffineGaussian(NamedTuple):
    """An affine-Gaussian conditional kernel x -> N(F x + d, Sigma).

    Attributes:
        F: Array
            Linear map, shape (dx, dx) for a transition or (dy, dx) for an
            observation.
        d: Array
            Offset vector.
        Sigma: Array
            Conditional covariance matrix.
    """

    F: Array
    d: Array
    Sigma: Array

    def sample(self, key: Array, x: Array) -> Array:
        """Draw a single sample from the conditional N(F x + d, Sigma)."""
        return random.multivariate_normal(key, self.F @ x + self.d, self.Sigma)

    def log_prob(self, y: Array, x: Array) -> Array:
        """Return the conditional log density log N(y; F x + d, Sigma)."""
        diff = y - self.F @ x - self.d
        return -0.5 * diff.T @ jsc.linalg.solve(self.Sigma, diff) - 0.5 * logdet(2.0 * jnp.pi * self.Sigma)

    def sample_and_log_prob(self, key: Array, x: Array) -> tuple[Array, Array]:
        """Draw a conditional sample and return it with its log density."""
        sample = self.sample(key, x)
        log_prob = self.log_prob(sample, x)
        return sample, log_prob


class GaussMarkov(NamedTuple):
    """A Gauss-Markov chain: a boundary marginal plus T affine-Gaussian kernels.

    Attributes:
        marginal: Gaussian
            The boundary marginal -- the root x_0 for a forward chain or the
            leaf x_T for a reverse chain.
        kernels: AffineGaussian
            Batched affine-Gaussian kernels of leading shape (T,) linking
            successive states.
    """

    marginal: Gaussian
    kernels: AffineGaussian


class AdditiveGaussianModel(NamedTuple):
    """A model y = fun(x) + noise with additive Gaussian noise.

    Attributes:
        fun: Callable
            Deterministic map applied to the state.
        noise: Gaussian
            Additive Gaussian noise; its mean offsets fun and its covariance
            sets the observation/transition noise.
    """

    fun: Callable
    noise: Gaussian

    def log_prob(self, y, x):
        """Return the log density of y under the additive-noise model at state x."""
        diff = y - self.fun(x)
        return -0.5 * diff.T @ jsc.linalg.solve(self.noise.cov, diff) - 0.5 * logdet(2 * jnp.pi * self.noise.cov)


class ConditionalMomentsModel(NamedTuple):
    """A model with state-dependent Gaussian moments (mean_fn, cov_fn).

    Attributes:
        mean_fn: Callable
            Conditional mean E[y | x] as a function of x.
        cov_fn: Callable
            Conditional covariance Cov[y | x] as a function of x.
    """

    mean_fn: Callable
    cov_fn: Callable

    def log_prob(self, y, x):
        """Return the log density of y given x using the conditional moments."""
        diff = y - self.mean_fn(x)
        cov = self.cov_fn(x)
        return -0.5 * diff.T @ jsc.linalg.solve(cov, diff) - 0.5 * logdet(2 * jnp.pi * cov)


class LogMessage(NamedTuple):
    """Quadratic log-message with fields (S, s, xi); see the module notation table.

    Attributes:
        S: Array
            Quadratic-form matrix of shape (dx, dx).
        s: Array
            Linear coefficient vector of shape (dx,).
        xi: Array
            Scalar constant offset.
    """

    S: Array
    s: Array
    xi: Array

    def predict(self, x: Array) -> Array:
        """Evaluate the log-message quadratic at x."""
        return quad_predict(self.S, self.s, self.xi, x)


class LogNormalizer(NamedTuple):
    """Quadratic marginal log-normalizer with fields (U, u, eta).

    Attributes:
        U: Array
            Quadratic-form matrix of shape (dx, dx).
        u: Array
            Linear coefficient vector of shape (dx,).
        eta: Array
            Scalar constant offset.
    """

    U: Array
    u: Array
    eta: Array

    def predict(self, x: Array) -> Array:
        """Evaluate the log-normalizer quadratic at x."""
        return quad_predict(self.U, self.u, self.eta, x)


class ValueFn(NamedTuple):
    """Quadratic value function with fields (R, r, rho); see the module notation table.

    Attributes:
        R: Array
            Quadratic-form matrix of shape (dx, dx).
        r: Array
            Linear coefficient vector of shape (dx,).
        rho: Array
            Scalar constant offset.
    """

    R: Array
    r: Array
    rho: Array

    def predict(self, x: Array) -> Array:
        """Evaluate the value-function quadratic at x."""
        return quad_predict(self.R, self.r, self.rho, x)


class LogPrior(NamedTuple):
    """Quadratic log-prior over x_0 with fields (L, l, nu).

    Attributes:
        L: Array
            Quadratic-form matrix (prior precision), shape (dx, dx).
        l: Array
            Linear coefficient vector of shape (dx,).
        nu: Array
            Scalar constant offset.
    """

    L: Array
    l: Array
    nu: Array


class LogTransition(NamedTuple):
    """Quadratic log-transition over z = (x_{k+1}, x_k); see the module notation table.

    Stores -0.5 z^T C z + [c1, c2]^T z + kappa with block structure
    C = [[C11, -C12], [-C21, C22]].

    Attributes:
        C11: Array
            Top-left quadratic block, shape (dx, dx).
        C12: Array
            Top-right cross block; enters C with a negative sign as -C12.
        C21: Array
            Bottom-left cross block; enters C with a negative sign as -C21.
        C22: Array
            Bottom-right quadratic block, shape (dx, dx).
        c1: Array
            Linear coefficient for x_{k+1}, shape (dx,).
        c2: Array
            Linear coefficient for x_k, shape (dx,).
        kappa: Array
            Scalar constant offset.
    """

    C11: Array
    C12: Array
    C21: Array
    C22: Array
    c1: Array
    c2: Array
    kappa: Array


class LogLikelihood(NamedTuple):
    """Quadratic log-likelihood with fields (L, l, nu).

    Attributes:
        L: Array
            Quadratic-form matrix of shape (dx, dx).
        l: Array
            Linear coefficient vector of shape (dx,).
        nu: Array
            Scalar constant offset.
    """

    L: Array
    l: Array
    nu: Array

    def predict(self, x: Array) -> Array:
        """Evaluate the log-likelihood quadratic at x."""
        return quad_predict(self.L, self.l, self.nu, x)
