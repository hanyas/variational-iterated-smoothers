from functools import partial
import itertools
from typing import Callable

import jax
from jax import Array
import jax.numpy as jnp
from numpy.polynomial.hermite import hermgauss

from varsmooth.approximation.sigma_points import SigmaPoints
from varsmooth.approximation.sigma_points import make_linearize
from varsmooth.approximation.sigma_points import quadratize_any
from varsmooth.objects import Gaussian


def quadratize(
    fun: Callable,
    q: Gaussian,
    order: int = 3,
):
    """Quadratize a scalar function under q with Gauss-Hermite cubature."""
    _get_sigma_points = lambda m, chol_P: get_sigma_points(m, chol_P, order)
    return quadratize_any(fun, q, _get_sigma_points)


def linearize(
    model,
    q: Gaussian,
    order: int = 3,
):
    """Statistically linearize a model under q with Gauss-Hermite cubature."""
    _get_sigma_points = lambda m, chol_P: get_sigma_points(m, chol_P, order)
    return make_linearize(_get_sigma_points)(model, q)


@partial(jax.jit, static_argnums=(2,))
def get_sigma_points(m: Array, chol_P: Array, order: int) -> SigmaPoints:
    """Return the order-point Gauss-Hermite sigma points for N(m, chol_P chol_P^T)."""
    num_dim = m.shape[0]
    wm, wc, xi = _gauss_hermite_weights(num_dim, order)
    sigma_points = m[None, :] + (chol_P @ xi).T
    return SigmaPoints(sigma_points, wm, wc, xi)


# Following code adapted from BayesNewton Repository
# https://github.com/AaltoML/BayesNewton/blob/main/bayesnewton/cubature.py


def mvhermgauss(H: int, D: int):
    """Return evaluation locations and weights for multivariate Gauss-Hermite quadrature.

    Adapted from GPflow: https://github.com/GPflow/GPflow

    The outputs approximate integrals of the form
    int exp(-x) f(x) dx ~ sum_i w[i, :] f(x[i, :]).

    Args:
        H: int
            Number of Gauss-Hermite evaluation points per dimension.
        D: int
            Number of input dimensions; must be known at call time.

    Returns:
        x: Array
            Evaluation locations of shape (H**D, D).
        w: Array
            Quadrature weights of shape (H**D,).
    """
    gh_x, gh_w = hermgauss(H)
    x = jnp.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = jnp.prod(jnp.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w


def _gauss_hermite_weights(num_dim, order):
    """Return the Gauss-Hermite mean/covariance weights and unit sigma points."""
    # sigma_pts, weights = hermgauss(order)  # Gauss-Hermite sigma points and weights
    sigma_pts, weights = mvhermgauss(order, num_dim)
    sigma_pts = jnp.sqrt(2) * sigma_pts.T
    weights = weights.T * jnp.pi ** (-0.5 * num_dim)  # scale weights by 1/√π
    return weights, weights, sigma_pts
