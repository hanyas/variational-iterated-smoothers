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
    """
    This function is adapted from GPflow: https://github.com/GPflow/GPflow

    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = jnp.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = jnp.prod(jnp.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w


def _gauss_hermite_weights(num_dim, order):
    """
    Return weights and sigma-points for Gauss-Hermite cubature
    """
    # sigma_pts, weights = hermgauss(order)  # Gauss-Hermite sigma points and weights
    sigma_pts, weights = mvhermgauss(order, num_dim)
    sigma_pts = jnp.sqrt(2) * sigma_pts.T
    weights = weights.T * jnp.pi ** (-0.5 * num_dim)  # scale weights by 1/√π
    return weights, weights, sigma_pts
