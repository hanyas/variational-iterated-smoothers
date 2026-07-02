from typing import Callable, Tuple

from jax import Array
import jax.numpy as jnp

from varsmooth.approximation.sigma_points import SigmaPoints
from varsmooth.approximation.sigma_points import make_linearize
from varsmooth.approximation.sigma_points import quadratize_any
from varsmooth.objects import Gaussian


def quadratize(
    fun: Callable,
    q: Gaussian,
):
    """Quadratize a scalar function under q with the spherical-cubature rule."""
    return quadratize_any(fun, q, get_sigma_points)


def get_sigma_points(m: Array, chol_P: Array) -> SigmaPoints:
    """Return the spherical-cubature sigma points for N(m, chol_P chol_P^T)."""
    nb_dim = m.shape[0]
    wm, wc, xi = _cubature_weights(nb_dim)
    sigma_points = m[None, :] + jnp.dot(chol_P, xi).T
    return SigmaPoints(sigma_points, wm, wc, xi)


def _cubature_weights(
    nb_dim: int,
) -> Tuple[Array, Array, Array]:
    I_dim = jnp.eye(nb_dim)
    wm = jnp.ones(shape=(2 * nb_dim,)) / (2 * nb_dim)
    xi = jnp.concatenate([I_dim, -I_dim], axis=0) * nb_dim**0.5
    return wm, wm, xi.T


linearize = make_linearize(get_sigma_points)
