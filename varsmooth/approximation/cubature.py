from typing import Callable, Tuple, Union

import jax.numpy as jnp
from jax import Array

from varsmooth.approximation.sigma_points import SigmaPoints, linearize_additive, linearize_conditional, quadratize_any
from varsmooth.objects import AdditiveGaussianModel, ConditionalMomentsModel, Gaussian


def quadratize(
    fun: Callable,
    q: Gaussian,
):
    return quadratize_any(fun, q, get_sigma_points)


def linearize(model: Union[AdditiveGaussianModel, ConditionalMomentsModel], q: Gaussian):
    if isinstance(model, AdditiveGaussianModel):
        fun, noise = model
        return linearize_additive(fun, noise, q, get_sigma_points)
    elif isinstance(model, ConditionalMomentsModel):
        mean_fn, covar_fn = model
        return linearize_conditional(mean_fn, covar_fn, q, get_sigma_points)
    else:
        raise NotImplementedError


def get_sigma_points(m: Array, chol_P: Array) -> SigmaPoints:

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
