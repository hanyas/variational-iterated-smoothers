from typing import Tuple, Union

import jax.numpy as jnp

from varsmooth.objects import Gaussian
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel

from varsmooth.linearization.sigma_points import SigmaPoints
from varsmooth.linearization.sigma_points import linearize_additive
from varsmooth.linearization.sigma_points import linearize_conditional


def linearize(
    model: Union[AdditiveGaussianModel, ConditionalMomentsModel],
    q: Gaussian
):
    if isinstance(model, AdditiveGaussianModel):
        fun, noise = model
        return linearize_additive(fun, noise, q, _get_sigma_points)
    elif isinstance(model, ConditionalMomentsModel):
        mean_fn, covar_fn = model
        return linearize_conditional(mean_fn, covar_fn, q, _get_sigma_points)
    else:
        raise NotImplementedError


def _get_sigma_points(
    m: jnp.ndarray,
    chol_P: jnp.ndarray
) -> SigmaPoints:

    nb_dim = m.shape[0]
    wm, wc, xi = _cubature_weights(nb_dim)
    sigma_points = m[None, :] + jnp.dot(chol_P, xi.T).T
    return SigmaPoints(sigma_points, wm, wc)


def _cubature_weights(
    nb_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    I_dim = jnp.eye(nb_dim)
    wm = jnp.ones(shape=(2 * nb_dim,)) / (2 * nb_dim)
    wc = wm
    xi = jnp.concatenate([I_dim, -I_dim], axis=0) * nb_dim**0.5
    return wm, wc, xi
