from typing import Tuple, Optional, Union, Callable

import jax.numpy as jnp

from varsmooth.objects import Gaussian
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel

from varsmooth.approximation.sigma_points import SigmaPoints
from varsmooth.approximation.sigma_points import linearize_additive
from varsmooth.approximation.sigma_points import linearize_conditional
from varsmooth.approximation.sigma_points import quadratize_any


def quadratize(
    fun: Callable,
    q: Gaussian,
    alpha: float = 1.0,
    beta: float = 0.0,
    kappa: float = None,
):
    _get_sigma_points = \
        lambda m, chol_P: get_sigma_points(m, chol_P, alpha, beta, kappa)
    return quadratize_any(fun, q, _get_sigma_points)


def linearize(
    model: Union[AdditiveGaussianModel, ConditionalMomentsModel],
    q: Gaussian,
    alpha: float = 1.0,
    beta: float = 0.0,
    kappa: float = None,
):
    _get_sigma_points = \
        lambda m, chol_P: get_sigma_points(m, chol_P, alpha, beta, kappa)

    if isinstance(model, AdditiveGaussianModel):
        fun, noise = model
        return linearize_additive(fun, noise, q, _get_sigma_points)
    elif isinstance(model, ConditionalMomentsModel):
        mean_fn, cov_fn = model
        return linearize_conditional(mean_fn, cov_fn, q, _get_sigma_points)
    else:
        raise NotImplementedError


def get_sigma_points(
    m: jnp.ndarray,
    chol_P: jnp.ndarray,
    alpha: float,
    beta: float,
    kappa: Optional[float]
) -> SigmaPoints:

    nb_dim = m.shape[0]
    if kappa is None:
        kappa = 3.0 + nb_dim

    wm, wc, xi = _unscented_weights(nb_dim, alpha, beta, kappa)
    sigma_points = m[None, :] + jnp.dot(chol_P, xi).T
    return SigmaPoints(sigma_points, wm, wc, xi)


def _unscented_weights(
    nb_dim: int,
    alpha: float,
    beta: float,
    kappa: Optional[float]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    lamda = alpha**2 * (nb_dim + kappa) - nb_dim
    wm = jnp.full(2 * nb_dim + 1, 1 / (2 * (nb_dim + lamda)))

    wm = wm.at[0].set(lamda / (nb_dim + lamda))
    wc = wm.at[0].set(lamda / (nb_dim + lamda) + (1 - alpha**2 + beta))

    zeros = jnp.zeros((1, nb_dim))
    I_dim = jnp.eye(nb_dim)

    xi = jnp.sqrt(nb_dim + lamda) * jnp.concatenate([zeros, I_dim, -I_dim], axis=0)
    return wm, wc, xi.T
