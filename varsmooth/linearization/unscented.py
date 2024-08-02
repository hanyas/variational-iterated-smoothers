from typing import Tuple, Optional, Union

import jax.numpy as jnp

from varsmooth.objects import Gaussian
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel

from varsmooth.linearization.sigma_points import SigmaPoints
from varsmooth.linearization.sigma_points import linearize_additive
from varsmooth.linearization.sigma_points import linearize_conditional


def linearize(
    model: Union[AdditiveGaussianModel, ConditionalMomentsModel],
    q: Gaussian,
    alpha: float = 1.0,
    beta: float = 0.0,
    kappa: float = None,
):
    get_sigma_points = \
        lambda m, chol_P: _get_sigma_points(m, chol_P, alpha, beta, kappa)

    if isinstance(model, AdditiveGaussianModel):
        fun, noise = model
        return linearize_additive(fun, noise, q, get_sigma_points)
    elif isinstance(model, ConditionalMomentsModel):
        mean_fn, cov_fn = model
        return linearize_conditional(mean_fn, cov_fn, q, get_sigma_points)
    else:
        raise NotImplementedError


def _get_sigma_points(
    m: jnp.ndarray,
    chol_P: jnp.ndarray,
    alpha: float,
    beta: float,
    kappa: Optional[float]
) -> SigmaPoints:

    nb_dim = m.shape[0]
    if kappa is None:
        kappa = 3.0 + nb_dim

    wm, wc, lamda = _unscented_weights(nb_dim, alpha, beta, kappa)
    scaled_chol = jnp.sqrt(nb_dim + lamda) * chol_P

    zeros = jnp.zeros((1, nb_dim))
    sigma_points = (
        m[None, :]
        + jnp.concatenate([zeros, scaled_chol.T, -scaled_chol.T], axis=0)
    )
    return SigmaPoints(sigma_points, wm, wc)


def _unscented_weights(
    nb_dim: int,
    alpha: float,
    beta: float,
    kappa: Optional[float]
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:

    lamda = alpha**2 * (nb_dim + kappa) - nb_dim
    wm = jnp.full(2 * nb_dim + 1, 1 / (2 * (nb_dim + lamda)))

    wm = wm.at[0].set(lamda / (nb_dim + lamda))
    wc = wm.at[0].set(lamda / (nb_dim + lamda) + (1 - alpha**2 + beta))
    return wm, wc, lamda
