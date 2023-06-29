from typing import Tuple, Union, Optional

import jax.numpy as jnp
import numpy as np

from varsmooth.objects import (
    StdGaussian,
    SqrtGaussian,
    FunctionalModel,
    ConditionalMomentsModel,
)
from varsmooth.linearization.sigma_points import (
    SigmaPoints,
    linearize_functional,
    linearize_conditional,
)


def linearize(
    model: Union[FunctionalModel, ConditionalMomentsModel],
    x: StdGaussian,
    alpha: float = 1.0,
    beta: float = 0.0,
    kappa: float = None,
):
    get_sigma_points = lambda mvn: _get_sigma_points(mvn, alpha, beta, kappa)
    if isinstance(model, FunctionalModel):
        f, q = model
        return linearize_functional(f, x, q, get_sigma_points)
    mean_func, cov_func = model
    return linearize_conditional(
        mean_func,
        cov_func,
        x,
        get_sigma_points,
    )


def _get_sigma_points(
    mvn: SqrtGaussian, alpha: float, beta: float, kappa: Optional[float]
) -> Tuple[SigmaPoints, jnp.ndarray]:
    mean, chol = mvn
    n_dim = mean.shape[0]

    if kappa is None:
        kappa = 3.0 + n_dim
    wm, wc, lamda = _unscented_weights(n_dim, alpha, beta, kappa)
    scaled_chol = jnp.sqrt(n_dim + lamda) * chol

    zeros = jnp.zeros((1, n_dim))
    sigma_points = mean[None, :] + jnp.concatenate(
        [zeros, scaled_chol.T, -scaled_chol.T], axis=0
    )
    return SigmaPoints(sigma_points, wm, wc)


def _unscented_weights(
    n_dim: int, alpha: float, beta: float, kappa: Optional[float]
) -> Tuple[np.ndarray, np.ndarray, float]:
    lamda = alpha**2 * (n_dim + kappa) - n_dim
    wm = jnp.full(2 * n_dim + 1, 1 / (2 * (n_dim + lamda)))

    wm = wm.at[0].set(lamda / (n_dim + lamda))
    wc = wm.at[0].set(lamda / (n_dim + lamda) + (1 - alpha**2 + beta))
    return wm, wc, lamda
