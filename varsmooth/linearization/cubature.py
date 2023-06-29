from typing import Tuple, Union

import jax.numpy as jnp
import numpy as np

from varsmooth.objects import StdGaussian, SqrtGaussian
from varsmooth.objects import (
    FunctionalModel,
    ConditionalMomentsModel,
)
from varsmooth.linearization.sigma_points import (
    SigmaPoints,
    linearize_functional,
    linearize_conditional,
)


def linearize(
    model: Union[FunctionalModel, ConditionalMomentsModel], x: StdGaussian
):
    if isinstance(model, FunctionalModel):
        f, q = model
        return linearize_functional(f, x, q, _get_sigma_points)
    mean_func, cov_func = model
    return linearize_conditional(
        mean_func,
        cov_func,
        x,
        _get_sigma_points,
    )


def _get_sigma_points(mvn: SqrtGaussian) -> Tuple[SigmaPoints, jnp.ndarray]:
    mean, chol = mvn
    n_dim = mean.shape[0]

    wm, wc, xi = _cubature_weights(n_dim)
    sigma_points = mean[None, :] + jnp.dot(chol, xi.T).T
    return SigmaPoints(sigma_points, wm, wc)


def _cubature_weights(
    n_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wm = np.ones(shape=(2 * n_dim,)) / (2 * n_dim)
    wc = wm
    xi = (
        np.concatenate([np.eye(n_dim), -np.eye(n_dim)], axis=0)
        * n_dim**0.5
    )
    return wm, wc, xi
