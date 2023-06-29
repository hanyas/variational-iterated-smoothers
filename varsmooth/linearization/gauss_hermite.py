from functools import partial
from typing import Tuple, Union, List

import jax
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
    order: int = 3,
):
    get_sigma_points = lambda mvn: _get_sigma_points(mvn, order)
    if isinstance(model, FunctionalModel):
        f, q = model
        return linearize_functional(f, x, q, get_sigma_points)
    mean_func, cov_func = model
    return linearize_conditional(mean_func, cov_func, x, get_sigma_points)


@partial(jax.jit, static_argnums=(1,))
def _get_sigma_points(
    mvn: SqrtGaussian, order: int
) -> Tuple[SigmaPoints, np.ndarray]:
    mean, chol = mvn
    n_dim = mean.shape[0]
    wm, wc, xi = _gauss_hermite_weights(n_dim, order)
    sigma_points = mean[None, :] + (chol @ xi).T

    return SigmaPoints(sigma_points, wm, wc)


def _gauss_hermite_weights(
    n_dim: int, order: int = 3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = n_dim
    p = order

    hermite_coeff = _hermite_coeff(p)
    hermite_roots = np.flip(np.roots(hermite_coeff[-1]))

    table = np.zeros(shape=(n, p**n))

    w_1d = np.zeros(shape=(p,))
    for i in range(p):
        w_1d[i] = (
            2 ** (p - 1)
            * np.math.factorial(p)
            * np.sqrt(np.pi)
            / (
                p**2
                * (np.polyval(hermite_coeff[p - 1], hermite_roots[i])) ** 2
            )
        )

    # Get roll table
    for i in range(n):
        base = np.ones(shape=(1, p ** (n - i - 1)))
        for j in range(1, p):
            base = np.concatenate(
                [base, (j + 1) * np.ones(shape=(1, p ** (n - i - 1)))], axis=1
            )
        table[n - i - 1, :] = np.tile(base, (1, int(p**i)))

    table = table.astype("int64") - 1

    s = 1 / (np.sqrt(np.pi) ** n)

    wm = s * np.prod(w_1d[table], axis=0)
    xi = np.sqrt(2) * hermite_roots[table]

    return wm, wm, xi


def _hermite_coeff(order: int) -> List:
    H0 = np.array([1])
    H1 = np.array([2, 0])

    H = [H0, H1]

    for i in range(2, order + 1):
        H.append(
            2 * np.append(H[i - 1], 0)
            - 2 * (i - 1) * np.pad(H[i - 2], (2, 0), "constant", constant_values=0)
        )

    return H
