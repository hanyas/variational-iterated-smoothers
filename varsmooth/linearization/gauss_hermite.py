from functools import partial
from typing import Tuple, List, Union

import numpy as np
import scipy as sc

import jax
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
    order: int = 5,
):
    get_sigma_points = \
        lambda m, chol_P: _get_sigma_points(m, chol_P, order)

    if isinstance(model, AdditiveGaussianModel):
        fun, noise = model
        return linearize_additive(fun, noise, q, get_sigma_points)
    elif isinstance(model, ConditionalMomentsModel):
        mean_fn, covar_fn = model
        return linearize_conditional(mean_fn, covar_fn, q, get_sigma_points)
    else:
        raise NotImplementedError


@partial(jax.jit, static_argnums=(2,))
def _get_sigma_points(
    m: jnp.ndarray,
    chol_P: jnp.ndarray,
    order: int
) -> SigmaPoints:

    nb_dim = m.shape[0]
    wm, wc, xi = _gauss_hermite_weights(nb_dim, order)
    sigma_points = m[None, :] + (chol_P @ xi).T
    return SigmaPoints(sigma_points, wm, wc)


def _gauss_hermite_weights(
    nb_dim: int,
    order: int = 3
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    n = nb_dim
    p = order

    hermite_coeff = _hermite_coeff(p)
    hermite_roots = np.flip(np.roots(hermite_coeff[-1]))

    table = np.zeros(shape=(n, p**n))

    w_1d = np.zeros(shape=(p,))
    for i in range(p):
        w_1d[i] = (
            2 ** (p - 1) * sc.special.factorial(p) * np.sqrt(np.pi)
            / (p**2 * (np.polyval(hermite_coeff[p - 1], hermite_roots[i])) ** 2)
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

    return jnp.asarray(wm), jnp.asarray(wm), jnp.asarray(xi)


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
