import pytest

import jax
import jax.numpy as jnp
import numpy as np

from varsmooth.objects import Gaussian
from varsmooth.approximation import gauss_hermite_quadratzation as gauss_hermite

QUADRATIZATION_METHODS = [gauss_hermite]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", False)


def quadratic_fn(x, A, b, c):
    return - 0.5 * x.T @ A @ x + jnp.dot(x, b) + c


@pytest.mark.parametrize("dim_x", [1, 3, 5])
@pytest.mark.parametrize("seed", [27, 33, 37])
@pytest.mark.parametrize("method", QUADRATIZATION_METHODS)
def test_linear_additive(dim_x, seed, method):

    np.random.seed(seed)
    A = np.random.randn(dim_x, dim_x)
    A = 0.5 * (A + A.T)

    b = np.random.randn(dim_x)
    c = np.random.randn()

    m_x = np.random.randn(dim_x)
    chol_x = np.random.rand(dim_x, dim_x)
    chol_x[np.triu_indices(dim_x, 1)] = 0
    q = Gaussian(m_x, chol_x @ chol_x.T)

    fn = lambda x: quadratic_fn(x, A, b, c)
    A_approx, b_approx, c_approx = method(fn, q)

    np.testing.assert_allclose(A, A_approx, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(b, b_approx, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(c, c_approx, rtol=1e-3, atol=1e-3)
