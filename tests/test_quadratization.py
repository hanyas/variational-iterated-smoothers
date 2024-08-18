import pytest

import jax
import jax.numpy as jnp

import numpy as np
import scipy as sc

from varsmooth.utils import logdet
from varsmooth.objects import Gaussian
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.approximation import gauss_hermite_quadratization as gauss_hermite

from tests.test_utils import generate_system

QUADRATIZATION_METHODS = [gauss_hermite]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", False)


def quadratic_fn(x, A, b, c):
    return - 0.5 * x.T @ A @ x + jnp.dot(x, b) + c


@pytest.mark.parametrize("dim_x", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [27, 33, 37])
@pytest.mark.parametrize("method", QUADRATIZATION_METHODS)
def test_quadratic(dim_x, seed, method):

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

    np.testing.assert_allclose(A, A_approx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(b, b_approx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(c, c_approx, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dim_x", [1, 4, 5, 6])
@pytest.mark.parametrize("dim_y", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [27, 33, 37])
@pytest.mark.parametrize("method", QUADRATIZATION_METHODS)
def test_log_observation_additive(dim_y, dim_x, seed, method):

    np.random.seed(seed)
    _, H, e, Delta, _ = generate_system(dim_x, dim_y)
    h = AdditiveGaussianModel(
        lambda x: H @ x + e,
        Gaussian(np.zeros((dim_y, )), Delta)
    )

    y = np.random.randn(dim_y)
    logpdf = lambda x: h.logpdf(y, x)

    m_x = np.random.randn(dim_x)
    chol_x = np.random.rand(dim_x, dim_x)
    chol_x[np.triu_indices(dim_x, 1)] = 0
    q = Gaussian(m_x, chol_x @ chol_x.T)

    L_approx, l_approx, nu_approx = method(logpdf, q)

    L = H.T @ sc.linalg.solve(Delta, H)
    l = H.T @ sc.linalg.solve(Delta, y - e)
    nu = (
        - 0.5 * logdet(2 * jnp.pi * Delta)
        - 0.5 * (y - e).T @ sc.linalg.solve(Delta, y - e)
    )

    np.testing.assert_allclose(L, L_approx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(l, l_approx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(nu, nu_approx, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dim_x", [1, 2, 3, 4])
@pytest.mark.parametrize("seed", [27, 33, 37])
@pytest.mark.parametrize("method", QUADRATIZATION_METHODS)
def test_log_transition_additive(dim_x, seed, method):

    np.random.seed(seed)
    _, A, b, Omega, _ = generate_system(dim_x, dim_x)
    f = AdditiveGaussianModel(
        lambda x: A @ x + b,
        Gaussian(np.zeros((dim_x, )), Omega)
    )

    m_x = np.random.randn(int(2 * dim_x))
    chol_x = np.random.rand(int(2 * dim_x), int(2 * dim_x))
    chol_x[np.triu_indices(int(2 * dim_x), 1)] = 0
    q = Gaussian(m_x, chol_x @ chol_x.T)

    logpdf = lambda x: f.logpdf(x[:dim_x], x[dim_x:])
    C_approx, c_approx, kappa_approx = method(logpdf, q)

    C11_approx = C_approx[:dim_x, :dim_x]
    C12_approx = -C_approx[:dim_x, dim_x:]
    C21_approx = -C_approx[dim_x:, :dim_x]
    C22_approx = C_approx[dim_x:, dim_x:]
    c1_approx = c_approx[:dim_x]
    c2_approx = c_approx[dim_x:]

    C11 = sc.linalg.inv(Omega)
    C12 = sc.linalg.solve(Omega, A)
    C21 = sc.linalg.solve(Omega, A).T
    C22 = A.T @ sc.linalg.solve(Omega, A)
    c1 = sc.linalg.solve(Omega, b)
    c2 = -A.T @ sc.linalg.solve(Omega, b)
    kappa = (
        - 0.5 * logdet(2 * jnp.pi * Omega)
        - 0.5 * b.T @ sc.linalg.solve(Omega, b)
    )

    np.testing.assert_allclose(C11, C11_approx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(C12, C12_approx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(C22, C22_approx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(c1, c1_approx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(c2, c2_approx, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(kappa, kappa_approx, rtol=1e-4, atol=1e-4)
