from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from varsmooth.objects import StdGaussian, FunctionalModel, ConditionalMomentsModel
from varsmooth.utils import tria
from varsmooth.linearization import extended, cubature, gauss_hermite, unscented

LINEARIZATION_METHODS = [extended, cubature, gauss_hermite, unscented]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", False)


def linear_function(x, a, c):
    return a @ x + c


def linear_conditional_mean(x, q, a, b, c):
    return a @ x + b @ q + c


def linear_conditional_cov(_x, b, cov_q):
    return b @ cov_q @ b.T


def linear_conditional_chol(_x, b, chol_q):
    ny, nq = b.shape
    if ny > nq:
        res = jnp.concatenate([b @ chol_q,
                               jnp.zeros((ny, ny - nq))], axis=1)
    else:
        res = tria(b @ chol_q)
    return res


def transition_mean(x):
    return jnp.log(44.7) + x - jnp.exp(x)


def transition_cov(_x):
    return jnp.array([[0.3 ** 2]])


def transition_chol(_x):
    return jnp.array([[jnp.sqrt(0.3 ** 2)]])


def observation_mean(x, lam):
    return lam * jnp.exp(x)


def observation_cov(x, lam):
    return (lam * jnp.exp(x)).reshape(1, 1)


def observation_chol(x, lam):
    return (jnp.sqrt(lam * jnp.exp(x))).reshape(1, 1)


@pytest.mark.parametrize("nx", [1, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
def test_linear_functional(nx, seed, method):
    np.random.seed(seed)
    a = np.random.randn(nx, nx)
    c = np.random.randn(nx)

    m_x = np.random.randn(nx)
    m_q = np.random.randn(nx)

    chol_x = np.random.rand(nx, nx)
    chol_x[np.triu_indices(nx, 1)] = 0

    chol_q = np.random.rand(nx, nx)
    chol_q[np.triu_indices(nx, 1)] = 0

    x = StdGaussian(m_x, chol_x @ chol_x.T)
    q = StdGaussian(m_q, chol_q @ chol_q.T)

    fun = partial(linear_function, a=a, c=c)

    fun_model = FunctionalModel(fun, q)
    F_x, remainder, Q_lin = method(fun_model, x)
    x_prime = np.random.randn(nx)

    expected = fun(x_prime) + m_q
    actual = F_x @ x_prime + remainder
    expected_Q = chol_q @ chol_q.T

    np.testing.assert_allclose(a, F_x, atol=1e-7)
    np.testing.assert_allclose(expected, actual, atol=1e-7)
    np.testing.assert_allclose(expected_Q, Q_lin, atol=1e-7)


@pytest.mark.parametrize("nx", [1, 3])
@pytest.mark.parametrize("nq", [1, 2, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
def test_linear_conditional(nx, nq, seed, method):
    np.random.seed(seed)
    a = np.random.randn(nx, nx)
    b = np.random.randn(nx, nq)
    c = np.random.randn(nx)

    m_x = np.random.randn(nx)
    m_q = np.random.randn(nq)

    chol_x = np.random.rand(nx, nx)
    chol_x[np.triu_indices(nx, 1)] = 0

    chol_q = np.random.rand(nq, nq)
    chol_q[np.triu_indices(nq, 1)] = 0

    E_f = partial(linear_conditional_mean, q=m_q, a=a, b=b, c=c)
    V_f = partial(linear_conditional_cov, b=b, cov_q=chol_q @ chol_q.T)
    chol_f = partial(linear_conditional_chol, b=b, chol_q=chol_q)

    moments_model = ConditionalMomentsModel(E_f, V_f)
    x = StdGaussian(m_x, chol_x @ chol_x.T)

    F_x, remainder, Q_lin = method(moments_model, x)
    x_prime = np.random.randn(nx)

    expected = linear_conditional_mean(x_prime, m_q, a, b, c)
    actual = F_x @ x_prime + remainder
    expected_Q = (b @ chol_q) @ (b @ chol_q).T
    np.testing.assert_allclose(a, F_x, atol=1e-3)
    np.testing.assert_allclose(expected, actual, atol=1e-7)
    np.testing.assert_allclose(expected_Q, Q_lin, atol=1e-7)
