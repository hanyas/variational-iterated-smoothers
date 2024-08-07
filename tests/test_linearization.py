import pytest

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from varsmooth.objects import Gaussian
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel

from varsmooth.approximation import extended_linearization as extended
from varsmooth.approximation import unscented_linearization as unscented
from varsmooth.approximation import cubature_linearization as cubature
from varsmooth.approximation import gauss_hermite_linearization as gauss_hermite

LINEARIZATION_METHODS = [extended, unscented, cubature, gauss_hermite]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_debug_nans", False)


def linear_fn(x, A, b):
    return A @ x + b


def linear_conditional_mean(x, r, A, b, c):
    return A @ x + b + c @ r.mean


def linear_conditional_cov(x, r, A, b, c):
    return c @ r.cov @ c.T


def transition_mean(x):
    return jnp.log(44.7) + x - jnp.exp(x)


def transition_cov(x):
    return jnp.array([[0.3 ** 2]])


def observation_mean(x, gamma):
    return gamma * jnp.exp(x)


def observation_cov(x, gamma):
    return (gamma * jnp.exp(x)).reshape(1, 1)


# @pytest.mark.parametrize("dim_x", [1, 3, 5])
# @pytest.mark.parametrize("seed", [0, 13, 42])
# @pytest.mark.parametrize("method", LINEARIZATION_METHODS)
# def test_linear_additive(dim_x, seed, method):
#     np.random.seed(seed)
#     A = np.random.randn(dim_x, dim_x)
#     b = np.random.randn(dim_x)
#
#     m_x = np.random.randn(dim_x)
#     chol_x = np.random.rand(dim_x, dim_x)
#     chol_x[np.triu_indices(dim_x, 1)] = 0
#     noise = Gaussian(m_x, chol_x @ chol_x.T)
#
#     fun = partial(linear_fn, A=A, b=b)
#
#     m_q = np.random.randn(dim_x)
#     chol_q = np.random.rand(dim_x, dim_x)
#     chol_q[np.triu_indices(dim_x, 1)] = 0
#     ref = Gaussian(m_q, chol_q @ chol_q.T)
#
#     additive_model = AdditiveGaussianModel(fun, noise)
#     mat, offset, cov = method(additive_model, ref)
#
#     arg = np.random.randn(dim_x)
#     true_mean = fun(arg) + noise.mean
#     approx_mean = mat @ arg + offset
#
#     np.testing.assert_allclose(A, mat, atol=1e-7)
#     np.testing.assert_allclose(true_mean, approx_mean, atol=1e-7)
#     np.testing.assert_allclose(noise.cov, cov, atol=1e-7)


@pytest.mark.parametrize("dim_x", [1, 3, 5])
@pytest.mark.parametrize("dim_q", [1, 3])
@pytest.mark.parametrize("seed", [0, 42])
@pytest.mark.parametrize("method", LINEARIZATION_METHODS)
def test_linear_conditional(dim_x, dim_q, seed, method):
    np.random.seed(seed)
    A = np.random.randn(dim_x, dim_x)
    b = np.random.randn(dim_x)
    c = np.random.randn(dim_x, dim_q)

    m_q = np.random.randn(dim_q)
    chol_q = np.random.rand(dim_q, dim_q)
    chol_q[np.triu_indices(dim_q, 1)] = 0
    noise = Gaussian(m_q, chol_q @ chol_q.T)

    m_x = np.random.randn(dim_x)
    chol_x = np.random.rand(dim_x, dim_x)
    chol_x[np.triu_indices(dim_x, 1)] = 0
    ref = Gaussian(m_x, chol_x @ chol_x.T)

    E_f = partial(linear_conditional_mean, r=noise, A=A, b=b, c=c)
    V_f = partial(linear_conditional_cov, r=noise, A=A, b=b, c=c)

    moments_model = ConditionalMomentsModel(E_f, V_f)
    mat, offset, cov = method(moments_model, ref)

    arg = np.random.randn(dim_x)
    true_mean = linear_conditional_mean(arg, noise, A, b, c)
    approx_mean = mat @ arg + offset

    np.testing.assert_allclose(A, mat, atol=1e-3)
    np.testing.assert_allclose(true_mean, approx_mean, atol=1e-3)
    np.testing.assert_allclose((c @ chol_q) @ (c @ chol_q).T, cov, atol=1e-3)
