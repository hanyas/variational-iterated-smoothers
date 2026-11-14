"""Sparse ESGVI checked against dense derivatives on the nonlinear robot."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from varsmooth.baselines import barfoot
from experiments.differential_drive import dd_common as dd
from varsmooth.baselines.gaussian import reverse_pairs

jax.config.update("jax_enable_x64", True)


def dense_precision(blocks):
    """Assemble a dense reference matrix from precision blocks.

    Args:
        blocks: BlockPrecision
            Block-tridiagonal precision to expand.

    Returns:
        matrix: array_like
            Dense joint precision matrix.
    """
    n, d, _ = blocks.diagonal.shape
    matrix = np.zeros((n * d, n * d))
    for k, diagonal in enumerate(blocks.diagonal):
        matrix[k*d:(k+1)*d, k*d:(k+1)*d] = diagonal
    for k, lower in enumerate(blocks.lower):
        matrix[(k+1)*d:(k+2)*d, k*d:(k+1)*d] = lower
        matrix[k*d:(k+1)*d, (k+1)*d:(k+2)*d] = lower.T
    return matrix


@pytest.fixture(scope="module")
def setup():
    """Create the shared robot model and inference objects.

    Returns:
        setup: tuple
            Model, initial posterior, and evaluator or update function.
    """
    Q, R, *_ = dd.make_parameters(**dd.PHYSICAL)
    prior = dd.make_prior()
    _, ys, controls = dd.simulate_data(prior, Q, R, 6, np.random.RandomState(4))
    model_fns = dd.make_model_fns(prior, Q, R)
    initial = dd.make_observation_init(prior, Q, R, controls, ys)
    return prior, R, ys, controls, model_fns, initial, barfoot.make_step(ys, controls, prior, R, model_fns[1])


def test_sparse_precision_solve_and_selected_inverse(setup):
    """Check sparse Gaussian algebra against dense matrix operations.

    Args:
        setup: tuple
            Shared model, initialization, and evaluator or update function.
    """
    _, _, _, _, _, initial, _ = setup
    precision = barfoot.precision_from_posterior(initial)
    marginal, pairs = reverse_pairs(initial)
    n, d = marginal.mean.shape
    # Independently form J from the reverse conditional residual map Bx-d.
    B, W = np.eye(n*d), np.zeros((n*d, n*d))
    for k in range(n-1):
        B[k*d:(k+1)*d, (k+1)*d:(k+2)*d] = -initial.kernels.F[k]
        covariance = np.asarray(initial.kernels.Sigma[k])
        W[k*d:(k+1)*d, k*d:(k+1)*d] = np.linalg.inv(0.5 * (covariance + covariance.T))
    covariance = np.asarray(initial.marginal.cov)
    W[-d:, -d:] = np.linalg.inv(0.5 * (covariance + covariance.T))
    J = B.T @ W @ B
    np.testing.assert_allclose(dense_precision(precision), J, atol=1e-11)
    inverse_diagonal, lower, valid = barfoot.factor_precision(precision)
    assert valid
    rhs = np.random.RandomState(9).normal(size=(n, d))
    solution = barfoot.solve_factored(inverse_diagonal, lower, jnp.asarray(rhs))
    np.testing.assert_allclose(solution.ravel(), np.linalg.solve(J, rhs.ravel()), rtol=1e-11, atol=1e-12)
    reconstructed = barfoot.posterior_from_factor(marginal.mean, inverse_diagonal, lower)
    selected = reverse_pairs(reconstructed)
    covariance = np.linalg.inv(J)
    for k in range(n):
        np.testing.assert_allclose(selected[0].cov[k], covariance[k*d:(k+1)*d, k*d:(k+1)*d], atol=1e-12)
    for k in range(n-1):
        np.testing.assert_allclose(selected[1].cov[k], covariance[k*d:(k+2)*d, k*d:(k+2)*d], atol=1e-12)
    np.testing.assert_allclose(selected[1].mean, pairs.mean, atol=1e-13)
    # Indefinite precision is reported, without an implicit eigenvalue repair.
    invalid = precision._replace(diagonal=precision.diagonal.at[0].set(-jnp.eye(d)))
    assert not barfoot.factor_precision(invalid)[2]


@pytest.mark.parametrize("rate", [1., 0.4])
def test_expected_hessian_and_fixed_rate_against_dense_equations(setup, rate):
    """Check the expected Hessian and fixed-rate update against dense equations.

    Args:
        setup: tuple
            Shared model, initialization, and evaluator or update function.
        rate: float
            Natural-gradient step size under test.
    """
    prior, R, ys, controls, model_fns, initial, step = setup
    marginal, pairs = reverse_pairs(initial)
    # Keep covariance fixed and differentiate the complete expected negative
    # log joint w.r.t. all means. This checks sparse factor scatter and Price's
    # covariance derivative against a different differentiation route.
    def expected_cost(flat_mean):
        """Evaluate the expected negative log joint at the supplied means.

        Args:
            flat_mean: array_like
                Flattened state means.

        Returns:
            cost: float
                Expected negative log joint.
        """
        mean = flat_mean.reshape(marginal.mean.shape)
        pair_mean = jnp.concatenate([mean[:-1], mean[1:]], axis=-1)
        return (-model_fns[0](mean[0], marginal.cov[0])
                - jnp.sum(jax.vmap(model_fns[1])(pair_mean, pairs.cov, controls))
                - jnp.sum(jax.vmap(model_fns[2])(mean[1:], marginal.cov[1:], ys)))

    g = jax.jit(jax.grad(expected_cost))(marginal.mean.ravel())
    H = jax.jit(jax.hessian(expected_cost))(marginal.mean.ravel())
    mean, gradient, blocks = barfoot.make_statistics(ys, controls, prior, R, model_fns[1])(initial)
    np.testing.assert_allclose(gradient.ravel(), g, atol=1e-10)
    np.testing.assert_allclose(dense_precision(blocks), H, atol=1e-10)
    precision = barfoot.precision_from_posterior(initial)
    candidate, updated_precision, valid = step(initial, precision, jnp.asarray(rate))
    assert valid
    target_J = (1-rate) * dense_precision(precision) + rate * H
    target_mean = mean.ravel() - rate * np.linalg.solve(H, g)
    target_covariance = np.linalg.inv(target_J)
    actual, actual_pairs = reverse_pairs(candidate)
    np.testing.assert_allclose(dense_precision(updated_precision), target_J, atol=1e-10)
    np.testing.assert_allclose(actual.mean.ravel(), target_mean, atol=1e-12)
    for k in range(len(controls)):
        np.testing.assert_allclose(actual_pairs.cov[k], target_covariance[k*5:(k+2)*5, k*5:(k+2)*5], atol=1e-12)
    if rate < 1:
        # Eq. 80 damps the direction computed using H, not the interpolated J.
        different_direction = mean.ravel() - rate * np.linalg.solve(target_J, g)
        assert np.max(np.abs(target_mean - different_direction)) > 1e-6
