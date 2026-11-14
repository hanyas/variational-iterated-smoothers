"""Nonlinear checks for statistical linearization and the LM-IPLS update."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from varsmooth.approximation import cubature_linearization
from varsmooth.baselines import lm_ipls, lm_ipls_smoother
from varsmooth.objects import AdditiveGaussianModel, AffineGaussian, Gaussian, GaussMarkov
from varsmooth.smoothers.utils import std_forward_message

jax.config.update("jax_enable_x64", True)


def independent_init(mean, cov):
    """Build a forward Gaussian chain with independent states.

    Args:
        mean: array_like
            State means of shape (T + 1, d).
        cov: array_like
            Marginal covariances of shape (T + 1, d, d).

    Returns:
        posterior: GaussMarkov
            Forward Gaussian Markov posterior.
    """
    T, dx = len(mean) - 1, mean.shape[1]
    return GaussMarkov(Gaussian(mean[0], cov[0]), AffineGaussian(jnp.zeros((T, dx, dx)), mean[1:], cov[1:]))


def cubature_reference(model, mean, cov):
    """Compute statistical regression using NumPy cubature points.

    Args:
        model: AdditiveGaussianModel
            Nonlinear function and additive Gaussian noise.
        mean: array_like
            Gaussian mean of shape (d,).
        cov: array_like
            Gaussian covariance of shape (d, d).

    Returns:
        F: array_like
            Regression matrix.
        d: array_like
            Affine offset including the noise mean.
        Sigma: array_like
            Regression residual covariance including observation noise.
    """
    offsets = np.sqrt(len(mean)) * np.linalg.cholesky(cov).T
    points = np.concatenate((mean + offsets, mean - offsets))
    values = np.array([model.fun(x) for x in points])
    average = values.mean(axis=0)
    centered = values - average
    cross = (points - mean).T @ centered / len(points)
    F = np.linalg.solve(cov, cross).T
    residual_cov = centered.T @ centered / len(points) - F @ cov @ F.T
    return F, average - F @ mean + model.noise.mean, residual_cov + model.noise.cov


@pytest.mark.parametrize("damping", [0.0, 0.5, 100.0])
def test_rts_step_matches_dense_statistical_quadratic(damping, monkeypatch):
    """Check the RTS step against a dense statistical quadratic.

    Args:
        damping: float
            Nonnegative Levenberg-Marquardt regularization.
        monkeypatch: pytest.MonkeyPatch
            Fixture for replacing functions during the test.
    """
    prior = Gaussian(jnp.array([0.2, -0.1]), jnp.array([[0.3, 0.02], [0.02, 0.2]]))
    transition = AdditiveGaussianModel(
        lambda x: jnp.array([0.8 * x[0] + 0.1 * x[1] ** 2, 0.7 * x[1] + 0.05 * x[0] ** 2]),
        Gaussian(jnp.array([0.02, -0.01]), jnp.array([[0.07, 0.01], [0.01, 0.09]])),
    )
    likelihood = AdditiveGaussianModel(
        lambda x: jnp.array([x[0] + 0.2 * x[1] ** 3, jnp.tanh(x[1]) + 0.1 * x[0] ** 2]),
        Gaussian(jnp.array([-0.03, 0.01]), jnp.array([[0.13, -0.02], [-0.02, 0.12]])),
    )
    mean = jnp.array([[0.2, -0.1], [0.1, -0.08], [0.3, 0.1], [0.25, 0.04]])
    cov = jnp.array([prior.cov * (1 + 0.1 * k) for k in range(len(mean))])
    observations = jnp.array([[0.25, -0.1], [0.4, 0.2], [0.15, 0.1]])
    moments = Gaussian(mean, cov)
    calls = []
    original_filter, original_smoother = lm_ipls.filtering, lm_ipls.smoothing

    def filtering(*args):
        """Record and forward a Kalman filtering call.

        Args:
            *args: tuple
                Positional arguments forwarded to the wrapped function.

        Returns:
            filtered: Gaussian
                Batched filtered state marginals.
        """
        calls.append("filtering")
        return original_filter(*args)

    def smoothing(*args):
        """Record and forward an RTS smoothing call.

        Args:
            *args: tuple
                Positional arguments forwarded to the wrapped function.

        Returns:
            posterior: GaussMarkov
                Forward Gaussian Markov posterior.
        """
        calls.append("smoothing")
        return original_smoother(*args)

    monkeypatch.setattr(lm_ipls, "filtering", filtering)
    monkeypatch.setattr(lm_ipls, "smoothing", smoothing)
    linearize, _, _, step = lm_ipls._make_functions(
        observations, prior, transition, likelihood, cubature_linearization
    )
    lin_t, lin_o = linearize(moments)
    posterior = step(mean, lin_t, lin_o, jnp.asarray(damping))
    # JIT traces the regularized and unregularized branches; both reuse core RTS.
    assert calls == ["filtering", "smoothing", "filtering", "smoothing"]

    precision = damping * np.eye(mean.size)
    information = damping * np.array(mean).ravel()
    prior_precision = np.linalg.inv(prior.cov)
    precision[:2, :2] += prior_precision
    information[:2] += prior_precision @ prior.mean
    for k in range(len(observations)):
        F, d, Q = cubature_reference(transition, np.array(mean[k]), np.array(cov[k]))
        H, c, R = cubature_reference(likelihood, np.array(mean[k + 1]), np.array(cov[k + 1]))
        A = np.zeros((2, mean.size))
        A[:, 2 * k : 2 * k + 2], A[:, 2 * k + 2 : 2 * k + 4] = -F, np.eye(2)
        B = np.zeros_like(A)
        B[:, 2 * k + 2 : 2 * k + 4] = H
        precision += A.T @ np.linalg.solve(Q, A) + B.T @ np.linalg.solve(R, B)
        information += A.T @ np.linalg.solve(Q, d) + B.T @ np.linalg.solve(R, observations[k] - c)
    expected_cov = np.linalg.inv(precision)
    expected_mean = np.linalg.solve(precision, information)
    actual = std_forward_message(posterior)
    np.testing.assert_allclose(actual.mean.ravel(), expected_mean, atol=1e-12)
    for k in range(len(mean)):
        np.testing.assert_allclose(
            actual.cov[k], expected_cov[2 * k : 2 * k + 2, 2 * k : 2 * k + 2], atol=1e-12
        )
        if k < len(observations):
            adjacent = actual.cov[k] @ posterior.kernels.F[k].T
            np.testing.assert_allclose(
                adjacent, expected_cov[2 * k : 2 * k + 2, 2 * k + 2 : 2 * k + 4], atol=1e-12
            )


def test_frozen_cost_uses_gaussian_expectations_and_old_weights():
    """Check Gaussian expectations and fixed weights in the IPLS cost."""
    prior = Gaussian(jnp.array([0.4]), jnp.array([[0.36]]))
    transition = AdditiveGaussianModel(
        lambda x: 0.8 * x + 0.2 * x**2, Gaussian(jnp.array([0.02]), jnp.array([[0.1]]))
    )
    likelihood = AdditiveGaussianModel(lambda x: x**3, Gaussian(jnp.array([-0.03]), jnp.array([[0.7]])))
    old = Gaussian(jnp.array([[0.4], [0.1], [-0.3]]), jnp.array([[[0.36]], [[0.2]], [[0.15]]]))
    mean = jnp.array([[0.5], [0.3], [-0.1]])
    observations = jnp.array([[0.8], [-0.4]])
    linearize, cost, reduction, _ = lm_ipls._make_functions(
        observations, prior, transition, likelihood, cubature_linearization
    )
    lin_t, lin_o = linearize(old)

    def reference(m):
        """Evaluate the scalar reference IPLS cost with frozen weights.

        Args:
            m: array_like
                Scalar state trajectory of shape (T + 1, 1).

        Returns:
            cost: float
                Frozen-covariance IPLS cost.
        """
        P = old.cov[:, 0, 0]
        fbar = 0.8 * m[:-1] + 0.2 * (m[:-1] ** 2 + P[:-1]) + 0.02
        hbar = m[1:] ** 3 + 3 * m[1:] * P[1:] - 0.03
        return 0.5 * (
            (m[0] - 0.4) ** 2 / 0.36
            + jnp.sum((m[1:] - fbar) ** 2 / lin_t.Sigma[:, 0, 0])
            + jnp.sum((observations[:, 0] - hbar) ** 2 / lin_o.Sigma[:, 0, 0])
        )

    np.testing.assert_allclose(cost(mean, old.cov, lin_t, lin_o), reference(mean[:, 0]), atol=1e-12)
    np.testing.assert_allclose(
        reduction(old.mean, mean, old.cov, lin_t, lin_o),
        reference(old.mean[:, 0]) - reference(mean[:, 0]),
        atol=1e-12,
    )
    # Cubature SLR for x^3 depends on variance and differs from a Taylor Jacobian.
    np.testing.assert_allclose(lin_o.F[:, 0, 0], 3 * old.mean[1:, 0] ** 2 + old.cov[1:, 0, 0])
    assert abs(float(cost(mean, old.cov * 0.5, lin_t, lin_o) - reference(mean[:, 0]))) > 0.01


def test_accept_reject_keeps_frozen_covariances_and_returns_native_posterior():
    """Check frozen-covariance acceptance and the returned LM-IPLS posterior."""
    prior = Gaussian(jnp.array([0.4]), jnp.array([[0.36]]))
    transition = AdditiveGaussianModel(
        lambda x: 0.95 * x + 0.02, Gaussian(jnp.zeros(1), jnp.array([[0.0351]]))
    )
    likelihood = AdditiveGaussianModel(lambda x: x**3, Gaussian(jnp.zeros(1), jnp.ones((1, 1))))
    observations = jnp.array([[20.0]])
    init = independent_init(jnp.full((2, 1), 0.4), jnp.full((2, 1, 1), 0.36))
    callbacks = []
    _, history = lm_ipls_smoother(
        observations,
        prior,
        transition,
        likelihood,
        init,
        max_iterations=2,
        callback=lambda record, q: callbacks.append((record, q)),
    )
    assert history[-1]["iterations"] == 2 and len(callbacks) == 3
    assert history[1]["trials"] > 1
    assert history[-1]["total_trials"] == sum(row["trials"] for row in history)
    linearize, cost, _, step = lm_ipls._make_functions(
        observations, prior, transition, likelihood, cubature_linearization
    )
    for (previous, old_q), (current, q) in zip(callbacks[:-1], callbacks[1:]):
        moments = std_forward_message(old_q)
        lin_t, lin_o = linearize(moments)
        damping = previous["next_damping"]
        for _ in range(current["trials"] - 1):
            rejected = std_forward_message(step(moments.mean, lin_t, lin_o, jnp.asarray(damping)))
            assert float(cost(rejected.mean, moments.cov, lin_t, lin_o)) >= current["cost_before"]
            damping *= 10.0
        np.testing.assert_allclose(current["damping"], damping)
        np.testing.assert_allclose(current["next_damping"], damping / 10.0)
        assert current["cost_after"] < current["cost_before"]
        native = step(moments.mean, lin_t, lin_o, jnp.asarray(damping))
        for actual, expected in zip(jax.tree.leaves(q), jax.tree.leaves(native)):
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    failed, history = lm_ipls_smoother(
        observations, prior, transition, likelihood, init, max_iterations=2, max_trials=1
    )
    assert history[-1]["iterations"] == 0 and not history[-1]["converged"]
    assert "no LM trial" in history[-1]["message"]
    for actual, expected in zip(jax.tree.leaves(failed), jax.tree.leaves(init)):
        np.testing.assert_array_equal(actual, expected)

    tiny, history = lm_ipls_smoother(
        observations,
        prior,
        transition,
        likelihood,
        init,
        max_iterations=1,
        initial_damping=1e8,
        stop_on_convergence=True,
    )
    assert history[-1]["iterations"] == 1
    assert np.max(np.abs(std_forward_message(tiny).mean - 0.4)) < 1e-6
    assert not history[-1]["converged"] and history[-1]["fixed_point_error"] > 1e-3
