"""Nonlinear checks for the recursive Kalman/RTS Newton baseline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from varsmooth.baselines import newton_smoother
from varsmooth.baselines import newton
from varsmooth.objects import AdditiveGaussianModel, AffineGaussian, Gaussian, GaussMarkov
from varsmooth.smoothers.utils import std_forward_message

jax.config.update("jax_enable_x64", True)


def independent_init(mean):
    """Build a forward Gaussian chain with independent states.

    Args:
        mean: array_like
            State means of shape (T + 1, d).

    Returns:
        posterior: GaussMarkov
            Forward Gaussian Markov posterior.
    """
    T, dx = len(mean) - 1, mean.shape[1]
    return GaussMarkov(
        Gaussian(mean[0], jnp.eye(dx)),
        AffineGaussian(jnp.zeros((T, dx, dx)), mean[1:],
                       jnp.broadcast_to(jnp.eye(dx), (T, dx, dx))),
    )


@pytest.fixture(scope="module")
def nonlinear_model():
    """Construct a nonlinear test model with a dense reference objective.

    Returns:
        observations: array_like
            Observation sequence with one missing row.
        prior: Gaussian
            Initial-state prior.
        transition: AdditiveGaussianModel
            Controlled nonlinear transition.
        likelihood: AdditiveGaussianModel
            Nonlinear observation model.
        initial: GaussMarkov
            Independent forward Gaussian initialization.
        options: dict
            Known transition and observation inputs.
        reference: callable
            Dense negative-log-joint evaluator.
    """
    T = 4
    prior = Gaussian(jnp.array([.2, -.1]), jnp.array([[.3, .02], [.02, .2]]))
    controls = jnp.linspace(-.1, .2, T)[:, None]
    observation_inputs = jnp.arange(T)[:, None] * .01
    f = lambda x, u: jnp.array([.8*x[0] + .1*jnp.sin(x[1]) + u[0], .7*x[1] + .05*x[0]**2])
    h = lambda x, v: jnp.array([x[0] + .2*x[1]**3 + v[0], jnp.tanh(x[1]) + .1*x[0]**2])
    Q = jnp.array([np.array([[.07, .01], [.01, .09]]) * (1 + .1*k) for k in range(T)])
    R = jnp.array([np.array([[.13, -.02], [-.02, .12]]) * (1 + .2*k) for k in range(T)])
    qm, rm = jnp.array([.02, -.01]), jnp.array([-.03, .01])
    transition = AdditiveGaussianModel(f, Gaussian(qm, Q))
    likelihood = AdditiveGaussianModel(h, Gaussian(rm, R))
    mean = jnp.array([[.2, -.1], [.1, -.08], [.2, -.05], [.3, .1], [.25, .04]])
    observations = jax.vmap(h)(mean[1:], observation_inputs) + rm + .015
    mask = np.array([True, False, True, True])
    observations = observations.at[1].set(jnp.nan)

    def reference(flat):
        """Evaluate the dense reference negative log joint.

        Args:
            flat: array_like
                Flattened state trajectory.

        Returns:
            cost: float
                Negative log-joint density including normalization constants.
        """
        x = flat.reshape(T+1, 2)
        result = -prior.log_prob(x[0])
        for k in range(T):
            result -= Gaussian(f(x[k], controls[k]) + qm, Q[k]).log_prob(x[k+1])
            if mask[k]:
                result -= Gaussian(h(x[k+1], observation_inputs[k]) + rm, R[k]).log_prob(observations[k])
        return result

    options = dict(transition_inputs=controls, observation_inputs=observation_inputs)
    init = independent_init(mean)
    return observations, prior, transition, likelihood, init, options, reference


@pytest.mark.parametrize("damping", [.5, 100.])
@pytest.mark.parametrize("gauss_newton", [False, True])
def test_recursive_step_matches_nonlinear_quadratic(nonlinear_model, damping, gauss_newton, monkeypatch):
    """Check recursive Newton steps against the dense nonlinear quadratic.

    Args:
        nonlinear_model: tuple
            Nonlinear model, observations, initialization, and reference objective.
        damping: float
            Nonnegative Levenberg-Marquardt regularization.
        gauss_newton: bool
            Omit residual-weighted second derivatives when True.
        monkeypatch: pytest.MonkeyPatch
            Fixture for replacing functions during the test.
    """
    observations, prior, transition, likelihood, init, options, reference = nonlinear_model
    mean = std_forward_message(init).mean
    calls = []
    original_filter, original_smoother = newton.filtering, newton.smoothing

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

    monkeypatch.setattr(newton, "filtering", filtering)
    monkeypatch.setattr(newton, "smoothing", smoothing)
    _, _, statistics, step = newton._make_functions(
        observations, prior, transition, likelihood, mean, gauss_newton=gauss_newton, **options
    )
    gradient = jax.grad(reference)(mean.ravel())
    hessian = jax.hessian(reference)(mean.ravel())
    if gauss_newton:
        def whitened_residuals(flat):
            """Concatenate the noise-whitened residuals for the dense reference.

            Args:
                flat: array_like
                    Flattened state trajectory.

            Returns:
                residuals: array_like
                    Flattened whitened prior, transition, and retained observation residuals.
            """
            x = flat.reshape(mean.shape)
            residuals = [jnp.linalg.solve(jnp.linalg.cholesky(prior.cov), x[0] - prior.mean)]
            for k in range(len(observations)):
                rt = x[k+1] - transition.fun(x[k], options["transition_inputs"][k]) - transition.noise.mean
                residuals.append(jnp.linalg.solve(jnp.linalg.cholesky(transition.noise.cov[k]), rt))
                if np.all(np.isfinite(observations[k])):
                    ry = (likelihood.fun(x[k+1], options["observation_inputs"][k])
                          + likelihood.noise.mean - observations[k])
                    residuals.append(jnp.linalg.solve(jnp.linalg.cholesky(likelihood.noise.cov[k]), ry))
            return jnp.concatenate(residuals)

        jacobian = jax.jacfwd(whitened_residuals)(mean.ravel())
        gn_hessian = jacobian.T @ jacobian
        assert np.max(np.abs(gn_hessian - hessian)) > 1e-3
        hessian = gn_hessian
    np.testing.assert_allclose(statistics(mean)[1].ravel(), gradient, atol=1e-12)
    q, updated, predicted = step(mean, jnp.asarray(damping))
    assert calls == ["filtering", "smoothing"]
    precision = hessian + damping*np.eye(mean.size)
    covariance = np.linalg.inv(precision)
    direction = np.linalg.solve(precision, -gradient)
    np.testing.assert_allclose(updated.ravel(), mean.ravel() + direction, atol=1e-12)
    np.testing.assert_allclose(predicted, -.5*gradient @ direction, atol=1e-12)
    marginal = std_forward_message(q)
    for k in range(len(mean)):
        np.testing.assert_allclose(marginal.cov[k], covariance[2*k:2*k+2, 2*k:2*k+2], atol=1e-12)
        if k < len(mean)-1:
            adjacent = marginal.cov[k] @ q.kernels.F[k].T
            np.testing.assert_allclose(adjacent, covariance[2*k:2*k+2, 2*k+2:2*k+4], atol=1e-12)
    # The returned covariance belongs to this regularized quadratic, not a
    # fresh undamped Hessian evaluated at the updated mean.
    fresh = np.linalg.inv(jax.hessian(reference)(updated.ravel()))
    assert np.max(np.abs(covariance - fresh)) > 1e-3


def test_fixed_budget_retains_native_posterior_and_lm_rule(nonlinear_model):
    """Check fixed-budget execution, native covariances, and LM damping updates.

    Args:
        nonlinear_model: tuple
            Nonlinear model, observations, initialization, and reference objective.
    """
    observations, prior, transition, likelihood, init, options, reference = nonlinear_model
    callbacks = []
    scale = jnp.array([.3, 2.])
    fit = newton_smoother(
        observations, prior, transition, likelihood, init, max_iterations=80,
        state_scale=scale, gradient_tolerance=1e-5,
        callback=lambda record, q: callbacks.append((record.copy(), q)), **options
    )
    assert fit.iterations == 80 and len(fit.history) == len(callbacks) == 81
    assert fit.converged and fit.posterior is not None
    assert sum(row["accepted"] for row in fit.history) == fit.accepted_updates
    assert np.max(np.diff([row["map_cost"] for row in fit.history])) <= 0
    gradient = jax.grad(reference)(fit.mean.ravel()).reshape(fit.mean.shape)
    np.testing.assert_allclose(fit.stationarity, np.max(np.abs(gradient*scale)), atol=1e-12)
    for a, b in zip(jax.tree.leaves(callbacks[0][1]), jax.tree.leaves(init)):
        np.testing.assert_array_equal(a, b)
    nu = 2.
    for (previous, old_q), (current, q) in zip(callbacks[:-1], callbacks[1:]):
        damping = current["damping"]
        if current["accepted"]:
            rho = current["reduction_ratio"]
            expected = max(1e-16, damping * max(1/3, 1-(2*rho-1)**3))
            nu = 2.
        else:
            expected, nu = min(1e16, damping) * nu, 2*nu
            for a, b in zip(jax.tree.leaves(q), jax.tree.leaves(old_q)):
                np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(current["next_damping"], expected)
    short = newton_smoother(
        observations, prior, transition, likelihood, init, max_iterations=80,
        stop_on_convergence=True, state_scale=scale, gradient_tolerance=1e-5, **options
    )
    assert short.converged and short.iterations < 80
    assert not any(row["converged"] for row in short.history[:-1])


def test_rejections_preserve_initial_distribution(nonlinear_model, monkeypatch):
    """Check that rejected updates preserve the complete initial posterior.

    Args:
        nonlinear_model: tuple
            Nonlinear model, observations, initialization, and reference objective.
        monkeypatch: pytest.MonkeyPatch
            Fixture for replacing functions during the test.
    """
    observations, prior, transition, likelihood, init, options, _ = nonlinear_model
    original = newton._make_functions

    def functions(*args, **kwargs):
        """Build Newton functions with a deliberately rejected update.

        Args:
            *args: tuple
                Positional arguments forwarded to the wrapped function.
            **kwargs: dict
                Keyword arguments forwarded to the wrapped function.

        Returns:
            mean: array_like
                Initial trajectory mean.
            cost: callable
                Nonlinear MAP cost evaluator.
            statistics: callable
                MAP cost and gradient evaluator.
            step: callable
                Update returning the unchanged initial posterior.
        """
        mean, cost, statistics, _ = original(*args, **kwargs)
        return mean, cost, statistics, lambda x, damping: (init, x, jnp.asarray(1.))

    monkeypatch.setattr(newton, "_make_functions", functions)
    fit = newton_smoother(observations, prior, transition, likelihood, init, max_iterations=4, **options)
    assert fit.iterations == 4 and fit.accepted_updates == 0 and not fit.converged
    for a, b in zip(jax.tree.leaves(fit.posterior), jax.tree.leaves(init)):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal([r["next_damping"] for r in fit.history[1:]], [200., 800., 6400., 102400.])


def test_partial_missing_observations_are_rejected(nonlinear_model):
    """Check rejection of partially missing observation rows.

    Args:
        nonlinear_model: tuple
            Nonlinear model, observations, initialization, and reference objective.
    """
    observations, prior, transition, likelihood, init, options, _ = nonlinear_model
    with pytest.raises(ValueError, match="fully finite or all-NaN"):
        newton_smoother(observations.at[1, 0].set(0.), prior, transition, likelihood, init, **options)


def test_missing_factor_with_undefined_derivatives_is_omitted():
    """Check omission of missing factors with undefined derivatives."""
    prior = Gaussian(jnp.array([-1.]), jnp.eye(1))
    transition = AdditiveGaussianModel(lambda x: .9*x + .1*jnp.sin(x), Gaussian(jnp.zeros(1), jnp.eye(1)))
    likelihood = AdditiveGaussianModel(jnp.sqrt, Gaussian(jnp.zeros(1), jnp.eye(1)))
    mean = jnp.array([[-1.], [-1.1]])
    fit = newton_smoother(
        jnp.array([[jnp.nan]]), prior, transition, likelihood, independent_init(mean), max_iterations=1
    )
    assert fit.accepted_updates == 1 and fit.posterior is not None and np.isfinite(fit.stationarity)
    expected = .5*(fit.mean[0] + 1.) @ (fit.mean[0] + 1.)
    expected += .5*jnp.sum((fit.mean[1]-transition.fun(fit.mean[0]))**2)
    np.testing.assert_allclose(fit.map_cost, expected, atol=1e-14)
