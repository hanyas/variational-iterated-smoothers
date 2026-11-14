"""Checks of the common nonlinear-robot convergence certificate and stopping."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from varsmooth.baselines import barfoot, courts
from varsmooth.baselines.convergence import decode_coordinates, encode_coordinates
from varsmooth.baselines.convergence import make_convergence_checker, make_stationarity_evaluator
from experiments.differential_drive import dd_common as dd
from varsmooth.baselines.gaussian import make_elbo, reverse_pairs

jax.config.update("jax_enable_x64", True)


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
    model_args = (ys, controls, *dd.make_model_fns(prior, Q, R))
    initial = dd.make_observation_init(prior, Q, R, controls, ys)
    evaluator, value_gradient = make_stationarity_evaluator(make_elbo(*model_args), 6, dd.STATE_SCALE)
    return prior, R, model_args, initial, evaluator, value_gradient


def test_common_coordinates_and_gradient(setup):
    """Check Gaussian-chain coordinates and their ELBO gradient.

    Args:
        setup: tuple
            Shared model, initialization, and evaluator or update function.
    """
    prior, R, model_args, initial, evaluator, value_gradient = setup
    eta = encode_coordinates(initial, dd.STATE_SCALE)
    decoded = decode_coordinates(eta, 6, dd.STATE_SCALE)
    before, after = reverse_pairs(initial)[1], reverse_pairs(decoded)[1]
    np.testing.assert_allclose(after.mean, before.mean, atol=1e-13)
    np.testing.assert_allclose(after.cov, before.cov, atol=1e-13)
    value, gradient = value_gradient(eta)
    np.testing.assert_allclose(value, make_elbo(*model_args)(initial), atol=1e-11)
    # Independent directional finite differences in all three coordinate groups
    # catch scaling, gain/mean coupling, and log-Cholesky chain-rule errors.
    rng = np.random.RandomState(12)
    evaluate = make_elbo(*model_args)
    for start, end in ((0, 35), (35, 185), (185, len(eta))):
        direction = np.zeros(len(eta))
        direction[start:end] = rng.normal(size=end - start)
        direction /= np.linalg.norm(direction)
        eps = 1e-5
        numerical = (
            evaluate(decode_coordinates(eta + eps * direction, 6, dd.STATE_SCALE))
            - evaluate(decode_coordinates(eta - eps * direction, 6, dd.STATE_SCALE))
        ) / (2 * eps)
        np.testing.assert_allclose(numerical, gradient @ direction, rtol=1e-5, atol=2e-8)


def test_both_representations_receive_the_same_residual(setup):
    """Check agreement of convergence residuals across posterior representations.

    Args:
        setup: tuple
            Shared model, initialization, and evaluator or update function.
    """
    _, _, _, initial, evaluator, _ = setup
    centers = np.asarray(reverse_pairs(initial)[1].mean)
    scale = np.tile(dd.STATE_SCALE, 2)
    courts_posterior = courts.decode(courts.encode(initial, centers, scale), centers, scale)
    first, second = evaluator(initial), evaluator(courts_posterior)
    assert first["valid"] and second["valid"]
    for key in ("elbo", "stationarity", "stationarity_mean", "stationarity_gain", "stationarity_covariance"):
        np.testing.assert_allclose(first[key], second[key], rtol=1e-9, atol=1e-11)


def test_stagnation_and_infeasibility_are_not_convergence(setup):
    """Check that stagnation and infeasibility cannot certify convergence.

    Args:
        setup: tuple
            Shared model, initialization, and evaluator or update function.
    """
    _, _, _, initial, evaluator, _ = setup
    check = make_convergence_checker(evaluator, 6)
    for _ in range(5):
        record = check(initial)
        assert record["valid"] and not record["converged"]
    assert record["stationarity"] > 1e-4
    assert abs(record["elbo_change_per_state"]) < 1e-12
    assert abs(record["symmetric_kl_per_state"]) < 1e-12
    # Even a very permissive gradient threshold cannot certify inconsistent
    # pair marginals or a non-PD conditional covariance.
    check = make_convergence_checker(evaluator, 6, stationarity_tolerance=1e6)
    assert check(initial)["converged"]
    invalid = check(initial, constraint_violation=1e-4)
    assert not invalid["valid"] and not invalid["converged"]
    bad = initial._replace(
        kernels=initial.kernels._replace(Sigma=initial.kernels.Sigma.at[0, 0, 0].set(-1.0))
    )
    invalid = check(bad)
    assert not invalid["valid"] and not invalid["converged"]


def test_revs_does_not_stop_on_repeated_zero_steps(setup, monkeypatch):
    """Check that repeated zero updates do not stop REVS prematurely.

    Args:
        setup: tuple
            Shared model, initialization, and evaluator or update function.
        monkeypatch: pytest.MonkeyPatch
            Fixture replacing the library update with a zero step.
    """
    prior, R, model_args, initial, evaluator, _ = setup

    def no_progress(observations, prior_fn, transition_fn, likelihood_fn, reference, **kwargs):
        """Return the unchanged reference posterior as a feasible zero update.

        Args:
            observations: array_like
                Unused observations.
            prior_fn: callable
                Unused prior expansion.
            transition_fn: callable
                Unused transition expansion.
            likelihood_fn: callable
                Unused likelihood expansion.
            reference: GaussMarkov
                Current reverse Gaussian posterior.
            **kwargs: dict
                Unused library solver options.

        Returns:
            posterior: GaussMarkov
                Reverse Gaussian Markov posterior.
            diagnostics: dict
                Feasible zero-step diagnostics.
        """
        return reference, dict(
            feasible=jnp.array([True]), realized_kl=jnp.array([0.0]), damping=jnp.array([0.0])
        )

    monkeypatch.setattr(dd, "iterated_reverse_markov_smoother", no_progress)
    _, history = dd.run_revs(
        model_args[0],
        *model_args[2:],
        initial,
        kl_constraint=5.0,
        evaluator=evaluator,
        max_iterations=4,
        stationarity_tolerance=1e-6,
        feasibility_tolerance=1e-8,
    )
    assert not history[-1]["converged"]
    assert history[-1]["iteration"] == 4
    assert history[-1]["stationarity"] > 1e-4


def test_baselines_stop_at_first_shared_certificate(setup):
    """Check that each driver stops at its first shared convergence certificate.

    Args:
        setup: tuple
            Shared model, initialization, and evaluator or update function.
    """
    prior, R, model_args, initial, evaluator, _ = setup
    tolerance = 1e-6
    for method in (courts, barfoot):
        extra = dict(state_scale=dd.STATE_SCALE) if method is courts else dict(prior=prior, R=R)
        posterior, history = method.fit(
            *model_args,
            initial,
            evaluator=evaluator,
            max_iterations=120,
            **extra,
            stationarity_tolerance=tolerance,
            feasibility_tolerance=1e-8,
        )
        final = history[-1]
        assert final["converged"] and final["valid"], final["message"]
        assert final["stationarity"] <= tolerance
        assert not any(row["converged"] for row in history[:-1])
        np.testing.assert_allclose(final["stationarity"], evaluator(posterior)["stationarity"], atol=1e-10)
        if method is courts:
            assert final["constraint_violation"] <= 1e-8
            assert final["solver_status"] == 3  # shared callback terminated SciPy
            assert not final["solver_success"]  # native success is not our certificate
