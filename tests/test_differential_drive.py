"""Checks for differential-drive dynamics, expectations, and initialization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from varsmooth.environments import differential_drive as robot
from varsmooth.objects import AdditiveGaussianModel, Gaussian
from varsmooth.smoothers.utils import std_backward_message
from varsmooth.baselines.courts import decode, encode, make_objective
from experiments.differential_drive import dd_common as dd
from varsmooth.environments.differential_drive import expected_transition
from varsmooth.baselines.gaussian import make_elbo, reverse_pairs

jax.config.update("jax_enable_x64", True)


def test_controlled_dynamics_and_jacobian():
    """Check the robot transition and its state Jacobian."""
    Q, R, f, h, df, dh = robot.make_parameters()
    x = jnp.array([0.0, 0.0, 0.0, 5.0, 0.0])
    u = jnp.array([0.5, 0.5])
    np.testing.assert_allclose(f(x, u), [0.1, 0.0, 0.0, 5.0, 0.0], atol=1e-14)
    x = jnp.array([1.0, -2.0, 0.7, 2.0, -0.3])
    eps = 1e-5
    finite_diff = np.column_stack(
        [(np.asarray(f(x + eps * e, u)) - np.asarray(f(x - eps * e, u))) / (2 * eps) for e in np.eye(5)]
    )
    np.testing.assert_allclose(df(x, u), finite_diff, atol=1e-9)
    np.testing.assert_array_equal(h(x), x[:3])
    np.testing.assert_array_equal(dh(x), np.eye(3, 5))


def test_simulation_uses_inference_transition_and_covariances():
    """Check agreement between robot simulation and inference parameters."""
    controls = np.array([[0.2, 0.3], [0.4, 0.1], [0.2, 0.2]])
    x0 = np.array([0.0, 0.0, 7.0, 2.0, 0.1])  # unwrapped heading
    Q, R, f, h, *_ = robot.make_parameters()
    ts, xs, ys = robot.get_data(x0, controls, random_state=7)
    rng = np.random.RandomState(7)
    ws = rng.standard_normal((3, 5)) @ np.linalg.cholesky(Q).T
    vs = rng.standard_normal((3, 3)) @ np.linalg.cholesky(R).T
    predicted = np.array([f(x, u) for x, u in zip(xs[:-1], controls)])
    np.testing.assert_allclose(xs[1:] - predicted, ws, atol=1e-14)
    np.testing.assert_allclose(ys - xs[1:, :3], vs, atol=1e-14)
    np.testing.assert_allclose(ts, [0.1, 0.2, 0.3], atol=1e-14)
    np.testing.assert_array_equal(xs, robot.get_data(x0, controls, random_state=7)[1])


def test_prior_initialization_ignores_observations_and_preserves_consistency():
    """Check observation independence and temporal consistency of the prior chain."""
    Q, R, *_ = dd.make_parameters(**dd.PHYSICAL)
    prior = dd.make_prior()
    _, ys, controls = dd.simulate_data(prior, Q, R, 12, np.random.RandomState(1))
    _, changed_ys, changed_controls = dd.simulate_data(prior, Q, R, 12, np.random.RandomState(7))
    assert not np.array_equal(ys, changed_ys)
    initial = dd.make_prior_init(prior, Q, controls)
    changed = dd.make_prior_init(prior, Q, changed_controls)
    for left, right in zip(jax.tree.leaves(initial), jax.tree.leaves(changed)):
        np.testing.assert_array_equal(left, right)

    # The nonlinear control rollout and its propagated uncertainty define the
    # Gaussian chain before any observations enter the optimization.
    _, _, transition, _, jacobian, _ = robot.make_parameters(Q=Q, R=R, **dd.PHYSICAL)
    mean, cov = prior
    expected_means, expected_covs = [mean], [cov]
    for control in controls:
        F = jacobian(mean, control)
        mean, cov = transition(mean, control), F @ cov @ F.T + Q
        expected_means.append(mean)
        expected_covs.append(cov)
    marginal = std_backward_message(initial)
    np.testing.assert_allclose(marginal.mean, expected_means, atol=1e-12)
    np.testing.assert_allclose(marginal.cov, expected_covs, atol=1e-12)
    assert np.any(np.asarray(reverse_pairs(initial)[1].cov[:, :5, 5:]) != 0.0)
    objective, data = make_objective(ys, controls, *dd.make_model_fns(prior, Q, R), initial, dd.STATE_SCALE)
    parameters = encode(initial, data["centers"], data["scale"], data["transform"])
    np.testing.assert_allclose(objective["constraints"](parameters), 0.0, atol=1e-12)


@pytest.mark.parametrize("seed", [3, 11])
def test_exact_gaussian_expectation_against_log_density_samples(seed):
    """Check analytic Gaussian expectations against sampled log densities.

    Args:
        seed: int
            Random seed for data generation.
    """
    Q, _, f, *_ = robot.make_parameters()
    u = jnp.array([0.2, 0.3])
    previous = jnp.array([1.0, 2.0, 0.7, 2.0, -0.3])
    mean = jnp.concatenate([previous, f(previous, u)])
    rng = np.random.RandomState(seed)
    root = rng.normal(size=(10, 10)) * 0.04
    cov = jnp.asarray(root @ root.T)
    transition = AdditiveGaussianModel(lambda x: f(x, u), Gaussian(jnp.zeros(5), Q))
    logpdf = lambda z: transition.log_prob(z[5:], z[:5])
    np.testing.assert_allclose(expected_transition(mean, jnp.zeros((10, 10)), u, Q), logpdf(mean), atol=1e-11)
    samples = mean + jnp.asarray(rng.normal(size=(120000, 10)) @ root.T)
    values = np.asarray(jax.jit(jax.vmap(logpdf))(samples))
    exact = float(expected_transition(mean, cov, u, Q))
    assert abs(exact - np.mean(values)) < 6 * np.std(values) / np.sqrt(len(values))
    # Independent Price/Bonnet check: average pointwise derivatives instead of
    # differentiating the analytic expectation formula used by the baselines.
    sample_grad, sample_hess = jax.jit(
        lambda z: (jax.vmap(jax.grad(logpdf))(z), jax.vmap(jax.hessian(logpdf))(z))
    )(samples[:20000])
    dm, dP = jax.grad(expected_transition, argnums=(0, 1))(mean, cov, u, Q)
    for exact_derivative, empirical in ((dm, sample_grad), (2 * dP, sample_hess)):
        empirical = np.asarray(empirical)
        error = np.abs(exact_derivative - empirical.mean(axis=0))
        se = empirical.std(axis=0) / np.sqrt(len(empirical))
        assert np.all(error < 7 * se + 1e-8)


def test_pairwise_objective():
    """Check agreement of pairwise and chain objectives."""
    Q, R, *_ = dd.make_parameters(**dd.PHYSICAL)
    prior = dd.make_prior()
    _, ys, controls = dd.simulate_data(prior, Q, R, 6, np.random.RandomState(4))
    initial = dd.make_observation_init(prior, Q, R, controls, ys)
    # Use a correlated chain to catch reverse-kernel/block-order mistakes.
    from varsmooth.objects import AffineGaussian, GaussMarkov

    F = jnp.broadcast_to(0.2 * jnp.eye(5), (6, 5, 5))
    initial = GaussMarkov(initial.marginal, AffineGaussian(F, initial.kernels.d, initial.kernels.Sigma))
    objective, data = make_objective(ys, controls, *dd.make_model_fns(prior, Q, R), initial, dd.STATE_SCALE)
    x = encode(initial, data["centers"], data["scale"])
    np.testing.assert_allclose(objective["constraints"](x), 0.0, atol=1e-14)
    actual = std_backward_message(decode(x, data["centers"], data["scale"]))
    expected = std_backward_message(initial)
    np.testing.assert_allclose(actual.mean, expected.mean, atol=1e-12)
    np.testing.assert_allclose(actual.cov, expected.cov, atol=1e-12)
    np.testing.assert_allclose(
        -objective["value_gradient"](x)[0],
        make_elbo(ys, controls, *dd.make_model_fns(prior, Q, R))(initial),
        atol=1e-8,
    )
    # Independent directional differences verify sparse derivative assembly,
    # including the Hessian of multiplier-weighted nonlinear constraints.
    rng = np.random.RandomState(8)
    direction = rng.normal(size=x.shape) * 0.01
    multipliers = rng.normal(size=objective["constraints"](x).shape)
    eps = 1e-4
    numerical_jv = (
        objective["constraints"](x + eps * direction) - objective["constraints"](x - eps * direction)
    ) / (2 * eps)
    np.testing.assert_allclose(objective["jacobian"](x) @ direction, numerical_jv, atol=1e-9)
    numerical_hv = (
        objective["value_gradient"](x + eps * direction)[1]
        - objective["value_gradient"](x - eps * direction)[1]
    ) / (2 * eps)
    np.testing.assert_allclose(objective["hessian"](x) @ direction, numerical_hv, atol=1e-6)
    numerical_chv = (
        (objective["jacobian"](x + eps * direction).T - objective["jacobian"](x - eps * direction).T)
        @ multipliers
        / (2 * eps)
    )
    np.testing.assert_allclose(
        objective["constraint_hessian"](x, multipliers) @ direction, numerical_chv, atol=1e-9
    )


def test_numerical_revs_uses_pairwise_smoothing_distribution():
    """Check reverse FH updates against quadrature on the actual smoothing pairs."""
    from varsmooth.approximation import cubature_quadratization
    from varsmooth.objects import LogTransition
    from varsmooth.smoothers.reverse_markov import log_forward_message, reverse_markov_smoother

    Q, R, f, *_ = dd.make_parameters(**dd.PHYSICAL)
    prior = dd.make_prior()
    _, ys, controls = dd.simulate_data(prior, Q, R, 6, np.random.RandomState(4))
    initial = dd.make_observation_init(prior, Q, R, controls, ys)
    marginals, pairs = reverse_pairs(initial)
    lp, lt, ll = dd.make_revs_fns(prior, Q, R, controls)
    order = jnp.r_[5:10, :5]

    def transition(mean, cov, control):
        """Integrate directly on a pair in (next, previous) state order.

        Args:
            mean: array_like
                Pair mean in chronological order.
            cov: array_like
                Pair covariance in chronological order.
            control: array_like
                Wheel-force inputs.

        Returns:
            factor: LogTransition
                Numerically integrated FH coefficients.
        """
        model = AdditiveGaussianModel(lambda x: f(x, control), Gaussian(jnp.zeros(5), Q))
        pair = Gaussian(mean[order], cov[order][:, order])
        C, c, kappa = cubature_quadratization(lambda z: model.log_prob(z[:5], z[5:]), pair)
        return LogTransition(C[:5, :5], -C[:5, 5:], -C[5:, :5], C[5:, 5:], c[:5], c[5:], kappa)

    factors = (
        lp(Gaussian(marginals.mean[0], marginals.cov[0])),
        jax.jit(jax.vmap(transition))(pairs.mean, pairs.cov, controls),
        ll(ys, Gaussian(marginals.mean[1:], marginals.cov[1:])),
    )
    candidate = reverse_markov_smoother(ys, lp, lt, ll, initial, 1.0)
    expected, *_ = log_forward_message(*factors, initial, 0.5)
    for actual, target in zip(jax.tree.leaves(candidate), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, target, rtol=1e-10, atol=1e-11)

    candidate, diagnostics = dd.iterated_reverse_markov_smoother(
        ys, lp, lt, ll, initial, 5.0, max_iterations=1, return_history=True, verbose=False
    )
    assert diagnostics["feasible"][0]
    expected, *_ = log_forward_message(*factors, initial, diagnostics["damping"][0])
    for actual, target in zip(jax.tree.leaves(candidate), jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, target, rtol=1e-10, atol=1e-11)
    assert np.isfinite(diagnostics["elbo"][0])
