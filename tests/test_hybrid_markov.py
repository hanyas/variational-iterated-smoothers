import jax
import numpy as np
import pytest

from tests.lgssm import simulate
from tests.test_utils import generate_system
from varsmooth.approximation import gauss_hermite_linearization as linearize
from varsmooth.approximation.linearization import get_log_observation
from varsmooth.approximation.linearization import get_log_prior
from varsmooth.approximation.linearization import get_log_transition
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.smoothers.hybrid_markov import hybrid_markov_smoother
from varsmooth.smoothers.hybrid_markov import iterated_hybrid_markov_smoother
from varsmooth.smoothers.rts_kalman import rts_smoother
from varsmooth.smoothers.utils import initialize_reverse_with_forward


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_disable_jit", False)
    jax.config.update("jax_debug_nans", False)


def _lg_setup(dim_x, dim_y, seed):
    """Build a random linear-Gaussian problem and its exact RTS marginals."""
    np.random.seed(seed)
    num_steps = 100

    prior_dist, A, b, Omega, _ = generate_system(dim_x, dim_x)
    transition_model = AdditiveGaussianModel(lambda x: A @ x + b, Gaussian(np.zeros((dim_x,)), Omega))

    _, H, e, Delta, _ = generate_system(dim_x, dim_y)
    observation_model = AdditiveGaussianModel(lambda x: H @ x + e, Gaussian(np.zeros((dim_y,)), Delta))

    _, ys = simulate(prior_dist.mean, A, b, Omega, H, e, Delta, num_steps)
    rts_marginals = rts_smoother(
        observations=ys,
        prior_dist=prior_dist,
        linear_transition=AffineGaussian(
            np.repeat([A], num_steps, axis=0),
            np.repeat([b], num_steps, axis=0),
            np.repeat([Omega], num_steps, axis=0),
        ),
        linear_observation=AffineGaussian(
            np.repeat([H], num_steps, axis=0),
            np.repeat([e], num_steps, axis=0),
            np.repeat([Delta], num_steps, axis=0),
        ),
    )

    F = 1e-1 * np.eye(dim_x)
    d = np.zeros((dim_x,))
    Sigma = 1.0 * np.eye(dim_x)
    init_forward = GaussMarkov(
        marginal=prior_dist,
        kernels=AffineGaussian(
            np.repeat([F], num_steps, axis=0),
            np.repeat([d], num_steps, axis=0),
            np.repeat([Sigma], num_steps, axis=0),
        ),
    )
    init_reverse = initialize_reverse_with_forward(init_forward)

    log_prior_fn = lambda q: get_log_prior(prior_dist, q, linearize)
    log_transition_fn = lambda q, _: get_log_transition(transition_model, q, linearize)
    log_observation_fn = lambda y, q: get_log_observation(y, observation_model, q, linearize)

    model_fns = (log_prior_fn, log_transition_fn, log_observation_fn)
    return ys, rts_marginals, init_forward, init_reverse, model_fns


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 1, 2])
@pytest.mark.parametrize("seed", [0, 13, 42])
def test_hybrid_single_pass_matches_rts(dim_x, dim_y, seed):
    ys, rts_marginals, init_forward, init_reverse, (lp, lt, lo) = _lg_setup(dim_x, dim_y, seed)

    # Undamped (temperature 0) single pass is exact for a linear-Gaussian model.
    var_marginals = hybrid_markov_smoother(ys, lp, lt, lo, init_forward, init_reverse, temperature=0.0)

    np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim_x, dim_y, seed", [(1, 1, 0), (2, 1, 13), (3, 2, 42)])
def test_hybrid_iterated_matches_rts(dim_x, dim_y, seed):
    ys, rts_marginals, init_forward, init_reverse, (lp, lt, lo) = _lg_setup(dim_x, dim_y, seed)

    # A wide trust region lets the exact undamped step be accepted immediately,
    # so a handful of iterations suffice to recover the RTS marginals.
    var_marginals = iterated_hybrid_markov_smoother(
        ys,
        lp,
        lt,
        lo,
        init_forward,
        init_reverse,
        kl_constraint=1e6,
        max_iterations=20,
        verbose=False,
    )

    np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)
