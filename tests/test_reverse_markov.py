import jax
import numpy as np
import pytest

from tests.lgssm import simulate
from tests.test_utils import generate_system
from varsmooth.approximation import gauss_hermite_linearization as linearize
from varsmooth.approximation import gauss_hermite_quadratization as quadratize
from varsmooth.approximation.fourier_hermite import get_log_likelihood as fh_get_log_likelihood
from varsmooth.approximation.fourier_hermite import get_log_prior as fh_get_log_prior
from varsmooth.approximation.fourier_hermite import get_log_transition as fh_get_log_transition
from varsmooth.approximation.linearization import get_log_likelihood as pl_get_log_likelihood
from varsmooth.approximation.linearization import get_log_prior as pl_get_log_prior
from varsmooth.approximation.linearization import get_log_transition as pl_get_log_transition
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.smoothers.reverse_markov import reverse_markov_smoother
from varsmooth.smoothers.rts_kalman import rts_smoother
from varsmooth.smoothers.utils import std_backward_message


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_disable_jit", False)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 1, 2])
@pytest.mark.parametrize("seed", [0, 13, 42])
def test_pl_rev_smoother(dim_x, dim_y, seed):

    np.random.seed(seed)

    num_steps = 100

    prior_dist, A, b, Omega, _ = generate_system(dim_x, dim_x)
    transition_model = AdditiveGaussianModel(lambda x: A @ x + b, Gaussian(np.zeros((dim_x,)), Omega))

    _, H, e, Delta, _ = generate_system(dim_x, dim_y)
    likelihood_model = AdditiveGaussianModel(lambda x: H @ x + e, Gaussian(np.zeros((dim_y,)), Delta))

    xs, ys = simulate(prior_dist.mean, A, b, Omega, H, e, Delta, num_steps)
    rts_marginals = rts_smoother(
        observations=ys,
        prior_dist=prior_dist,
        linear_transition=AffineGaussian(
            np.repeat([A], num_steps, axis=0),
            np.repeat([b], num_steps, axis=0),
            np.repeat([Omega], num_steps, axis=0),
        ),
        linear_likelihood=AffineGaussian(
            np.repeat([H], num_steps, axis=0),
            np.repeat([e], num_steps, axis=0),
            np.repeat([Delta], num_steps, axis=0),
        ),
    )

    F = 1e-1 * np.eye(dim_x)
    d = np.zeros((dim_x,))
    Sigma = 1.0 * np.eye(dim_x)

    init_posterior = GaussMarkov(
        marginal=prior_dist,
        kernels=AffineGaussian(
            np.repeat([F], num_steps, axis=0),
            np.repeat([d], num_steps, axis=0),
            np.repeat([Sigma], num_steps, axis=0),
        ),
    )

    log_prior_fn = lambda q: pl_get_log_prior(prior_dist, q, linearize)
    log_transition_fn = lambda q, _: pl_get_log_transition(transition_model, q, linearize)
    log_likelihood_fn = lambda y, q: pl_get_log_likelihood(y, likelihood_model, q, linearize)

    reverse_markov = reverse_markov_smoother(
        ys, log_prior_fn, log_transition_fn, log_likelihood_fn, init_posterior, 0.0
    )
    var_marginals = std_backward_message(reverse_markov)

    np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 1, 2])
@pytest.mark.parametrize("seed", [0, 13, 42])
def test_fh_rev_smoother(dim_x, dim_y, seed):

    np.random.seed(seed)

    num_steps = 100

    prior_dist, A, b, Omega, _ = generate_system(dim_x, dim_x)
    transition_model = AdditiveGaussianModel(lambda x: A @ x + b, Gaussian(np.zeros((dim_x,)), Omega))

    _, H, e, Delta, _ = generate_system(dim_x, dim_y)
    likelihood_model = AdditiveGaussianModel(lambda x: H @ x + e, Gaussian(np.zeros((dim_y,)), Delta))

    xs, ys = simulate(prior_dist.mean, A, b, Omega, H, e, Delta, num_steps)
    rts_marginals = rts_smoother(
        observations=ys,
        prior_dist=prior_dist,
        linear_transition=AffineGaussian(
            np.repeat([A], num_steps, axis=0),
            np.repeat([b], num_steps, axis=0),
            np.repeat([Omega], num_steps, axis=0),
        ),
        linear_likelihood=AffineGaussian(
            np.repeat([H], num_steps, axis=0),
            np.repeat([e], num_steps, axis=0),
            np.repeat([Delta], num_steps, axis=0),
        ),
    )

    F = 1e-1 * np.eye(dim_x)
    d = np.zeros((dim_x,))
    Sigma = 1.0 * np.eye(dim_x)

    init_posterior = GaussMarkov(
        marginal=prior_dist,
        kernels=AffineGaussian(
            np.repeat([F], num_steps, axis=0),
            np.repeat([d], num_steps, axis=0),
            np.repeat([Sigma], num_steps, axis=0),
        ),
    )

    log_prior_fn = lambda q: fh_get_log_prior(prior_dist, q, quadratize)
    log_transition_fn = lambda q, p: fh_get_log_transition(transition_model, q, p, quadratize)
    log_likelihood_fn = lambda y, q: fh_get_log_likelihood(y, likelihood_model, q, quadratize)

    reverse_markov = reverse_markov_smoother(
        ys, log_prior_fn, log_transition_fn, log_likelihood_fn, init_posterior, 0.0
    )
    var_marginals = std_backward_message(reverse_markov)

    np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)
