import pytest

import jax
import numpy as np

from varsmooth.objects import Gaussian
from varsmooth.objects import AffineGaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import AdditiveGaussianModel

from varsmooth.smoothers.reverse_markov import reverse_markov_smoother
from varsmooth.smoothers.reverse_markov import backward_std_message

from tests.lgssm import simulate
from tests.test_utils import generate_system
from tests.kalman import rts_smoother


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 1, 2])
@pytest.mark.parametrize("seed", [0, 13, 42])
def test_pl_fwd_smoother(dim_x, dim_y, seed):

    from varsmooth.approximation import gauss_hermite_linearization
    from varsmooth.approximation.posterior_linearization import get_log_prior
    from varsmooth.approximation.posterior_linearization import get_log_transition
    from varsmooth.approximation.posterior_linearization import get_log_observation

    np.random.seed(seed)

    nb_steps = 100

    prior_dist, A, b, Omega, _ = generate_system(dim_x, dim_x)
    transition_model = AdditiveGaussianModel(
        lambda x: A @ x + b,
        Gaussian(np.zeros((dim_x,)), Omega)
    )

    _, H, e, Delta, _ = generate_system(dim_x, dim_y)
    observation_model = AdditiveGaussianModel(
        lambda x: H @ x + e,
        Gaussian(np.zeros((dim_y,)), Delta)
    )

    xs, ys = simulate(prior_dist.mean, A, b, Omega, H, e, Delta, nb_steps)
    rts_marginals = rts_smoother(
        ys,
        prior_dist,
        AffineGaussian(
            np.repeat([A], nb_steps, axis=0),
            np.repeat([b], nb_steps, axis=0),
            np.repeat([Omega], nb_steps, axis=0)
        ),
        AffineGaussian(
            np.repeat([H], nb_steps, axis=0),
            np.repeat([e], nb_steps, axis=0),
            np.repeat([Delta], nb_steps, axis=0)
        )
    )

    F = 1e-1 * np.eye(dim_x)
    d = np.zeros((dim_x,))
    Sigma = 1.0 * np.eye(dim_x)

    init_posterior = GaussMarkov(
        marginal=prior_dist,
        kernels=AffineGaussian(
            np.repeat([F], nb_steps, axis=0),
            np.repeat([d], nb_steps, axis=0),
            np.repeat([Sigma], nb_steps, axis=0),
        )
    )

    log_prior_fn = lambda q: get_log_prior(prior_dist, q, gauss_hermite_linearization)
    log_transition_fn = lambda q, _: get_log_transition(transition_model, q, gauss_hermite_linearization)
    log_observation_fn = lambda y, q: get_log_observation(y, observation_model, q, gauss_hermite_linearization)

    reverse_markov = reverse_markov_smoother(
        ys,
        log_prior_fn,
        log_transition_fn,
        log_observation_fn,
        init_posterior,
        0.0
    )
    var_marginals = backward_std_message(reverse_markov)

    np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_y", [1, 1, 2])
@pytest.mark.parametrize("seed", [0, 13, 42])
def test_fh_fwd_smoother(dim_x, dim_y, seed):

    from varsmooth.approximation import gauss_hermite_quadratization
    from varsmooth.approximation.fourier_hermite import get_log_prior
    from varsmooth.approximation.fourier_hermite import get_log_transition
    from varsmooth.approximation.fourier_hermite import get_log_observation

    np.random.seed(seed)

    nb_steps = 100

    prior_dist, A, b, Omega, _ = generate_system(dim_x, dim_x)
    transition_model = AdditiveGaussianModel(
        lambda x: A @ x + b,
        Gaussian(np.zeros((dim_x,)), Omega)
    )

    _, H, e, Delta, _ = generate_system(dim_x, dim_y)
    observation_model = AdditiveGaussianModel(
        lambda x: H @ x + e,
        Gaussian(np.zeros((dim_y,)), Delta)
    )

    xs, ys = simulate(prior_dist.mean, A, b, Omega, H, e, Delta, nb_steps)
    rts_marginals = rts_smoother(
        ys,
        prior_dist,
        AffineGaussian(
            np.repeat([A], nb_steps, axis=0),
            np.repeat([b], nb_steps, axis=0),
            np.repeat([Omega], nb_steps, axis=0)
        ),
        AffineGaussian(
            np.repeat([H], nb_steps, axis=0),
            np.repeat([e], nb_steps, axis=0),
            np.repeat([Delta], nb_steps, axis=0)
        )
    )

    F = 1e-1 * np.eye(dim_x)
    d = np.zeros((dim_x,))
    Sigma = 1.0 * np.eye(dim_x)

    init_posterior = GaussMarkov(
        marginal=prior_dist,
        kernels=AffineGaussian(
            np.repeat([F], nb_steps, axis=0),
            np.repeat([d], nb_steps, axis=0),
            np.repeat([Sigma], nb_steps, axis=0),
        )
    )

    log_prior_fn = lambda q: get_log_prior(prior_dist, q, gauss_hermite_quadratization)
    log_transition_fn = lambda q, p: get_log_transition(transition_model, q, p, gauss_hermite_quadratization)
    log_observation_fn = lambda y, q: get_log_observation(y, observation_model, q, gauss_hermite_quadratization)

    reverse_markov = reverse_markov_smoother(
        ys,
        log_prior_fn,
        log_transition_fn,
        log_observation_fn,
        init_posterior,
        0.0
    )
    var_marginals = backward_std_message(reverse_markov)

    np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)
