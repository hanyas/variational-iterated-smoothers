import pytest

import jax
import numpy as np

from varsmooth.objects import Gaussian
from varsmooth.objects import AffineGaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import AdditiveGaussianModel

from tests.kalman import rts_smoother

from varsmooth.smoothers.reverse_markov import reverse_markov_smoother
from varsmooth.smoothers.reverse_markov import backward_pass

from tests.lgssm import simulate
from tests.test_utils import generate_system


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.parametrize("dim_x", [1, 3, 5])
@pytest.mark.parametrize("dim_y", [2, 3, 4])
@pytest.mark.parametrize("seed", [0, 13, 42, 69])
def test_rts_smoother_vs_var_smoother(dim_x, dim_y, seed):

    np.random.seed(seed)

    nb_steps = 100

    prior, A, b, Omega, _ = generate_system(dim_x, dim_x)
    transition_model = AdditiveGaussianModel(
        lambda x: A @ x + b,
        Gaussian(np.zeros((dim_x,)), Omega)
    )

    _, H, e, Delta, _ = generate_system(dim_x, dim_y)
    observation_model = AdditiveGaussianModel(
        lambda x: H @ x + e,
        Gaussian(np.zeros((dim_y,)), Delta)
    )

    xs, ys = simulate(prior.mean, A, b, Omega, H, e, Delta, nb_steps)
    rts_smoothed = rts_smoother(
        ys, prior,
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

    F = np.eye(dim_x)
    d = np.zeros((dim_x,))
    Sigma = 10.0 * np.eye(dim_x)

    init_forward_markov = GaussMarkov(
        marginal=Gaussian(
            np.zeros((dim_x,)),
            10.0 * np.eye(dim_x),
        ),
        kernels=AffineGaussian(
            np.repeat([F], nb_steps, axis=0),
            np.repeat([d], nb_steps, axis=0),
            np.repeat([Sigma], nb_steps, axis=0),
        )
    )

    var_reverse_markov = reverse_markov_smoother(
        ys, prior,
        transition_model,
        observation_model,
        init_forward_markov,
        1e-6
    )
    var_smoothed = backward_pass(var_reverse_markov)

    np.testing.assert_allclose(rts_smoothed.mean, var_smoothed.mean, atol=1e-3)
    np.testing.assert_allclose(rts_smoothed.cov, var_smoothed.cov, atol=1e-3)
