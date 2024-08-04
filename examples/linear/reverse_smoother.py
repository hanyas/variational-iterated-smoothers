import jax
import numpy as np

from varsmooth.objects import Gaussian
from varsmooth.objects import AffineGaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import AdditiveGaussianModel

from tests.kalman import rts_smoother

from varsmooth.smoothers.reverse_markov import iterated_reverse_markov_smoother
from varsmooth.smoothers.reverse_markov import reverse_markov_smoother
from varsmooth.smoothers.reverse_markov import backward_pass

from varsmooth.linearization import gauss_hermite

from tests.lgssm import simulate
from tests.test_utils import generate_system


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# jax.config.update('jax_disable_jit', True)

dim_x, dim_y = 1, 1

np.random.seed(0)

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

xs, ys = simulate(prior_dist.mean, A, b, Omega, H, e, Delta, nb_steps, random_state=42)
rts_smoothed = rts_smoother(
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

m = np.zeros((dim_x,))
P = 10.0 * np.eye(dim_x)

F = 1e0 * np.eye(dim_x)
d = np.zeros((dim_x,))
Sigma = 1.0 * np.eye(dim_x)

init_posterior = GaussMarkov(
    marginal=Gaussian(m, P),
    kernels=AffineGaussian(
        np.repeat([F], nb_steps, axis=0),
        np.repeat([d], nb_steps, axis=0),
        np.repeat([Sigma], nb_steps, axis=0),
    )
)

# single iteration
reverse_markov = reverse_markov_smoother(
    ys,
    prior_dist,
    transition_model,
    observation_model,
    gauss_hermite,
    init_posterior,
    0.0
)
var_smoothed = backward_pass(reverse_markov)

np.testing.assert_allclose(rts_smoothed.mean, var_smoothed.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_smoothed.cov, var_smoothed.cov, rtol=1e-3, atol=1e-3)

# iterated smoother
reverse_markov = iterated_reverse_markov_smoother(
    ys,
    prior_dist,
    transition_model,
    observation_model,
    gauss_hermite,
    init_posterior,
    kl_constraint=50.0,
    init_temperature=1e2,
    nb_iter=25,
)
var_smoothed = backward_pass(reverse_markov)

np.testing.assert_allclose(rts_smoothed.mean, var_smoothed.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_smoothed.cov, var_smoothed.cov, rtol=1e-3, atol=1e-3)
