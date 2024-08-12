import jax
import numpy as np

from varsmooth.objects import Gaussian
from varsmooth.objects import AffineGaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import AdditiveGaussianModel

from varsmooth.smoothers.reverse_markov import iterated_reverse_markov_smoother
from varsmooth.smoothers.reverse_markov import reverse_markov_smoother
from varsmooth.smoothers.reverse_markov import backward_std_message
from varsmooth.smoothers.forward_markov import forward_std_message
from varsmooth.smoothers.utils import initialize_reverse_with_forward

from varsmooth.approximation import gauss_hermite_quadratization as quadratize
from varsmooth.approximation.fourier_hermite import get_log_prior
from varsmooth.approximation.fourier_hermite import get_log_transition
from varsmooth.approximation.fourier_hermite import get_log_observation

from tests.kalman import rts_smoother
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

xs, ys = simulate(prior_dist.mean, A, b, Omega, H, e, Delta, nb_steps, random_state=13)
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
Sigma = 10.0 * np.eye(dim_x)

forward_markov = GaussMarkov(
    marginal=Gaussian(
        mean=np.random.randn(prior_dist.mean.shape[0]),
        cov=np.eye(prior_dist.mean.shape[0])
    ),
    kernels=AffineGaussian(
        np.repeat([F], nb_steps, axis=0),
        np.repeat([d], nb_steps, axis=0),
        np.repeat([Sigma], nb_steps, axis=0),
    )
)
forward_marginals = forward_std_message(forward_markov)

init_posterior = initialize_reverse_with_forward(forward_markov)

log_prior_fn = lambda q: get_log_prior(prior_dist, q, quadratize)
log_transition_fn = lambda q, p: get_log_transition(transition_model, q, p, quadratize)
log_observation_fn = lambda y, q: get_log_observation(y, observation_model, q, quadratize)

# single iteration no damping
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

# single iteration maximum damping
reverse_markov = reverse_markov_smoother(
    ys,
    log_prior_fn,
    log_transition_fn,
    log_observation_fn,
    init_posterior,
    1e8,
)

np.testing.assert_allclose(init_posterior.marginal.mean, reverse_markov.marginal.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_posterior.marginal.cov, reverse_markov.marginal.cov, rtol=1e-3, atol=1e-3)

np.testing.assert_allclose(init_posterior.kernels.F, reverse_markov.kernels.F, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_posterior.kernels.d, reverse_markov.kernels.d, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_posterior.kernels.Sigma, reverse_markov.kernels.Sigma, rtol=1e-3, atol=1e-3)

# iterated smoother
reverse_markov = iterated_reverse_markov_smoother(
    ys,
    log_prior_fn,
    log_transition_fn,
    log_observation_fn,
    init_posterior,
    kl_constraint=100.0,
    init_temperature=1e2,
    max_iter=25
)
var_marginals = backward_std_message(reverse_markov)

np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)
