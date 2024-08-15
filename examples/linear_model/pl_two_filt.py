import jax
import numpy as np

from varsmooth.objects import Gaussian
from varsmooth.objects import AffineGaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import AdditiveGaussianModel

from varsmooth.smoothers.log_two_filter import log_two_filter_smoother
from varsmooth.smoothers.log_two_filter import iterated_log_two_filter_smoother
from varsmooth.smoothers.utils import initialize_reverse_with_forward
from varsmooth.smoothers.forward_markov import forward_std_message

from varsmooth.approximation import gauss_hermite_linearization as linearize
from varsmooth.approximation.posterior_linearization import get_log_prior
from varsmooth.approximation.posterior_linearization import get_log_transition
from varsmooth.approximation.posterior_linearization import get_log_observation

from tests.kalman import rts_smoother
from tests.lgssm import simulate
from tests.test_utils import generate_system


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# jax.config.update('jax_disable_jit', True)

np.random.seed(0)

dim_x, dim_y = 3, 2
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
Sigma = 1.0 * np.eye(dim_x)

init_fwd_posterior = GaussMarkov(
    marginal=prior_dist,
    kernels=AffineGaussian(
        np.repeat([F], nb_steps, axis=0),
        np.repeat([d], nb_steps, axis=0),
        np.repeat([Sigma], nb_steps, axis=0),
    )
)

init_rvs_posterior = initialize_reverse_with_forward(init_fwd_posterior)

log_prior_fn = lambda q: get_log_prior(prior_dist, q, linearize)
log_transition_fn = lambda q, _: get_log_transition(transition_model, q, linearize)
log_observation_fn = lambda y, q: get_log_observation(y, observation_model, q, linearize)

# single iteration with no damping
var_marginals = log_two_filter_smoother(
    ys,
    log_prior_fn,
    log_transition_fn,
    log_observation_fn,
    init_fwd_posterior,
    init_rvs_posterior,
    0.0
)

np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)

# single iteration maximum damping
var_marginals = log_two_filter_smoother(
    ys,
    log_prior_fn,
    log_transition_fn,
    log_observation_fn,
    init_fwd_posterior,
    init_rvs_posterior,
    1e14,
)
init_marginals = forward_std_message(init_fwd_posterior)

np.testing.assert_allclose(init_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)

var_marginals = iterated_log_two_filter_smoother(
    ys,
    log_prior_fn,
    log_transition_fn,
    log_observation_fn,
    init_fwd_posterior,
    init_rvs_posterior,
    kl_constraint=100.0,
    init_temperature=1e6,
    max_iter=100,
)

np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)
