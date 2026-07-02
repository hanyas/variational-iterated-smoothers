import jax
import numpy as np

from varsmooth.approximation import gauss_hermite_quadratization as quadratize
from varsmooth.approximation.fourier_hermite import get_log_observation
from varsmooth.approximation.fourier_hermite import get_log_prior
from varsmooth.approximation.fourier_hermite import get_log_transition
from varsmooth.environments.linear_gaussian import get_data
from varsmooth.environments.linear_gaussian import make_parameters
from varsmooth.environments.linear_gaussian import make_random_system
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.smoothers.utils import std_forward_message
from varsmooth.smoothers.rts_kalman import rts_smoother
from varsmooth.smoothers.hybrid_markov import iterated_hybrid_markov_smoother
from varsmooth.smoothers.hybrid_markov import hybrid_markov_smoother
from varsmooth.smoothers.utils import initialize_reverse_with_forward

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# jax.config.update('jax_disable_jit', True)

np.random.seed(0)

dim_x, dim_y = 3, 2
nb_steps = 100

mu0, P0, A, b, Omega, H, e, Delta = make_random_system(dim_x, dim_y, random_state=0)
prior_dist = Gaussian(mu0, P0)
_, _, transition_function, observation_function, _, _ = make_parameters(A, b, Omega, H, e, Delta)
transition_model = AdditiveGaussianModel(
    fun=transition_function,
    noise=Gaussian(np.zeros((dim_x,)), Omega),
)
observation_model = AdditiveGaussianModel(
    fun=observation_function,
    noise=Gaussian(np.zeros((dim_y,)), Delta),
)

_transition_model = AffineGaussian(
    np.repeat([A], nb_steps, axis=0),
    np.repeat([b], nb_steps, axis=0),
    np.repeat([Omega], nb_steps, axis=0),
)
_observation_model = AffineGaussian(
    np.repeat([H], nb_steps, axis=0),
    np.repeat([e], nb_steps, axis=0),
    np.repeat([Delta], nb_steps, axis=0),
)

_, xs, ys = get_data(mu0, A, b, Omega, H, e, Delta, nb_steps, random_state=13)
rts_marginals = rts_smoother(
    observations=ys,
    prior_dist=prior_dist,
    linear_transition=_transition_model,
    linear_observation=_observation_model,
)

F = 1e-1 * np.eye(dim_x)
d = np.zeros((dim_x,))
Sigma = 1.0 * np.eye(dim_x)

init_fwd_posterior = GaussMarkov(
    marginal=prior_dist,
    kernels=AffineGaussian(
        F=np.repeat([F], nb_steps, axis=0),
        d=np.repeat([d], nb_steps, axis=0),
        Sigma=np.repeat([Sigma], nb_steps, axis=0),
    ),
)

init_rvs_posterior = initialize_reverse_with_forward(init_fwd_posterior)

log_prior_fn = lambda q: get_log_prior(prior_dist, q, quadratize)
log_transition_fn = lambda q, p: get_log_transition(transition_model, q, p, quadratize)
log_observation_fn = lambda y, q: get_log_observation(y, observation_model, q, quadratize)

# single iteration with no damping
var_marginals = hybrid_markov_smoother(
    observations=ys,
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_observation_fn=log_observation_fn,
    forward_reference=init_fwd_posterior,
    reverse_reference=init_rvs_posterior,
    temperature=0.0,
)

np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)

# single iteration maximum damping
var_marginals = hybrid_markov_smoother(
    observations=ys,
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_observation_fn=log_observation_fn,
    forward_reference=init_fwd_posterior,
    reverse_reference=init_rvs_posterior,
    temperature=1e8,
)
init_marginals = std_forward_message(init_fwd_posterior)

np.testing.assert_allclose(init_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)

var_marginals = iterated_hybrid_markov_smoother(
    observations=ys,
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_observation_fn=log_observation_fn,
    init_forward_posterior=init_fwd_posterior,
    init_reverse_posterior=init_rvs_posterior,
    kl_constraint=100,
    init_temperature=1e8,
)

np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)
