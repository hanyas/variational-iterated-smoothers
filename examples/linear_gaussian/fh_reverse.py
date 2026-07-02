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
from varsmooth.smoothers.reverse_markov import iterated_reverse_markov_smoother
from varsmooth.smoothers.reverse_markov import reverse_markov_smoother
from varsmooth.smoothers.utils import std_backward_message
from varsmooth.smoothers.rts_kalman import rts_smoother
from varsmooth.smoothers.utils import initialize_reverse_with_forward

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
# jax.config.update('jax_disable_jit', True)

np.random.seed(0)

dim_x, dim_y = 3, 2
nb_steps = 25

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

_, xs, ys = get_data(mu0, A, b, Omega, H, e, Delta, nb_steps, random_state=1)
rts_marginals = rts_smoother(
    observations=ys,
    prior_dist=prior_dist,
    linear_transition=_transition_model,
    linear_observation=_observation_model,
)

F = 1e-1 * np.eye(dim_x)
d = np.zeros((dim_x,))
Sigma = 10.0 * np.eye(dim_x)

forward_markov = GaussMarkov(
    marginal=Gaussian(
        mean=np.random.randn(dim_x),
        cov=np.eye(dim_x),
    ),
    kernels=AffineGaussian(
        F=np.repeat([F], nb_steps, axis=0),
        d=np.repeat([d], nb_steps, axis=0),
        Sigma=np.repeat([Sigma], nb_steps, axis=0),
    ),
)
forward_marginals = std_forward_message(forward_markov)

init_posterior = initialize_reverse_with_forward(forward_markov)

log_prior_fn = lambda q: get_log_prior(prior_dist, q, quadratize)
log_transition_fn = lambda q, p: get_log_transition(transition_model, q, p, quadratize)
log_observation_fn = lambda y, q: get_log_observation(y, observation_model, q, quadratize)

# single iteration no damping
reverse_markov = reverse_markov_smoother(
    observations=ys,
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_observation_fn=log_observation_fn,
    reference_posterior=init_posterior,
    temperature=0.0,
)
var_marginals = std_backward_message(reverse_markov)

np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)

# single iteration maximum damping
reverse_markov = reverse_markov_smoother(
    observations=ys,
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_observation_fn=log_observation_fn,
    reference_posterior=init_posterior,
    temperature=1e8,
)

np.testing.assert_allclose(init_posterior.marginal.mean, reverse_markov.marginal.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_posterior.marginal.cov, reverse_markov.marginal.cov, rtol=1e-3, atol=1e-3)

np.testing.assert_allclose(init_posterior.kernels.F, reverse_markov.kernels.F, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_posterior.kernels.d, reverse_markov.kernels.d, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_posterior.kernels.Sigma, reverse_markov.kernels.Sigma, rtol=1e-3, atol=1e-3)

# iterated smoother
reverse_markov = iterated_reverse_markov_smoother(
    observations=ys,
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_observation_fn=log_observation_fn,
    init_posterior=init_posterior,
    kl_constraint=100,
    init_temperature=1e6,
)
var_marginals = std_backward_message(reverse_markov)

np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)
