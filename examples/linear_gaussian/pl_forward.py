"""Linear-Gaussian system: iterated forward smoother with posterior linearization.

Checks a single undamped pass, a single fully damped pass, and the iterated
smoother against the exact RTS smoother.
"""

import jax
import numpy as np

from varsmooth.approximation import gauss_hermite_linearization as linearize
from varsmooth.approximation.linearization import get_log_likelihood
from varsmooth.approximation.linearization import get_log_prior
from varsmooth.approximation.linearization import get_log_transition
from varsmooth.environments.linear_gaussian import get_data
from varsmooth.environments.linear_gaussian import make_parameters
from varsmooth.environments.linear_gaussian import make_random_system
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.smoothers.forward_markov import forward_markov_smoother
from varsmooth.smoothers.forward_markov import iterated_forward_markov_smoother
from varsmooth.smoothers.rts_kalman import rts_smoother
from varsmooth.smoothers.utils import std_forward_message

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


dim_x, dim_y = 3, 2
num_steps = 100

mu0, P0, A, b, Omega, H, e, Delta = make_random_system(dim_x, dim_y, random_state=0)
prior_dist = Gaussian(mu0, P0)
_, _, transition_fn, likelihood_fn, _, _ = make_parameters(A, b, Omega, H, e, Delta)
transition_model = AdditiveGaussianModel(
    fun=transition_fn,
    noise=Gaussian(np.zeros((dim_x,)), Omega),
)
likelihood_model = AdditiveGaussianModel(
    fun=likelihood_fn,
    noise=Gaussian(np.zeros((dim_y,)), Delta),
)

linear_transition = AffineGaussian(
    np.repeat([A], num_steps, axis=0),
    np.repeat([b], num_steps, axis=0),
    np.repeat([Omega], num_steps, axis=0),
)
linear_likelihood = AffineGaussian(
    np.repeat([H], num_steps, axis=0),
    np.repeat([e], num_steps, axis=0),
    np.repeat([Delta], num_steps, axis=0),
)

_, _, observations = get_data(mu0, A, b, Omega, H, e, Delta, num_steps, random_state=13)
rts_marginals = rts_smoother(
    observations=observations,
    prior_dist=prior_dist,
    linear_transition=linear_transition,
    linear_likelihood=linear_likelihood,
)

F = 1e-1 * np.eye(dim_x)
d = np.zeros((dim_x,))
Sigma = 1.0 * np.eye(dim_x)

rng = np.random.RandomState(0)
init_posterior = GaussMarkov(
    marginal=Gaussian(
        mean=rng.randn(dim_x),
        cov=np.eye(dim_x),
    ),
    kernels=AffineGaussian(
        F=np.repeat([F], num_steps, axis=0),
        d=np.repeat([d], num_steps, axis=0),
        Sigma=np.repeat([Sigma], num_steps, axis=0),
    ),
)

log_prior_fn = lambda q: get_log_prior(prior_dist, q, linearize)
log_transition_fn = lambda q, _: get_log_transition(transition_model, q, linearize)
log_likelihood_fn = lambda y, q: get_log_likelihood(y, likelihood_model, q, linearize)

# single iteration, no damping
forward_markov = forward_markov_smoother(
    observations=observations,
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_likelihood_fn=log_likelihood_fn,
    reference_posterior=init_posterior,
    temperature=0.0,
)
var_marginals = std_forward_message(forward_markov)

np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)

# single iteration, maximum damping
forward_markov = forward_markov_smoother(
    observations=observations,
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_likelihood_fn=log_likelihood_fn,
    reference_posterior=init_posterior,
    temperature=1e8,
)

np.testing.assert_allclose(init_posterior.marginal.mean, forward_markov.marginal.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_posterior.marginal.cov, forward_markov.marginal.cov, rtol=1e-3, atol=1e-3)

np.testing.assert_allclose(init_posterior.kernels.F, forward_markov.kernels.F, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_posterior.kernels.d, forward_markov.kernels.d, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(init_posterior.kernels.Sigma, forward_markov.kernels.Sigma, rtol=1e-3, atol=1e-3)

# iterated smoother
forward_markov = iterated_forward_markov_smoother(
    observations=observations,
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_likelihood_fn=log_likelihood_fn,
    init_posterior=init_posterior,
    kl_constraint=100,
    init_temperature=1e6,
)
var_marginals = std_forward_message(forward_markov)

np.testing.assert_allclose(rts_marginals.mean, var_marginals.mean, rtol=1e-3, atol=1e-3)
np.testing.assert_allclose(rts_marginals.cov, var_marginals.cov, rtol=1e-3, atol=1e-3)

print("Variational marginals match the RTS smoother.")
