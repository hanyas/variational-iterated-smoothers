import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from varsmooth.approximation import gauss_hermite_quadratization as quadratize
from varsmooth.approximation.fourier_hermite import get_log_likelihood
from varsmooth.approximation.fourier_hermite import get_log_prior
from varsmooth.approximation.fourier_hermite import get_log_transition
from varsmooth.environments.bearing_only import get_data
from varsmooth.environments.bearing_only import make_parameters
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.smoothers.utils import std_forward_message
from varsmooth.smoothers.reverse_markov import iterated_reverse_markov_smoother
from varsmooth.smoothers.utils import std_backward_message
from varsmooth.smoothers.utils import initialize_reverse_with_forward

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)


s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1.0, 1.0])  # Second sensor location
x0 = jnp.array([0.1, 0.2, 1, 0])  # initial true location
r = 0.5  # Observation noise (stddev)
dt = 0.01  # discretization time step
qc = 0.01  # discretization noise
qw = 0.1  # discretization noise

num_steps = 100  # number of observations
dim_x, dim_y = 5, 2

_, true_states, observations = get_data(x0, dt, r, num_steps, s1, s2, random_state=42)
transition_cov, likelihood_cov, transition_fn, likelihood_fn, _, _ = make_parameters(qc, qw, r, dt, s1, s2)

transition_model = AdditiveGaussianModel(
    fun=transition_fn,
    noise=Gaussian(jnp.zeros((dim_x,)), transition_cov),
)
likelihood_model = AdditiveGaussianModel(
    fun=likelihood_fn,
    noise=Gaussian(jnp.zeros((dim_y,)), likelihood_cov),
)
prior_dist = Gaussian(
    mean=jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0]),
    cov=jnp.eye(dim_x),
)

F = 1e-1 * np.eye(dim_x)
d = np.zeros((dim_x,))
Sigma = 1.0 * np.eye(dim_x)

forward_markov = GaussMarkov(
    marginal=prior_dist,
    kernels=AffineGaussian(
        F=np.repeat([F], num_steps, axis=0),
        d=np.repeat([d], num_steps, axis=0),
        Sigma=np.repeat([Sigma], num_steps, axis=0),
    ),
)
forward_marginals = std_forward_message(forward_markov)

init_posterior = initialize_reverse_with_forward(forward_markov)

log_prior_fn = lambda q: get_log_prior(prior_dist, q, quadratize)
log_transition_fn = lambda q, p: get_log_transition(transition_model, q, p, quadratize)
log_likelihood_fn = lambda y, q: get_log_likelihood(y, likelihood_model, q, quadratize)

reverse_markov = iterated_reverse_markov_smoother(
    observations=jnp.array(observations),
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_likelihood_fn=log_likelihood_fn,
    init_posterior=init_posterior,
    kl_constraint=100,
    init_temperature=1e6,
)
marginals = std_backward_message(reverse_markov)

plt.figure(figsize=(7, 7))
plt.plot(marginals.mean[:, 0], marginals.mean[:, 1], "-*", label="Smoothed")
plt.plot(true_states[:, 0], true_states[:, 1], "*", label="True")
plt.grid()
plt.legend()
plt.show()
