import jax
import jax.numpy as jnp
import numpy as np

from varsmooth.objects import Gaussian
from varsmooth.objects import AffineGaussian
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import GaussMarkov

from varsmooth.smoothers.reverse_markov import iterated_reverse_markov_smoother
from varsmooth.smoothers.reverse_markov import backward_pass
from varsmooth.smoothers.forward_markov import forward_pass
from varsmooth.smoothers.utils import initialize_reverse_with_forward

from varsmooth.approximation import gauss_hermite_linearization
from varsmooth.approximation.bayes_gauss_newton import get_log_prior
from varsmooth.approximation.bayes_gauss_newton import get_log_transition
from varsmooth.approximation.bayes_gauss_newton import get_log_observation

from bearing_model import get_data, make_parameters

import matplotlib.pyplot as plt

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_disable_jit', True)


s1 = jnp.array([-1.5, 0.5])  # First sensor location
s2 = jnp.array([1.0, 1.0])  # Second sensor location
r = 0.5  # Observation noise (stddev)
x0 = jnp.array([0.1, 0.2, 1, 0])  # initial true location
dt = 0.01  # discretization time step

qc = 0.01  # discretization noise
qw = 0.1  # discretization noise

nb_steps = 500  # number of observations
dim_x, dim_y = 5, 2

_, true_states, observations = get_data(x0, dt, r, nb_steps, s1, s2, random_state=42)
transition_cov, observation_cov, \
    transition_fn, observation_fn, _, _ = make_parameters(qc, qw, r, dt, s1, s2)

transition_model = AdditiveGaussianModel(
    fun=transition_fn,
    noise=Gaussian(jnp.zeros((dim_x,)), transition_cov)
)
observation_model = AdditiveGaussianModel(
    fun=observation_fn,
    noise=Gaussian(jnp.zeros((dim_y,)), observation_cov)
)
prior_dist = Gaussian(
    mean=jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0]),
    cov=jnp.eye(dim_x)
)

F = 1e-1 * np.eye(dim_x)
d = np.zeros((dim_x,))
Sigma = 1.0 * np.eye(dim_x)

forward_markov = GaussMarkov(
    marginal=prior_dist,
    kernels=AffineGaussian(
        F=np.repeat([F], nb_steps, axis=0),
        d=np.repeat([d], nb_steps, axis=0),
        Sigma=np.repeat([Sigma], nb_steps, axis=0),
    )
)
forward_marginals = forward_pass(forward_markov)

init_posterior = initialize_reverse_with_forward(forward_markov, forward_marginals)

log_prior_fn = lambda q: get_log_prior(prior_dist, q, gauss_hermite_linearization)
log_transition_fn = lambda q: get_log_transition(transition_model, q, gauss_hermite_linearization)
log_observation_fn = lambda y, q: get_log_observation(y, observation_model, q, gauss_hermite_linearization)

reverse_markov = iterated_reverse_markov_smoother(
    jnp.array(observations),
    log_prior_fn,
    log_transition_fn,
    log_observation_fn,
    init_posterior,
    kl_constraint=500,
    init_temperature=1e2,
    nb_iter=100,
)
marginals = backward_pass(reverse_markov)

plt.figure(figsize=(7, 7))
plt.plot(
    marginals.mean[:, 0],
    marginals.mean[:, 1],
    "-*",
    label="Smoothed"
)
plt.plot(true_states[:, 0], true_states[:, 1], "*", label="True")
plt.grid()
plt.legend()
plt.show()
