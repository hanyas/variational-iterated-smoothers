"""Differential-drive robot: iterated reverse smoother with Fourier-Hermite expansion."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from varsmooth.approximation import cubature_quadratization as quadratize
from varsmooth.approximation.fourier_hermite import get_log_likelihood
from varsmooth.approximation.fourier_hermite import get_log_prior
from varsmooth.approximation.fourier_hermite import quadratize_log_transition
from varsmooth.environments.differential_drive import get_data
from varsmooth.environments.differential_drive import make_parameters
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.smoothers.reverse_markov import iterated_reverse_markov_smoother
from varsmooth.smoothers.utils import initialize_reverse_with_forward
from varsmooth.smoothers.utils import std_backward_message

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


dt = 0.1  # discretization time step

num_steps = 500  # number of observations
dim_x, dim_y = 5, 3

prior_dist = Gaussian(
    mean=jnp.zeros((dim_x,)),
    cov=jnp.diag(jnp.array([0.1, 0.1, 0.1, 0.5, 0.2]) ** 2),
)

# wheel forces: a slowly varying forward push plus a sinusoidal turn
time = dt * np.arange(num_steps)
forward = 0.25 + 0.05 * np.sin(0.1 * time)
turn = 0.15 * np.sin(0.25 * time)
controls = jnp.column_stack([forward + turn, forward - turn])

rng = np.random.RandomState(23)
x0 = rng.multivariate_normal(prior_dist.mean, prior_dist.cov)
_, true_states, observations = get_data(x0, controls, dt=dt, random_state=rng)
transition_cov, likelihood_cov, transition_fn, likelihood_fn, _, _ = make_parameters(dt=dt)

transition_noise = Gaussian(jnp.zeros((dim_x,)), transition_cov)
likelihood_model = AdditiveGaussianModel(
    fun=likelihood_fn,
    noise=Gaussian(jnp.zeros((dim_y,)), likelihood_cov),
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
init_posterior = initialize_reverse_with_forward(forward_markov)


def log_transition_fn(marginals, kernels):
    """Quadratize each transition under its own wheel forces."""

    def single_step(marginal, kernel, control):
        transition_model = AdditiveGaussianModel(fun=lambda x: transition_fn(x, control), noise=transition_noise)
        return quadratize_log_transition(transition_model, marginal, kernel, quadratize)

    return jax.vmap(single_step)(marginals, kernels, controls)


log_prior_fn = lambda q: get_log_prior(prior_dist, q, quadratize)
log_likelihood_fn = lambda y, q: get_log_likelihood(y, likelihood_model, q, quadratize)

reverse_markov = iterated_reverse_markov_smoother(
    observations=jnp.array(observations),
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_likelihood_fn=log_likelihood_fn,
    init_posterior=init_posterior,
    kl_constraint=10,
    init_temperature=1e6,
)
marginals = std_backward_message(reverse_markov)

ts = np.arange(num_steps + 1)
mean = np.asarray(marginals.mean)
std = np.sqrt(np.diagonal(np.asarray(marginals.cov), axis1=1, axis2=2))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].plot(observations[:, 0], observations[:, 1], ".", ms=3, alpha=0.4, label="Observed")
axes[0].plot(mean[:, 0], mean[:, 1], "-", label="Smoothed")
axes[0].plot(true_states[:, 0], true_states[:, 1], "--", label="True")
axes[0].set(xlabel="x", ylabel="y", title="Planar path", aspect="equal")
for ax, idx, name in zip(axes[1:], (3, 4), ("linear momentum $p_1$", "angular momentum $p_2$")):
    ax.fill_between(ts, mean[:, idx] - 2 * std[:, idx], mean[:, idx] + 2 * std[:, idx], alpha=0.2, label=r"$\pm 2\sigma$")
    ax.plot(ts, mean[:, idx], "-", label="Smoothed")
    ax.plot(ts, true_states[:, idx], "--", label="True")
    ax.set(xlabel="time step", title=f"Unobserved {name}")
for ax in axes:
    ax.grid()
    ax.legend()
fig.tight_layout()
plt.show()
