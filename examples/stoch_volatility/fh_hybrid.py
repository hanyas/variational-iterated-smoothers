import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from varsmooth.approximation import gauss_hermite_quadratization as quadratize
from varsmooth.approximation.fourier_hermite import get_log_observation
from varsmooth.approximation.fourier_hermite import get_log_prior
from varsmooth.approximation.fourier_hermite import get_log_transition
from varsmooth.environments.stoch_volatility import get_data
from varsmooth.environments.stoch_volatility import make_parameters
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import ConditionalMomentsModel
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.smoothers.hybrid_markov import iterated_hybrid_markov_smoother
from varsmooth.smoothers.utils import initialize_reverse_with_forward

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


mu = -0.5  # long-run mean log-volatility
phi = 0.98  # autoregressive persistence
sigma = 0.16  # transition noise std

nb_steps = 1024  # number of observations
dim_x, dim_y = 1, 1

p0 = sigma**2 / (1.0 - phi**2)  # prior variance
rng = np.random.RandomState(23)
x0 = mu + np.sqrt(p0) * rng.randn()
_, true_states, observations = get_data(x0, mu, phi, sigma, nb_steps, random_state=rng)
transition_cov, observation_cov, transition_fn, observation_fn, _, _ = make_parameters(mu, phi, sigma)

transition_model = AdditiveGaussianModel(
    fun=transition_fn,
    noise=Gaussian(jnp.zeros((dim_x,)), transition_cov),
)
observation_model = ConditionalMomentsModel(
    mean_fn=observation_fn,
    cov_fn=observation_cov,
)
prior_dist = Gaussian(
    mean=jnp.array([mu]),
    cov=jnp.array([[p0]]),
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

marginals = iterated_hybrid_markov_smoother(
    observations=jnp.array(observations),
    log_prior_fn=log_prior_fn,
    log_transition_fn=log_transition_fn,
    log_observation_fn=log_observation_fn,
    init_forward_posterior=init_fwd_posterior,
    init_reverse_posterior=init_rvs_posterior,
    kl_constraint=10,
    init_temperature=1e6,
)

ts = np.arange(nb_steps + 1)
mean = np.asarray(marginals.mean)[:, 0]
std = np.sqrt(np.asarray(marginals.cov)[:, 0, 0])

plt.figure(figsize=(10, 4))
plt.fill_between(ts, mean - 2 * std, mean + 2 * std, alpha=0.2, label=r"$\pm 2\sigma$")
plt.plot(ts, mean, "-", label="Smoothed")
plt.plot(ts, np.asarray(true_states)[:, 0], "--", label="True")
plt.xlabel("time step")
plt.ylabel(r"log-volatility $x_k$")
plt.grid()
plt.legend()
plt.show()
