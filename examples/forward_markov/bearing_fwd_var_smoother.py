import jax
import jax.numpy as jnp

from varsmooth.objects import StdGaussian, FunctionalModel
from varsmooth.objects import ForwardGaussMarkov, StdLinearGaussian

from varsmooth.smoothers import forward_markov_iterated_smoother
from varsmooth.linearization import cubature

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

T = 500  # number of observations
nx, ny = 5, 2

_, true_states, observations = get_data(x0, dt, r, T, s1, s2, random_state=1337)
Q, R, trans_fcn, obsrv_fcn, _, _ = make_parameters(qc, qw, r, dt, s1, s2)

trans_mdl = FunctionalModel(trans_fcn, StdGaussian(jnp.zeros((nx,)), Q))
obsrv_mdl = FunctionalModel(obsrv_fcn, StdGaussian(jnp.zeros((ny,)), R))

prior_dist = StdGaussian(
    mean=jnp.array([-1.0, -1.0, 0.0, 0.0, 0.0]), cov=jnp.eye(nx)
)

init_posterior = ForwardGaussMarkov(
    init=StdGaussian(jnp.zeros((nx, )), 10.0 * jnp.eye(nx)),
    kernels=StdLinearGaussian(
        mat=jnp.zeros((T, nx, nx)),
        bias=jnp.zeros((T, nx)),
        cov=jnp.repeat(10.0 * jnp.eye(nx).reshape(1, nx, nx), T, axis=0),
    )
)

gauss_markov, marginals, dampings = forward_markov_iterated_smoother(
    observations,
    prior_dist,
    trans_mdl,
    obsrv_mdl,
    cubature,
    init_posterior,
    max_damping=1e16,
    min_damping=1e-16,
    init_damping=1e6,
    damping_mult=10.0,
    max_iter=1000,
)

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
