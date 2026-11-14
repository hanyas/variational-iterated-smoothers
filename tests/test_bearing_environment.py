"""Matched nonlinear turn simulation and derivatives at zero angular velocity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from varsmooth.environments import bearing_only as environment

jax.config.update("jax_enable_x64", True)


def test_simulator_replays_full_inference_noise():
    """Check simulation with the full inference noise covariance."""
    parameters = dict(qc=0.01, qw=0.1, r=0.5, dt=0.01, s1=[-1.5, 0.5], s2=[1., 1.])
    initial = np.array([0.1, 0.2, 1., 0., 0.5])
    steps = 2000
    times, states, observations = environment.get_data(initial, T=steps, random_state=17, **parameters)
    Q, R, f, h, *_ = environment.make_parameters(**parameters)
    rng = np.random.RandomState(17)
    process_noise = rng.standard_normal((steps, 5)) @ np.linalg.cholesky(Q).T
    observation_noise = rng.standard_normal((steps, 2)) @ np.linalg.cholesky(R).T
    np.testing.assert_allclose(states[1:] - jax.vmap(f)(states[:-1]), process_noise, atol=2e-14)
    np.testing.assert_allclose(observations - jax.vmap(h)(states[1:]), observation_noise, atol=2e-14)
    np.testing.assert_array_equal(states[0], initial)
    np.testing.assert_array_equal(times, parameters["dt"] * np.arange(1, steps+1))
    assert states.dtype == np.float64 and observations.dtype == np.float64
    assert Q[0, 2] > 0 and Q[1, 3] > 0
    np.testing.assert_array_equal(states, environment.get_data(initial, T=steps, random_state=17, **parameters)[1])
    with pytest.raises(ValueError, match="five"):
        environment.get_data(initial[:4], T=steps, random_state=17, **parameters)


def test_turn_limit_and_first_two_derivatives():
    """Check the zero-turn limit and its first two derivatives."""
    dt = 0.1
    x = jnp.array([1., 2., 0.7, -0.4, 0.])
    f = lambda w: environment._transition_function(x.at[4].set(w), dt)
    expected_first = np.array([0.5*dt**2*x[3], -0.5*dt**2*x[2], dt*x[3], -dt*x[2], 1.])
    expected_second = np.array([-dt**3*x[2]/3, -dt**3*x[3]/3, -dt**2*x[2], -dt**2*x[3], 0.])
    np.testing.assert_allclose(jax.jacfwd(f)(0.), expected_first, atol=1e-14)
    np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(f))(0.), expected_second, atol=1e-14)
    np.testing.assert_allclose(environment._transition_function_dx(x, dt), jax.jacfwd(environment._transition_function)(x, dt))
    for rate in (0., 1e-12, -1e-12, 0.7, -0.7):
        continuous = np.array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., 0., 0., rate], [0., 0., -rate, 0.]])
        np.testing.assert_allclose(f(rate)[:4], expm(continuous*dt) @ x[:4], atol=1e-14)
