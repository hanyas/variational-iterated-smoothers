"""Cubic sensor environment (Katayama, 2013): stationary AR(1) state, cubic observation."""

import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np

__all__ = ["make_parameters", "get_data"]


def _transition_function(x, phi0, mu0):
    """Deterministic transition function used in the state space model.

    Args:
        x: array_like
            The current state
        phi0: float
            Autoregressive coefficient
        mu0: float
            Long-run mean of the latent state

    Returns:
        array_like
            The transitioned state
    """
    return phi0 * x + (1.0 - phi0) * mu0


def _transition_function_dx(x, phi0, mu0):
    return phi0 * jnp.eye(1)


def _observation_function(x, beta):
    """Returns the cubic sensor observation as a function of the state.

    Args:
        x: array_like
            The current state
        beta: float
            The observation gain

    Returns:
        array_like
            The cubic observation beta * x ** 3
    """
    return beta * x**3


def _observation_function_dx(x, beta):
    return jnp.array([[3.0 * beta * x[0] ** 2]])


def make_parameters(phi0, mu0, sigma0, beta, r):
    """Builds the cubic sensor model of (Katayama, 2013).

    The latent state is a stationary scalar AR(1) process and the sensor is cubic:
        x_t | x_{t-1} ~ N(phi0 x_{t-1} + (1 - phi0) mu0, Q),  Q = (1 - phi0 ** 2) sigma0
        y_t   | x_t   ~ N(beta x_t ** 3, R),                  R = r ** 2

    Args:
        phi0: float
            Autoregressive coefficient
        mu0: float
            Long-run mean of the latent state
        sigma0: float
            Stationary (prior) variance of the latent state
        beta: float
            Observation gain
        r: float
            Observation error standard deviation

    Returns:
        Q: array_like
            The transition covariance matrix
        R: array_like
            The observation covariance matrix
        transition_function: callable
            The transition function
        observation_function: callable
            The observation function
        transition_function_dx: callable
            The derivative of transition function
        observation_function_dx: callable
            The derivative of observation function
    """

    Q = jnp.array([[(1.0 - phi0**2) * sigma0]])
    R = jnp.array([[r**2]])

    transition_function = Partial(_transition_function, phi0=phi0, mu0=mu0)
    observation_function = Partial(_observation_function, beta=beta)
    transition_function_dx = Partial(_transition_function_dx, phi0=phi0, mu0=mu0)
    observation_function_dx = Partial(_observation_function_dx, beta=beta)

    return (
        Q,
        R,
        transition_function,
        observation_function,
        transition_function_dx,
        observation_function_dx,
    )


def _get_data(x, phi0, mu0, beta, sq, r, state_noise, obs_noise, observations, true_states):
    for i in range(state_noise.shape[0]):
        x = phi0 * x + (1.0 - phi0) * mu0 + sq * state_noise[i]
        observations[i] = beta * x**3 + r * obs_noise[i]
        true_states[i] = x
    return true_states, observations


def get_data(x0, phi0, mu0, sigma0, beta, r, T, random_state=None):
    """Simulate an AR(1) trajectory and its cubic-sensor observations.

    Args:
        x0: float
            true initial state
        phi0: float
            autoregressive coefficient
        mu0: float
            long-run mean of the latent state
        sigma0: float
            stationary (prior) variance of the latent state
        beta: float
            observation gain
        r: float
            observation model standard deviation
        T: int
            number of time steps
        random_state: np.random.RandomState or int, optional
            numpy random state

    Returns:
        ts: array_like
            array of time steps
        true_states: array_like
            array of true states
        observations: array_like
            array of observations
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    sq = np.sqrt((1.0 - phi0**2) * sigma0)
    state_noise = random_state.randn(T).astype(np.float32)
    obs_noise = random_state.randn(T).astype(np.float32)

    x = np.float32(x0)
    observations = np.empty((T, 1), dtype=np.float32)
    true_states = np.zeros((T + 1, 1), dtype=np.float32)
    ts = np.arange(1, T + 1).astype(np.float32)
    true_states[0, 0] = x

    _get_data(x, phi0, mu0, beta, sq, r, state_noise, obs_noise, observations, true_states[1:])
    return ts, true_states, observations
