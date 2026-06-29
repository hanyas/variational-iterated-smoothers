"""Stochastic-volatility environment: AR(1) log-volatility, zero-mean state-dependent observation.

Unlike the additive-Gaussian environments, the observation here is not an additive-noise mean
function: y_t | x_t ~ N(0, exp(x_t)). All the information about the state lives in the
observation *variance*, so `make_parameters` returns the conditional moments of the observation
(a zero mean function and a state-dependent covariance function) in place of a mean function and a
constant covariance matrix.
"""

import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np

__all__ = ["make_parameters", "get_data"]


def _transition_function(x, mu, phi):
    """Deterministic transition function used in the state space model
    Parameters
    ----------
    x: array_like
        The current state
    mu: float
        Long-run mean log-volatility
    phi: float
        Autoregressive persistence
    Returns
    -------
    out: array_like
        The transitioned state
    """
    return mu + phi * (x - mu)


def _transition_function_dx(x, mu, phi):
    return phi * jnp.eye(1)


def _observation_mean_function(x):
    """
    Returns the (zero) conditional mean of the stochastic-volatility observation
    Parameters
    ----------
    x: array_like
        The current state
    Returns
    -------
    out: array_like
        The conditional mean E[y | x] = 0
    """
    return jnp.zeros(1)


def _observation_mean_function_dx(x):
    return jnp.zeros((1, 1))


def _observation_covariance_function(x):
    """
    Returns the state-dependent conditional covariance Cov[y | x] = exp(x)
    Parameters
    ----------
    x: array_like
        The current state
    Returns
    -------
    out: array_like
        The conditional covariance matrix diag(exp(x))
    """
    return jnp.diag(jnp.exp(x))


def make_parameters(mu, phi, sigma):
    """Builds the univariate stochastic-volatility model.
    The latent log-volatility is a stationary AR(1) process and the sensor is zero-mean with a
    state-dependent variance:
        x_t | x_{t-1} ~ N(mu + phi (x_{t-1} - mu), Q),  Q = sigma ** 2
        y_t   | x_t   ~ N(0, exp(x_t))
    Parameters
    ----------
    mu: float
        Long-run mean log-volatility
    phi: float
        Autoregressive persistence
    sigma: float
        Transition (log-volatility) noise standard deviation
    Returns
    -------
    Q: array_like
        The transition covariance matrix
    observation_covariance_function: callable
        The state-dependent observation covariance (replaces the constant R of additive models)
    transition_function: callable
        The transition function
    observation_mean_function: callable
        The (zero) observation mean function (replaces the observation function of additive models)
    transition_function_dx: callable
        The derivative of transition function
    observation_mean_function_dx: callable
        The derivative of the observation mean function
    """

    Q = jnp.array([[sigma**2]])

    transition_function = Partial(_transition_function, mu=mu, phi=phi)
    transition_function_dx = Partial(_transition_function_dx, mu=mu, phi=phi)

    return (
        Q,
        _observation_covariance_function,
        transition_function,
        _observation_mean_function,
        transition_function_dx,
        _observation_mean_function_dx,
    )


def _get_data(x, mu, phi, sigma, state_noise, obs_noise, observations, true_states):
    for i in range(state_noise.shape[0]):
        x = mu + phi * (x - mu) + sigma * state_noise[i]
        observations[i] = np.exp(0.5 * x) * obs_noise[i]
        true_states[i] = x
    return true_states, observations


def get_data(x0, mu, phi, sigma, T, random_state=None):
    """
    Parameters
    ----------
    x0: float
        true initial state
    mu: float
        long-run mean log-volatility
    phi: float
        autoregressive persistence
    sigma: float
        transition noise standard deviation
    T: int
        number of time steps
    random_state: np.random.RandomState or int, optional
        numpy random state
    Returns
    -------
    ts: array_like
        array of time steps
    true_states: array_like
        array of true states
    observations: array_like
        array of observations
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    state_noise = random_state.randn(T).astype(np.float32)
    obs_noise = random_state.randn(T).astype(np.float32)

    x = np.float32(x0)
    observations = np.empty((T, 1), dtype=np.float32)
    true_states = np.zeros((T + 1, 1), dtype=np.float32)
    ts = np.arange(1, T + 1).astype(np.float32)
    true_states[0, 0] = x

    _get_data(x, mu, phi, sigma, state_noise, obs_noise, observations, true_states[1:])
    return ts, true_states, observations
