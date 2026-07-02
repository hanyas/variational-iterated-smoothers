"""Linear-Gaussian environment: stable linear dynamics and a linear observation with Gaussian noise."""

import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np

__all__ = ["make_parameters", "get_data", "make_random_system"]


def _transition_function(x, A, b):
    """Deterministic transition function used in the state space model.

    Args:
        x: array_like
            The current state
        A: array_like
            The transition matrix
        b: array_like
            The transition offset

    Returns:
        array_like
            The transitioned state
    """
    return A @ x + b


def _transition_function_dx(x, A, b):
    return A


def _likelihood_function(x, H, e):
    """Returns the linear observation as a function of the state.

    Args:
        x: array_like
            The current state
        H: array_like
            The observation matrix
        e: array_like
            The observation offset

    Returns:
        array_like
            The linear observation H x + e
    """
    return H @ x + e


def _likelihood_function_dx(x, H, e):
    return H


def _random_spd(dim, random_state, scale=1.0):
    chol = random_state.rand(dim, dim)
    chol[np.triu_indices(dim, 1)] = 0.0
    return scale * (chol @ chol.T)


def make_random_system(dim_x, dim_y, random_state=None, transition_scale=0.9):
    """Draw a stable linear-Gaussian system: a scale-shrunk identity transition with random offsets,
    noise covariances and prior.

    Args:
        dim_x: int
            State dimension
        dim_y: int
            Observation dimension
        random_state: np.random.RandomState or int, optional
            numpy random state
        transition_scale: float
            Scaling of the (identity) transition and observation matrices

    Returns:
        mu0, P0: array_like
            The prior mean and covariance
        A, b, Omega: array_like
            The transition matrix, offset and covariance
        H, e, Delta: array_like
            The observation matrix, offset and covariance
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    mu0 = random_state.randn(dim_x)
    P0 = _random_spd(dim_x, random_state)
    A = transition_scale * np.eye(dim_x)
    b = random_state.randn(dim_x)
    Omega = _random_spd(dim_x, random_state)
    H = transition_scale * np.eye(dim_y, dim_x)
    e = random_state.randn(dim_y)
    Delta = _random_spd(dim_y, random_state)
    return mu0, P0, A, b, Omega, H, e, Delta


def make_parameters(A, b, Omega, H, e, Delta):
    """Wraps a linear-Gaussian system into transition / observation functions.

        x_t | x_{t-1} ~ N(A x_{t-1} + b, Omega)
        y_t   | x_t   ~ N(H x_t + e, Delta)

    Args:
        A, b, Omega: array_like
            The transition matrix, offset and covariance
        H, e, Delta: array_like
            The observation matrix, offset and covariance

    Returns:
        Q: array_like
            The transition covariance matrix (Omega)
        R: array_like
            The observation covariance matrix (Delta)
        transition_function: callable
            The transition function
        likelihood_function: callable
            The observation function
        transition_function_dx: callable
            The derivative of transition function
        likelihood_function_dx: callable
            The derivative of observation function
    """
    Q = jnp.asarray(Omega)
    R = jnp.asarray(Delta)

    transition_function = Partial(_transition_function, A=jnp.asarray(A), b=jnp.asarray(b))
    likelihood_function = Partial(_likelihood_function, H=jnp.asarray(H), e=jnp.asarray(e))
    transition_function_dx = Partial(_transition_function_dx, A=jnp.asarray(A), b=jnp.asarray(b))
    likelihood_function_dx = Partial(_likelihood_function_dx, H=jnp.asarray(H), e=jnp.asarray(e))

    return (
        Q,
        R,
        transition_function,
        likelihood_function,
        transition_function_dx,
        likelihood_function_dx,
    )


def _get_data(x, A, b, chol_Omega, H, e, chol_Delta, state_noise, obs_noise, observations, true_states):
    for i in range(state_noise.shape[0]):
        x = A @ x + b + chol_Omega @ state_noise[i]
        observations[i] = H @ x + e + chol_Delta @ obs_noise[i]
        true_states[i] = x
    return true_states, observations


def get_data(x0, A, b, Omega, H, e, Delta, T, random_state=None):
    """Simulate a linear-Gaussian trajectory and its observations.

    Args:
        x0: array_like
            true initial state
        A, b, Omega: array_like
            The transition matrix, offset and covariance
        H, e, Delta: array_like
            The observation matrix, offset and covariance
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
    A, b, Omega = np.asarray(A), np.asarray(b), np.asarray(Omega)
    H, e, Delta = np.asarray(H), np.asarray(e), np.asarray(Delta)
    dim_x, dim_y = A.shape[0], H.shape[0]
    chol_Omega = np.linalg.cholesky(Omega)
    chol_Delta = np.linalg.cholesky(Delta)
    state_noise = random_state.randn(T, dim_x)
    obs_noise = random_state.randn(T, dim_y)

    x = np.copy(np.asarray(x0))
    observations = np.empty((T, dim_y))
    true_states = np.zeros((T + 1, dim_x))
    ts = np.arange(1, T + 1)
    true_states[0] = x

    _get_data(x, A, b, chol_Omega, H, e, chol_Delta, state_noise, obs_noise, observations, true_states[1:])
    return ts, true_states, observations
