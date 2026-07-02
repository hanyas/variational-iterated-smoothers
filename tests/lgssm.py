from jax import Array
import jax.numpy as jnp
import numpy as np


def transition_fn(x, A):
    """Apply the linear transition map A to the state x.

    Args:
        x: Array
            State vector of shape (dim_x,).
        A: Array
            Transition matrix of shape (dim_x, dim_x).

    Returns:
        Array
            The propagated state A x of shape (dim_x,).
    """
    return jnp.dot(A, x)


def observation_fn(x, H):
    """Apply the linear observation map H to the state x.

    Args:
        x: Array
            State vector of shape (dim_x,).
        H: Array
            Observation matrix of shape (dim_y, dim_x).

    Returns:
        Array
            The noise-free observation H x of shape (dim_y,).
    """
    return jnp.dot(H, x)


def simulate(
    x0: Array,
    A: Array,
    b: Array,
    Omega: Array,
    H: Array,
    e: Array,
    Delta: Array,
    num_steps: int,
    random_state=None,
):
    """Simulate a trajectory from a linear-Gaussian state-space model.

    Rolls the affine transition x -> A x + b with Gaussian noise of covariance
    Omega forward from x0, emitting an affine observation H x + e with Gaussian
    noise of covariance Delta at each of the num_steps states after the initial
    one.

    Args:
        x0: Array
            Initial state of shape (dim_x,).
        A: Array
            Transition matrix of shape (dim_x, dim_x).
        b: Array
            Transition offset of shape (dim_x,).
        Omega: Array
            Transition noise covariance of shape (dim_x, dim_x).
        H: Array
            Observation matrix of shape (dim_y, dim_x).
        e: Array
            Observation offset of shape (dim_y,).
        Delta: Array
            Observation noise covariance of shape (dim_y, dim_y).
        num_steps: int
            Number of transition/observation steps to simulate.
        random_state: int or np.random.RandomState or None
            Seed or NumPy random state used to draw the noise; an int or None
            is promoted to a fresh np.random.RandomState.

    Returns:
        true_states: Array
            Simulated states of shape (num_steps + 1, dim_x), including x0.
        observations: Array
            Simulated observations of shape (num_steps, dim_y).
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    dim_x = Omega.shape[0]
    dim_y = Delta.shape[0]

    normals = random_state.randn(num_steps, dim_x + dim_y).astype(np.float32)

    x = np.copy(x0).astype(np.float32)
    observations = np.empty((num_steps, dim_y), dtype=np.float32)
    true_states = np.empty((num_steps + 1, dim_x), dtype=np.float32)
    true_states[0] = x

    for i in range(num_steps):
        x = A @ x + np.linalg.cholesky(Omega) @ normals[i, :dim_x] + b
        true_states[i + 1] = x
        y = H @ x + np.linalg.cholesky(Delta) @ normals[i, dim_x:] + e
        observations[i] = y

    return true_states, observations
