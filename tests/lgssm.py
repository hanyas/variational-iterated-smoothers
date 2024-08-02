import jax.numpy as jnp
import numpy as np


def transition_fn(x, A):
    return jnp.dot(A, x)


def observation_fn(x, H):
    return jnp.dot(H, x)


def simulate(
    x0,
    A, b, Omega,
    H, e, Delta,
    nb_steps,
    random_state=None,
):
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    dim_x = Omega.shape[0]
    dim_y = Delta.shape[0]

    normals = random_state.randn(nb_steps, dim_x + dim_y).astype(np.float32)

    x = np.copy(x0).astype(np.float32)
    observations = np.empty((nb_steps, dim_y), dtype=np.float32)
    true_states = np.empty((nb_steps + 1, dim_x), dtype=np.float32)
    true_states[0] = x

    for i in range(nb_steps):
        x = A @ x + np.linalg.cholesky(Omega) @ normals[i, :dim_x] + b
        true_states[i + 1] = x
        y = H @ x + np.linalg.cholesky(Delta) @ normals[i, dim_x:] + e
        observations[i] = y

    return true_states, observations
