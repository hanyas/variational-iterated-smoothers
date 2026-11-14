"""Bearing-only tracking environment: coordinated-turn dynamics, two range-free sensors."""

from jax import jacfwd
from jax import jit
from jax import lax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np

__all__ = ["make_parameters", "get_data"]


def _sinc(angle):
    """Evaluate sin(angle) / angle with smooth derivatives at zero.

    Args:
        angle: array_like
            Angle in radians.

    Returns:
        value: array_like
            Sinc value with the same shape as angle.
    """
    squared = angle**2
    polynomial = 1 + squared * (
        -1 / 6 + squared * (1 / 120 + squared * (-1 / 5040 + squared * (1 / 362880 - squared / 39916800)))
    )
    small = jnp.abs(angle) < 0.1
    safe_angle = jnp.where(small, 1.0, angle)
    return jnp.where(small, polynomial, jnp.sin(safe_angle) / safe_angle)


def _transition_matrix(w, dt):
    """Build the coordinated-turn matrix with its smooth zero-rate limit.

    Args:
        w: float
            Angular turn rate.
        dt: float
            Discretization time step.

    Returns:
        matrix: array_like
            State transition matrix of shape (5, 5).
    """
    angle = w * dt
    coswt, sinwt = jnp.cos(angle), jnp.sin(angle)
    sinwtpw = dt * _sinc(angle)
    coswtopw = -0.5 * dt * angle * _sinc(angle / 2) ** 2
    return jnp.array(
        [
            [1, 0, sinwtpw, -coswtopw, 0],
            [0, 1, coswtopw, sinwtpw, 0],
            [0, 0, coswt, sinwt, 0],
            [0, 0, -sinwt, coswt, 0],
            [0, 0, 0, 0, 1],
        ]
    )


def _transition_function(x, dt):
    """Propagate the state through the deterministic dynamics.

    Args:
        x: array_like
            State vector of shape (5,).
        dt: float
            Discretization time step.

    Returns:
        next_state: array_like
            Next state of shape (5,).
    """
    return _transition_matrix(x[-1], dt) @ x


def _transition_function_dx(x, dt):
    """Evaluate the state Jacobian of the transition.

    Args:
        x: array_like
            State vector of shape (5,).
        dt: float
            Discretization time step.

    Returns:
        jacobian: array_like
            Transition Jacobian of shape (5, 5).
    """
    return jacfwd(_transition_function, argnums=0)(x, dt)


def _likelihood_function(x, s1, s2):
    """Map the state to its noiseless observation.

    Args:
        x: array_like
            State vector of shape (5,).
        s1: array_like
            First sensor position of shape (2,).
        s2: array_like
            Second sensor position of shape (2,).

    Returns:
        observation: array_like
            Noiseless observation vector.
    """
    return jnp.array(
        [
            jnp.arctan2(x[1] - s1[1], x[0] - s1[0]),
            jnp.arctan2(x[1] - s2[1], x[0] - s2[0]),
        ]
    )


def _likelihood_function_dx(x, s1, s2):
    """Evaluate the state Jacobian of the observation.

    Args:
        x: array_like
            State vector of shape (5,).
        s1: array_like
            First sensor position of shape (2,).
        s2: array_like
            Second sensor position of shape (2,).

    Returns:
        jacobian: array_like
            Observation Jacobian of shape (dy, 5).
    """
    return jnp.array(
        [
            [
                -(x[1] - s1[1]) / ((x[0] - s1[0]) ** 2 + (x[1] - s1[1]) ** 2),
                (x[0] - s1[0]) / ((x[0] - s1[0]) ** 2 + (x[1] - s1[1]) ** 2),
                0,
                0,
                0,
            ],
            [
                -(x[1] - s2[1]) / ((x[0] - s2[0]) ** 2 + (x[1] - s2[1]) ** 2),
                (x[0] - s2[0]) / ((x[0] - s2[0]) ** 2 + (x[1] - s2[1]) ** 2),
                0,
                0,
                0,
            ],
        ]
    )


@Partial(jnp.vectorize, excluded=(1, 2), signature="(m)->(d)")
def inverse_bearings(observation, s1, s2):
    """Recover the noiseless position from two bearing observations.

    Args:
        observation: array_like
            Bearing observations of shape (2,).
        s1: array_like
            First sensor position of shape (2,).
        s2: array_like
            Second sensor position of shape (2,).

    Returns:
        position: array_like
            Recovered position of shape (2,).
    """
    tan_theta = jnp.tan(observation)
    A = jnp.array([[tan_theta[0], -1], [tan_theta[1], -1]])
    b = jnp.array([s1[0] * tan_theta[0] - s1[1], s2[0] * tan_theta[1] - s2[1]])
    return jnp.linalg.solve(A, b)


def make_parameters(qc, qw, r, dt, s1, s2):
    """Build the discrete model, covariances, and Jacobians.

    Args:
        qc: float
            Continuous translational-noise intensity.
        qw: float
            Continuous turn-rate-noise intensity.
        r: float
            Observation noise standard deviation.
        dt: float
            Discretization time step.
        s1: array_like
            First sensor position of shape (2,).
        s2: array_like
            Second sensor position of shape (2,).

    Returns:
        Q: array_like
            Discrete process covariance.
        R: array_like
            Observation covariance.
        transition_function: callable
            Deterministic transition function.
        likelihood_function: callable
            Observation function.
        transition_function_dx: callable
            State Jacobian of the transition.
        likelihood_function_dx: callable
            State Jacobian of the observation.
    """

    Q = jnp.array(
        [
            [qc * dt**3 / 3, 0, qc * dt**2 / 2, 0, 0],
            [0, qc * dt**3 / 3, 0, qc * dt**2 / 2, 0],
            [qc * dt**2 / 2, 0, qc * dt, 0, 0],
            [0, qc * dt**2 / 2, 0, qc * dt, 0],
            [0, 0, 0, 0, dt * qw],
        ]
    )

    R = r**2 * jnp.eye(2)

    likelihood_function = Partial(_likelihood_function, s1=s1, s2=s2)
    transition_function = Partial(_transition_function, dt=dt)
    likelihood_function_dx = Partial(_likelihood_function_dx, s1=s1, s2=s2)
    transition_function_dx = Partial(_transition_function_dx, dt=dt)

    return (
        Q,
        R,
        transition_function,
        likelihood_function,
        transition_function_dx,
        likelihood_function_dx,
    )


def get_data(x0, dt, r, T, s1, s2, *, qc, qw, random_state=None):
    """Simulate the discrete coordinated-turn model with correlated process noise.

    Args:
        x0: array_like
            Initial state vector of shape (5,).
        dt: float
            Discretization time step.
        r: float
            Observation noise standard deviation.
        T: int
            Number of transitions.
        s1: array_like
            First sensor position of shape (2,).
        s2: array_like
            Second sensor position of shape (2,).
        qc: float
            Continuous translational-noise intensity.
        qw: float
            Continuous turn-rate-noise intensity.
        random_state: numpy.random.RandomState, numpy.random.Generator or int, optional
            Random generator or seed.

    Returns:
        times: array_like
            Observation times of shape (T,).
        states: array_like
            True states including the initial state, shape (T + 1, 5).
        observations: array_like
            Simulated observations of shape (T, 2).
    """
    if random_state is None or isinstance(random_state, (int, np.integer)):
        random_state = np.random.RandomState(random_state)
    x0 = np.asarray(x0, dtype=float)
    if x0.shape != (5,) or not np.all(np.isfinite(x0)):
        raise ValueError("x0 must contain five finite state components, including turn rate")
    if not isinstance(T, (int, np.integer)) or T < 1:
        raise ValueError("T must be a positive integer")
    if any(not np.isfinite(v) or v <= 0 for v in (dt, qc, qw, r)):
        raise ValueError("dt, qc, qw, and r must be finite and positive")
    Q, R, f, h, *_ = make_parameters(qc, qw, r, dt, s1, s2)
    ws = random_state.standard_normal((T, 5)) @ np.linalg.cholesky(np.asarray(Q)).T
    vs = random_state.standard_normal((T, 2)) @ np.linalg.cholesky(np.asarray(R)).T

    @jit
    def simulate(initial, process_noise, observation_noise):
        """Simulate the trajectory using precomputed noise draws.

        Args:
            initial: array_like
                Initial state of shape (5,).
            process_noise: array_like
                Process-noise draws of shape (T, 5).
            observation_noise: array_like
                Observation-noise draws of shape (T, 2).

        Returns:
            states: array_like
                States after each transition, shape (T, 5).
            observations: array_like
                Noisy bearings of shape (T, 2).
        """

        def step(x, noises):
            """Apply one noisy state transition and bearing observation.

            Args:
                x: array_like
                    State vector of shape (5,).
                noises: tuple
                    Process and observation noise for the current step.

            Returns:
                next_state: array_like
                    Updated state of shape (5,).
                output: tuple
                    Updated state and noisy bearing observation.
            """
            w, v = noises
            next_x = f(x) + w
            return next_x, (next_x, h(next_x) + v)

        return lax.scan(step, initial, (process_noise, observation_noise))[1]

    states, observations = simulate(jnp.asarray(x0), jnp.asarray(ws), jnp.asarray(vs))
    return (dt * np.arange(1, T + 1), np.concatenate([x0[None], np.asarray(states)]), np.asarray(observations))
