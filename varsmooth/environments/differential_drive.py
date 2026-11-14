"""Differential-drive robot with pose observations (Courts et al., 2021, Eq. 40)."""

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
import numpy as np

__all__ = ["make_parameters", "get_data", "expected_transition"]


def _transition_function(x, u, dt, m, J, l, r1, r2, a):
    """Propagate the state through the deterministic dynamics.

    Args:
        x: array_like
            State vector of shape (5,).
        u: array_like
            Left and right wheel forces of shape (2,).
        dt: float
            Discretization time step.
        m: float
            Robot mass.
        J: float
            Robot moment of inertia.
        l: float
            Offset of the center of mass.
        r1: float
            Linear damping coefficient.
        r2: float
            Angular damping coefficient.
        a: float
            Wheel-force lever arm.

    Returns:
        next_state: array_like
            Next state of shape (5,).
    """
    inertia = J + m * l**2
    theta, p1, p2 = x[2], x[3], x[4]
    drift = jnp.array(
        [
            jnp.cos(theta) * p1 / m,
            jnp.sin(theta) * p1 / m,
            p2 / inertia,
            -r1 * p1 / m - m * l * p2**2 / inertia**2 + u[0] + u[1],
            (l * p1 - r2) * p2 / inertia + a * (u[0] - u[1]),
        ]
    )
    return x + dt * drift


def _transition_function_dx(x, u, dt, m, J, l, r1, r2, a):
    """Evaluate the state Jacobian of the transition.

    Args:
        x: array_like
            State vector of shape (5,).
        u: array_like
            Left and right wheel forces of shape (2,).
        dt: float
            Discretization time step.
        m: float
            Robot mass.
        J: float
            Robot moment of inertia.
        l: float
            Offset of the center of mass.
        r1: float
            Linear damping coefficient.
        r2: float
            Angular damping coefficient.
        a: float
            Wheel-force lever arm.

    Returns:
        jacobian: array_like
            Transition Jacobian of shape (5, 5).
    """
    return jax.jacfwd(_transition_function, argnums=0)(x, u, dt, m, J, l, r1, r2, a)


def _likelihood_function(x):
    """Map the state to its noiseless observation.

    Args:
        x: array_like
            State vector of shape (5,).

    Returns:
        observation: array_like
            Noiseless observation vector.
    """
    return x[:3]


def _likelihood_function_dx(x):
    """Evaluate the state Jacobian of the observation.

    Args:
        x: array_like
            State vector of shape (5,).

    Returns:
        jacobian: array_like
            Observation Jacobian of shape (dy, 5).
    """
    return jnp.eye(3, 5, dtype=x.dtype)


def make_parameters(dt=0.1, m=5.0, J=2.0, l=0.15, r1=1.0, r2=1.0, a=0.5, Q=None, R=None):
    """Build the discrete model, covariances, and Jacobians.

    Args:
        dt: float
            Discretization time step.
        m: float
            Robot mass.
        J: float
            Robot moment of inertia.
        l: float
            Offset of the center of mass.
        r1: float
            Linear damping coefficient.
        r2: float
            Angular damping coefficient.
        a: float
            Wheel-force lever arm.
        Q: array_like, optional
            Discrete process covariance of shape (5, 5).
        R: array_like, optional
            Observation covariance of shape (3, 3).

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
    if dt <= 0 or m <= 0 or J <= 0:
        raise ValueError("dt, m and J must be positive")
    Q = jnp.asarray(np.diag([0.001, 0.001, 1.745e-3, 0.001, 0.001]) if Q is None else Q)
    R = jnp.asarray(np.diag([0.1**2, 0.1**2, 0.0349**2]) if R is None else R)
    if Q.shape != (5, 5) or R.shape != (3, 3):
        raise ValueError("Q must have shape (5, 5) and R must have shape (3, 3)")
    kwargs = dict(dt=dt, m=m, J=J, l=l, r1=r1, r2=r2, a=a)
    return (
        Q,
        R,
        Partial(_transition_function, **kwargs),
        _likelihood_function,
        Partial(_transition_function_dx, **kwargs),
        _likelihood_function_dx,
    )


def get_data(x0, controls, dt=0.1, m=5.0, J=2.0, l=0.15, r1=1.0, r2=1.0, a=0.5, Q=None, R=None, random_state=None):
    """Simulate Euler robot dynamics and noisy unwrapped pose observations.

    Args:
        x0: array_like
            Initial state vector of shape (5,).
        controls: array_like
            Wheel-force inputs of shape (T, 2).
        dt: float
            Discretization time step.
        m: float
            Robot mass.
        J: float
            Robot moment of inertia.
        l: float
            Offset of the center of mass.
        r1: float
            Linear damping coefficient.
        r2: float
            Angular damping coefficient.
        a: float
            Wheel-force lever arm.
        Q: array_like, optional
            Discrete process covariance of shape (5, 5).
        R: array_like, optional
            Observation covariance of shape (3, 3).
        random_state: numpy.random.RandomState, numpy.random.Generator or int, optional
            Random generator or seed.

    Returns:
        times: array_like
            Observation times of shape (T,).
        states: array_like
            True states including the initial state, shape (T + 1, 5).
        observations: array_like
            Simulated observations of shape (T, 3).
    """
    if random_state is None or isinstance(random_state, (int, np.integer)):
        random_state = np.random.RandomState(random_state)
    controls = np.asarray(controls, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    if controls.ndim != 2 or controls.shape[1] != 2 or x0.shape != (5,):
        raise ValueError("controls must have shape (T, 2) and x0 must have shape (5,)")
    Q, R, transition, observation, _, _ = make_parameters(dt, m, J, l, r1, r2, a, Q, R)
    state_noise = random_state.standard_normal((len(controls), 5)) @ np.linalg.cholesky(Q).T
    obs_noise = random_state.standard_normal((len(controls), 3)) @ np.linalg.cholesky(R).T

    @jax.jit
    def simulate(x, us, ws, vs):
        """Simulate the controlled trajectory using precomputed noise draws.

        Args:
            x: array_like
                State vector of shape (5,).
            us: array_like
                Wheel-force inputs of shape (T, 2).
            ws: array_like
                Process-noise draws of shape (T, 5).
            vs: array_like
                Observation-noise draws of shape (T, 3).

        Returns:
            states: array_like
                States after each transition, shape (T, 5).
            observations: array_like
                Noisy poses of shape (T, 3).
        """

        def step(x, inputs):
            """Apply one controlled transition and noisy pose observation.

            Args:
                x: array_like
                    State vector of shape (5,).
                inputs: tuple
                    Wheel forces, process noise, and observation noise.

            Returns:
                next_state: array_like
                    Updated state of shape (5,).
                output: tuple
                    Updated state and noisy pose observation.
            """
            u, w, v = inputs
            x = transition(x, u) + w
            return x, (x, observation(x) + v)

        return jax.lax.scan(step, x, (us, ws, vs))[1]

    states, observations = simulate(
        jnp.asarray(x0), jnp.asarray(controls), jnp.asarray(state_noise), jnp.asarray(obs_noise)
    )
    return (
        dt * np.arange(1, len(controls) + 1),
        np.concatenate([x0[None], np.asarray(states)]),
        np.asarray(observations),
    )


def _product_trig(mi, mj, cij, cit, cjt, theta, vt, harmonic=1):
    """Evaluate Gaussian second-order moments multiplied by sine and cosine.

    Args:
        mi: float
            Mean of Xi.
        mj: float
            Mean of Xj.
        cij: float
            Covariance between Xi and Xj.
        cit: float
            Covariance between Xi and the heading.
        cjt: float
            Covariance between Xj and the heading.
        theta: float
            Mean heading in radians.
        vt: float
            Heading variance.
        harmonic: int
            Trigonometric harmonic number.

    Returns:
        cosine_moment: float
            Expectation of Xi Xj cos(harmonic * theta).
        sine_moment: float
            Expectation of Xi Xj sin(harmonic * theta).
    """
    h = harmonic
    real = mi * mj + cij - h**2 * cit * cjt
    imag = h * (mi * cjt + mj * cit)
    attenuation = jnp.exp(-0.5 * h**2 * vt)
    cosine, sine = jnp.cos(h * theta), jnp.sin(h * theta)
    return attenuation * (real * cosine - imag * sine), attenuation * (real * sine + imag * cosine)


def expected_transition(mean, cov, control, Q, dt=0.1, m=5.0, J=2.0, l=0.15, r1=1.0, r2=1.0, a=0.5):
    """Evaluate the exact Gaussian expected robot transition log density.

    Args:
        mean: array_like
            Adjacent-state mean in (previous, next) order, shape (10,).
        cov: array_like
            Adjacent-state covariance of shape (10, 10).
        control: array_like
            Left and right wheel forces of shape (2,).
        Q: array_like
            Diagonal discrete process covariance of shape (5, 5).
        dt: float
            Discretization time step.
        m: float
            Robot mass.
        J: float
            Robot moment of inertia.
        l: float
            Offset of the center of mass.
        r1: float
            Linear damping coefficient.
        r2: float
            Angular damping coefficient.
        a: float
            Wheel-force lever arm.

    Returns:
        value: float
            Expected log density.
    """
    P = 0.5 * (cov + cov.T)
    inertia = J + m * l**2
    # Each row is the linear part of a transition residual in (prev, next).
    W = jnp.concatenate([-jnp.eye(5), jnp.eye(5)], axis=1)
    W = W.at[2, 4].set(-dt / inertia)
    W = W.at[3, 3].set(-1 + dt * r1 / m)
    W = W.at[4, 4].set(-1 + dt * r2 / inertia)
    offset = jnp.array([0.0, 0.0, 0.0, -dt * (control[0] + control[1]), -dt * a * (control[0] - control[1])])
    am = W @ mean + offset
    cross = W @ P
    second = am**2 + jnp.einsum("ij,ij->i", cross, W)
    p, s, theta = mean[3], mean[4], mean[2]
    vp, vs, cps = P[3, 3], P[4, 4], P[3, 4]
    p_squared = p**2 + vp
    p2_cos2, _ = _product_trig(p, p, vp, P[3, 2], P[3, 2], theta, P[2, 2], 2)
    ax_p_cos, _ = _product_trig(am[0], p, cross[0, 3], cross[0, 2], P[3, 2], theta, P[2, 2])
    _, ay_p_sin = _product_trig(am[1], p, cross[1, 3], cross[1, 2], P[3, 2], theta, P[2, 2])
    b = dt / m
    second = second.at[0].add(0.5 * b**2 * (p_squared + p2_cos2) - 2 * b * ax_p_cos)
    second = second.at[1].add(0.5 * b**2 * (p_squared - p2_cos2) - 2 * b * ay_p_sin)
    c = dt * m * l / inertia**2
    es4 = s**4 + 6 * s**2 * vs + 3 * vs**2
    ea_s2 = am[3] * (s**2 + vs) + 2 * cross[3, 4] * s
    second = second.at[3].add(2 * c * ea_s2 + c**2 * es4)
    c = -dt * l / inertia
    ep2s2 = p_squared * (s**2 + vs) + 4 * p * s * cps + 2 * cps**2
    ea_ps = am[4] * (p * s + cps) + cross[4, 3] * s + cross[4, 4] * p
    second = second.at[4].add(2 * c * ea_ps + c**2 * ep2s2)
    variance = jnp.diag(Q)
    return -0.5 * (5 * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(variance)) + jnp.sum(second / variance))
