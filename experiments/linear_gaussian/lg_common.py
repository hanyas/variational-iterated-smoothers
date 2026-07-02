"""Shared harness for the linear-Gaussian (LG) experiments."""

from collections import namedtuple
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np

from varsmooth.approximation import fourier_hermite as fh
from varsmooth.approximation import linearization as gslr
from varsmooth.environments import linear_gaussian as lg_env
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.smoothers.forward_markov import forward_markov_smoother
from varsmooth.smoothers.hybrid_markov import hybrid_markov_smoother
from varsmooth.smoothers.reverse_markov import reverse_markov_smoother
from varsmooth.smoothers.rts_kalman import rts_smoother
from varsmooth.smoothers.utils import initialize_reverse_with_forward

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FH_BACKENDS
from common import GSLR_BACKENDS
from common import avg_kl
from common import get_marginals
from common import get_markov_history
from common import make_forward_init
from common import rmse
from common import run_iterated_smoother
from common import save_fig as _common_save_fig
from common import set_style
from common import write_csv

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---- linear-Gaussian system + data ------------------------------------------
LGSystem = namedtuple("LGSystem", ["prior", "A", "b", "Omega", "H", "e", "Delta"])


def make_lg_system(dim_x, dim_y, rng, transition_scale=0.9):
    """Build a random linear-Gaussian LGSystem of the given dimensions."""
    mu0, P0, A, b, Omega, H, e, Delta = lg_env.make_random_system(dim_x, dim_y, rng, transition_scale)
    return LGSystem(Gaussian(jnp.asarray(mu0), jnp.asarray(P0)), A, b, Omega, H, e, Delta)


def make_linear_system(rho=0.985, theta=0.16, q=0.05, r=0.25, r0=4.0):
    """Build the fixed 2D-rotation LGSystem shared by the LG experiments.

    Args:
        rho: float
            Spectral radius scaling the rotation transition.
        theta: float
            Rotation angle per step.
        q: float
            Process-noise standard deviation.
        r: float
            Observation-noise standard deviation.
        r0: float
            First coordinate of the prior mean.

    Returns:
        LGSystem
            The linear-Gaussian system.
    """
    c, s = np.cos(theta), np.sin(theta)

    A = rho * np.array([[c, -s], [s, c]])
    b = np.zeros(2)
    Omega = (q**2) * np.eye(2)

    H = np.eye(2)
    e = np.zeros(2)
    Delta = (r**2) * np.eye(2)

    prior = Gaussian(jnp.asarray([r0, 0.0]), jnp.asarray(0.01 * np.eye(2)))
    return LGSystem(prior, A, b, Omega, H, e, Delta)


def simulate_data(system, num_steps, rng):
    """Simulate a trajectory and observations from an LGSystem.

    Args:
        system: LGSystem
            The system to simulate.
        num_steps: int
            Number of transitions T.
        rng: np.random.RandomState
            Random state driving the simulation.

    Returns:
        true_states: Array
            Latent trajectory of shape (T + 1, dim_x).
        observations: Array
            Observations of shape (T, dim_y).
    """
    mu0 = np.asarray(system.prior.mean)
    P0 = np.asarray(system.prior.cov)
    x0 = mu0 + np.linalg.cholesky(P0) @ rng.randn(mu0.shape[0])
    _, true_states, observations = lg_env.get_data(
        x0, system.A, system.b, system.Omega, system.H, system.e, system.Delta, num_steps, rng
    )
    return jnp.asarray(true_states), jnp.asarray(observations)


# ---- exact oracles ----------------------------------------------------------
def rts_marginals(system, observations):
    """Return the exact RTS smoother marginals for an LGSystem."""
    num_steps = observations.shape[0]

    linear_transition = AffineGaussian(
        np.repeat([system.A], num_steps, axis=0),
        np.repeat([system.b], num_steps, axis=0),
        np.repeat([system.Omega], num_steps, axis=0),
    )
    linear_likelihood = AffineGaussian(
        np.repeat([system.H], num_steps, axis=0),
        np.repeat([system.e], num_steps, axis=0),
        np.repeat([system.Delta], num_steps, axis=0),
    )
    return rts_smoother(observations, system.prior, linear_transition, linear_likelihood)


def kalman_log_evidence(system, observations):
    """Return the exact Kalman log evidence log p(y_1..y_T) for an LGSystem."""
    A, b, Omega = system.A, system.b, system.Omega
    H, e, Delta = system.H, system.e, system.Delta

    def body(carry, y):
        m, P, ll = carry
        m_pred = A @ m + b
        P_pred = A @ P @ A.T + Omega
        y_hat = H @ m_pred + e
        S = H @ P_pred @ H.T + Delta

        ll = ll + Gaussian(y_hat, S).log_prob(y)

        K = jsc.linalg.solve(S.T, H @ P_pred.T).T
        m = m_pred + K @ (y - y_hat)
        P = P_pred - K @ S @ K.T
        return (m, P, ll), None

    (_, _, ll), _ = jax.lax.scan(body, (system.prior.mean, system.prior.cov, jnp.array(0.0)), observations)
    return float(ll)


# ---- model log-potential builders -------------------------------------------
def make_model_fns(system, family, backend):
    """Build the (log_prior, log_transition, log_likelihood) closures for a family/backend.

    Args:
        system: LGSystem
            The system whose transition and likelihood are expanded.
        family: str
            "GSLR" (generalized statistical linear regression) or "FH" (Fourier-Hermite).
        backend: str
            Key into GSLR_BACKENDS / FH_BACKENDS selecting the sigma-point rule.

    Returns:
        tuple
            The (lp, lt, lo) log-potential builders.
    """
    dim_x = system.A.shape[0]
    dim_y = system.H.shape[0]
    Q, R, transition_function, likelihood_function, _, _ = lg_env.make_parameters(
        system.A, system.b, system.Omega, system.H, system.e, system.Delta
    )
    transition_model = AdditiveGaussianModel(
        fun=transition_function,
        noise=Gaussian(jnp.zeros((dim_x,)), Q),
    )
    likelihood_model = AdditiveGaussianModel(
        fun=likelihood_function,
        noise=Gaussian(jnp.zeros((dim_y,)), R),
    )

    if family == "GSLR":
        method = GSLR_BACKENDS[backend]
        lp = lambda q: gslr.get_log_prior(system.prior, q, method)
        lt = lambda q, _: gslr.get_log_transition(transition_model, q, method)
        lo = lambda y, q: gslr.get_log_likelihood(y, likelihood_model, q, method)
    elif family == "FH":
        method = FH_BACKENDS[backend]
        lp = lambda q: fh.get_log_prior(system.prior, q, method)
        lt = lambda q, p: fh.get_log_transition(transition_model, q, p, method)
        lo = lambda y, q: fh.get_log_likelihood(y, likelihood_model, q, method)
    else:
        raise ValueError(f"unknown family {family!r}")
    return lp, lt, lo


# ---- single-pass smoother runners -------------------------------------------
def run_single_pass(direction, model_fns, observations, init_fwd_posterior, temperature=0.0):
    """Run one damped single pass (forward / reverse / hybrid) from a forward init.

    Args:
        direction: str
            One of "forward", "reverse", "hybrid".
        model_fns: tuple
            The (lp, lt, lo) log-potential builders.
        observations: Array
            Observation sequence of leading shape (T,).
        init_fwd_posterior: GaussMarkov
            Forward Gauss-Markov init; the reverse/hybrid inits are derived from it.
        temperature: float
            Trust-region temperature t; damping = t / (1 + t).

    Returns:
        Gaussian
            The smoothed marginals.
    """
    lp, lt, lo = model_fns
    if direction == "forward":
        res = forward_markov_smoother(observations, lp, lt, lo, init_fwd_posterior, temperature)
    elif direction == "reverse":
        rev_init = initialize_reverse_with_forward(init_fwd_posterior)
        res = reverse_markov_smoother(observations, lp, lt, lo, rev_init, temperature)
    elif direction == "hybrid":
        rev_init = initialize_reverse_with_forward(init_fwd_posterior)
        res = hybrid_markov_smoother(observations, lp, lt, lo, init_fwd_posterior, rev_init, temperature)
    else:
        raise ValueError(direction)
    return get_marginals(direction, res)


# ---- reporting helpers ------------------------------------------------------
def save_fig(fig, name):
    """Save fig into this suite's outputs/ under name (delegates to common.save_fig)."""
    return _common_save_fig(fig, OUTPUT_DIR / name)
