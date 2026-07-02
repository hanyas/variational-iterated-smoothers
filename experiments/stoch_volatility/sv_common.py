"""Shared harness for the stochastic-volatility (SV) experiments."""

from collections import namedtuple
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

from varsmooth.approximation import fourier_hermite as fh
from varsmooth.approximation import linearization as gslr
from varsmooth.environments import stoch_volatility as sv_env
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel
from varsmooth.objects import Gaussian

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FH_BACKENDS
from common import GSLR_BACKENDS
from common import make_forward_init
from common import nlpd
from common import rmse
from common import run_iterated_smoother
from common import save_fig as _common_save_fig
from common import set_style
from common import write_csv

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---- system + data ----------------------------------------------------------
SVSystem = namedtuple("SVSystem", ["mu", "phi", "sigma", "prior"])


def make_sv_system(mu=-0.5, phi=0.98, sigma=0.16):
    """Build a stochastic-volatility SVSystem with a stationary prior."""
    p0 = sigma**2 / (1.0 - phi**2)
    prior = Gaussian(jnp.array([mu]), jnp.array([[p0]]))
    return SVSystem(mu, phi, sigma, prior)


def simulate_data(system, num_steps, rng):
    """Simulate a log-volatility trajectory and observations from an SVSystem.

    Args:
        system: SVSystem
            The system to simulate.
        num_steps: int
            Number of transitions T.
        rng: np.random.RandomState
            Random state driving the simulation.

    Returns:
        true_states: Array
            Latent log-volatility trajectory of shape (T + 1, 1).
        observations: Array
            Observations of shape (T, 1).
    """
    mu, phi, sigma = system.mu, system.phi, system.sigma
    p0 = float(system.prior.cov[0, 0])
    x0 = mu + np.sqrt(p0) * rng.randn()
    _, true_states, observations = sv_env.get_data(x0, mu, phi, sigma, num_steps, rng)
    return jnp.asarray(true_states), jnp.asarray(observations)


# ---- model log-potential builders -------------------------------------------
def make_model_fns(system, family, backend):
    """Build the (log_prior, log_transition, log_likelihood) closures for a family/backend.

    Args:
        system: SVSystem
            The system whose transition and likelihood are expanded.
        family: str
            "GSLR" or "FH".
        backend: str
            Key into GSLR_BACKENDS / FH_BACKENDS selecting the sigma-point rule.

    Returns:
        tuple
            The (lp, lt, lo) log-potential builders.
    """
    Q, cov_fn, transition_function, mean_fn, _, _ = sv_env.make_parameters(system.mu, system.phi, system.sigma)
    transition_model = AdditiveGaussianModel(fun=transition_function, noise=Gaussian(jnp.zeros((1,)), Q))
    likelihood_model = ConditionalMomentsModel(mean_fn=mean_fn, cov_fn=cov_fn)

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


# ---- reporting helpers ------------------------------------------------------
def save_fig(fig, name):
    """Save fig into this suite's outputs/ under name (delegates to common.save_fig)."""
    return _common_save_fig(fig, OUTPUT_DIR / name)
