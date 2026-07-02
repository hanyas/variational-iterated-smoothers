"""Shared harness for the cubic-sensor experiments."""

from collections import namedtuple
from functools import partial
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

from varsmooth.approximation import cubature_linearization
from varsmooth.approximation import cubature_quadratization
from varsmooth.approximation import fourier_hermite as fh
from varsmooth.approximation import gauss_hermite_quadratization
from varsmooth.approximation import linearization as gslr
from varsmooth.environments.cubic_sensor import get_data
from varsmooth.environments.cubic_sensor import make_parameters
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import FH_BACKENDS
from common import GSLR_BACKENDS
from common import make_elbo_evaluator as _common_make_elbo_evaluator
from common import nlpd as _common_nlpd
from common import save_fig as _common_save_fig
from common import set_style
from common import write_csv

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

GH5_QUAD = partial(gauss_hermite_quadratization, order=5)  # fixed ELBO-evaluation backend
CUB_QUAD = cubature_quadratization
CUB_LINEAR = cubature_linearization

DIM_X, DIM_Y = 1, 1


# ---- system + data ----------------------------------------------------------
CubicSystem = namedtuple("CubicSystem", ["phi0", "mu0", "Sigma0", "beta", "r", "prior"])


def make_cubic_system(phi0=0.95, mu0=0.4, Sigma0=0.36, beta=1.0, r=1.0):
    """Build a scalar cubic-sensor CubicSystem with a Gaussian prior."""
    prior = Gaussian(jnp.array([mu0]), jnp.array([[Sigma0]]))
    return CubicSystem(phi0, mu0, Sigma0, beta, r, prior)


def simulate_data(system, num_steps, rng):
    """Simulate a trajectory and observations from a CubicSystem.

    Args:
        system: CubicSystem
            The system to simulate.
        num_steps: int
            Number of transitions T.
        rng: np.random.RandomState
            Random state driving the simulation.

    Returns:
        true_states: Array
            Latent trajectory of shape (T + 1, 1).
        observations: Array
            Observations of shape (T, 1).
    """
    x0 = system.mu0 + np.sqrt(system.Sigma0) * rng.randn()
    _, true_states, observations = get_data(
        x0, system.phi0, system.mu0, system.Sigma0, system.beta, system.r, num_steps, rng
    )
    return jnp.asarray(true_states), jnp.asarray(observations)


def make_additive_models(system):
    """Return the additive-Gaussian (transition_model, likelihood_model) for a CubicSystem."""
    Q, R, transition_fn, likelihood_fn, _, _ = make_parameters(
        system.phi0, system.mu0, system.Sigma0, system.beta, system.r
    )
    transition_model = AdditiveGaussianModel(fun=transition_fn, noise=Gaussian(jnp.zeros(DIM_X), Q))
    likelihood_model = AdditiveGaussianModel(fun=likelihood_fn, noise=Gaussian(jnp.zeros(DIM_Y), R))
    return transition_model, likelihood_model


def make_model_fns(system, family, backend):
    """Build the (log_prior, log_transition, log_likelihood) closures for a family/backend.

    Args:
        system: CubicSystem
            The system whose transition and likelihood are expanded.
        family: str
            "GSLR" or "FH".
        backend: str or Callable
            A GSLR_BACKENDS / FH_BACKENDS key, or a linearization/quadratization callable.

    Returns:
        tuple
            The (lp, lt, lo) log-potential builders.
    """
    transition_model, likelihood_model = make_additive_models(system)
    if family == "GSLR":
        method = backend if callable(backend) else GSLR_BACKENDS[backend]
        lp = lambda q: gslr.get_log_prior(system.prior, q, method)
        lt = lambda q, _: gslr.get_log_transition(transition_model, q, method)
        lo = lambda y, q: gslr.get_log_likelihood(y, likelihood_model, q, method)
    elif family == "FH":
        method = backend if callable(backend) else FH_BACKENDS[backend]
        lp = lambda q: fh.get_log_prior(system.prior, q, method)
        lt = lambda q, p: fh.get_log_transition(transition_model, q, p, method)
        lo = lambda y, q: fh.get_log_likelihood(y, likelihood_model, q, method)
    else:
        raise ValueError(f"unknown family {family!r}")
    return lp, lt, lo


def make_prior_chain_init(system, num_steps):
    """Build the stationary-AR(1) forward Gauss-Markov init for a CubicSystem."""
    Q = (1.0 - system.phi0**2) * system.Sigma0
    F = system.phi0 * np.eye(DIM_X)
    d = np.array([(1.0 - system.phi0) * system.mu0])
    Sigma = Q * np.eye(DIM_X)

    return GaussMarkov(
        marginal=system.prior,
        kernels=AffineGaussian(
            F=np.repeat([F], num_steps, axis=0),
            d=np.repeat([d], num_steps, axis=0),
            Sigma=np.repeat([Sigma], num_steps, axis=0),
        ),
    )


def make_elbo_evaluator(system, observations):
    """Build the fixed GH5 Fourier-Hermite ELBO evaluator for a CubicSystem."""
    return _common_make_elbo_evaluator(observations, *make_model_fns(system, "FH", GH5_QUAD))


def make_nlpd_evaluator(true_states):
    """Build an NLPD evaluator marginals -> NLPD of the true trajectory."""
    return lambda marginals: _common_nlpd(marginals, true_states)


def save_fig(fig, name):
    """Save fig into this suite's outputs/ under name (delegates to common.save_fig)."""
    return _common_save_fig(fig, OUTPUT_DIR / name)
