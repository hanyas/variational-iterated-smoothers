"""Shared harness for the stochastic-volatility (SV) experiments."""

from collections import namedtuple
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from varsmooth.approximation import fourier_hermite as _fh
from varsmooth.approximation import linearization as _pl
from varsmooth.environments import stoch_volatility as sv_env
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel
from varsmooth.objects import Gaussian

# make `varsmooth` importable regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
from common import FH_BACKENDS
from common import GSLR_BACKENDS

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---- system + data ----------------------------------------------------------
SVSystem = namedtuple("SVSystem", ["mu", "phi", "sigma", "prior"])


def make_sv_system(mu=-0.5, phi=0.98, sigma=0.16):
    p0 = sigma**2 / (1.0 - phi**2)
    prior = Gaussian(jnp.array([mu]), jnp.array([[p0]]))
    return SVSystem(mu, phi, sigma, prior)


def simulate_data(system, nb_steps, rng):
    mu, phi, sigma = system.mu, system.phi, system.sigma
    p0 = float(system.prior.cov[0, 0])
    x0 = mu + np.sqrt(p0) * rng.randn()
    _, true_states, observations = sv_env.get_data(x0, mu, phi, sigma, nb_steps, rng)
    return jnp.asarray(true_states), jnp.asarray(observations)


# ---- model log-potential builders -------------------------------------------
def make_model_fns(system, family, backend):
    Q, cov_fn, transition_function, mean_fn, _, _ = sv_env.make_parameters(system.mu, system.phi, system.sigma)
    trans = AdditiveGaussianModel(fun=transition_function, noise=Gaussian(jnp.zeros((1,)), Q))
    obs = ConditionalMomentsModel(mean_fn=mean_fn, cov_fn=cov_fn)

    if family == "GSLR":
        method = GSLR_BACKENDS[backend]
        lp = lambda q: _pl.get_log_prior(system.prior, q, method)
        lt = lambda q, _: _pl.get_log_transition(trans, q, method)
        lo = lambda y, q: _pl.get_log_observation(y, obs, q, method)
    elif family == "FH":
        method = FH_BACKENDS[backend]
        lp = lambda q: _fh.get_log_prior(system.prior, q, method)
        lt = lambda q, p: _fh.get_log_transition(trans, q, p, method)
        lo = lambda y, q: _fh.get_log_observation(y, obs, q, method)
    else:
        raise ValueError(f"unknown family {family!r}")
    return lp, lt, lo


# ---- reporting helpers ------------------------------------------------------
def save_fig(fig, name):
    return common.save_fig(fig, OUTPUT_DIR / name)
