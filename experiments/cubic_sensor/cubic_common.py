"""Shared harness for the cubic-sensor experiments."""

from collections import namedtuple
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
from common import FH_BACKENDS
from common import GSLR_BACKENDS
from common import set_style
from common import silence_stdout
from common import write_csv

from varsmooth.approximation import cubature_linearization
from varsmooth.approximation import cubature_quadratization
from varsmooth.approximation import fourier_hermite as _fh
from varsmooth.approximation import gauss_hermite_linearization
from varsmooth.approximation import gauss_hermite_quadratization
from varsmooth.approximation import linearization as _pl
from varsmooth.environments.cubic_sensor import get_data
from varsmooth.environments.cubic_sensor import make_parameters
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

GH5_QUAD = lambda fun, q: gauss_hermite_quadratization(fun, q, order=5)
GH5_LINEAR = lambda fun, q: gauss_hermite_linearization(fun, q, order=5)

CUB_QUAD = lambda fun, q: cubature_quadratization(fun, q)
CUB_LINEAR = lambda fun, q: cubature_linearization(fun, q)

DIM_X, DIM_Y = 1, 1


# ---- system + data ----------------------------------------------------------
CubicSystem = namedtuple("CubicSystem", ["phi0", "mu0", "Sigma0", "beta", "r", "prior"])


def make_cubic_system(phi0=0.95, mu0=0.4, Sigma0=0.36, beta=1.0, r=1.0):
    prior = Gaussian(jnp.array([mu0]), jnp.array([[Sigma0]]))
    return CubicSystem(phi0, mu0, Sigma0, beta, r, prior)


def simulate_data(system, num_steps, rng):
    x0 = system.mu0 + np.sqrt(system.Sigma0) * rng.randn()
    _, true_states, observations = get_data(
        x0, system.phi0, system.mu0, system.Sigma0, system.beta, system.r, num_steps, rng
    )
    return jnp.asarray(true_states), jnp.asarray(observations)


def make_additive_models(system):
    Q, R, trans_fn, obsrv_fn, _, _ = make_parameters(system.phi0, system.mu0, system.Sigma0, system.beta, system.r)
    trans = AdditiveGaussianModel(fun=trans_fn, noise=Gaussian(jnp.zeros(DIM_X), Q))
    obs = AdditiveGaussianModel(fun=obsrv_fn, noise=Gaussian(jnp.zeros(DIM_Y), R))
    return trans, obs


def make_model_fns(system, family, backend):
    trans, obs = make_additive_models(system)
    if family == "GSLR":
        method = backend if callable(backend) else GSLR_BACKENDS[backend]
        lp = lambda q: _pl.get_log_prior(system.prior, q, method)
        lt = lambda q, _: _pl.get_log_transition(trans, q, method)
        lo = lambda y, q: _pl.get_log_likelihood(y, obs, q, method)
    elif family == "FH":
        method = backend if callable(backend) else FH_BACKENDS[backend]
        lp = lambda q: _fh.get_log_prior(system.prior, q, method)
        lt = lambda q, p: _fh.get_log_transition(trans, q, p, method)
        lo = lambda y, q: _fh.get_log_likelihood(y, obs, q, method)
    else:
        raise ValueError(f"unknown family {family!r}")
    return lp, lt, lo


def make_prior_chain_init(system, num_steps):
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
    return common.make_elbo_evaluator(observations, *make_model_fns(system, "FH", GH5_QUAD))


def make_nlpd_evaluator(true_states):
    return lambda marginals: common.nlpd(marginals, true_states)


def save_fig(fig, name):
    return common.save_fig(fig, OUTPUT_DIR / name)
