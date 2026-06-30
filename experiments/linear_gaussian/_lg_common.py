"""Shared harness for the linear-Gaussian (LG) experiments."""

from collections import namedtuple
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import jax.scipy as jsc
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from tests.kalman import rts_smoother
from varsmooth.approximation import fourier_hermite as _fh
from varsmooth.approximation import linearization as _pl
from varsmooth.environments import linear_gaussian as lg_env
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.smoothers.forward_markov import forward_markov_smoother
from varsmooth.smoothers.forward_markov import log_evidence as fwd_log_evidence
from varsmooth.smoothers.reverse_markov import log_evidence as rev_log_evidence
from varsmooth.smoothers.reverse_markov import reverse_markov_smoother
from varsmooth.smoothers.two_filter import two_filter_smoother
from varsmooth.smoothers.utils import statistical_expansion
from varsmooth.smoothers.utils import std_backward_message
from varsmooth.smoothers.utils import std_forward_message

# make `varsmooth` and `tests` importable regardless of CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common
from common import FH_BACKENDS
from common import GSLR_BACKENDS
from common import get_marginals
from common import make_forward_init
from common import make_reverse_init

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---- linear-Gaussian system + data ------------------------------------------
LGSystem = namedtuple("LGSystem", ["prior", "A", "b", "Omega", "H", "e", "Delta"])


def make_lg_system(dim_x, dim_y, rng, transition_scale=0.9):
    """Build a stable linear-Gaussian SSM."""
    mu0, P0, A, b, Omega, H, e, Delta = lg_env.make_random_system(dim_x, dim_y, rng, transition_scale)
    return LGSystem(Gaussian(jnp.asarray(mu0), jnp.asarray(P0)), A, b, Omega, H, e, Delta)


def simulate_data(system, nb_steps, rng):
    """Simulate (x_{0:T}, y_{1:T}) with the linear_gaussian environment."""
    mu0 = np.asarray(system.prior.mean)
    P0 = np.asarray(system.prior.cov)
    x0 = mu0 + np.linalg.cholesky(P0) @ rng.randn(mu0.shape[0])
    _, true_states, observations = lg_env.get_data(
        x0, system.A, system.b, system.Omega, system.H, system.e, system.Delta, nb_steps, rng
    )
    return jnp.asarray(true_states), jnp.asarray(observations)


# ---- exact oracles ----------------------------------------------------------
def _batched_models(system, nb_steps):
    lin_trans = AffineGaussian(
        np.repeat([system.A], nb_steps, axis=0),
        np.repeat([system.b], nb_steps, axis=0),
        np.repeat([system.Omega], nb_steps, axis=0),
    )
    lin_obs = AffineGaussian(
        np.repeat([system.H], nb_steps, axis=0),
        np.repeat([system.e], nb_steps, axis=0),
        np.repeat([system.Delta], nb_steps, axis=0),
    )
    return lin_trans, lin_obs


def rts_marginals(system, observations):
    """Exact smoothing marginals N(m_k, P_k), k = 0..T."""
    nb_steps = observations.shape[0]
    lin_trans, lin_obs = _batched_models(system, nb_steps)
    return rts_smoother(observations, system.prior, lin_trans, lin_obs)


def kalman_log_evidence(system, observations):
    """Exact log marginal likelihood log p(y_{1:T}) from the Kalman filter."""
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


# ---- model log-potential builders -------------
def make_model_fns(system, family, backend):
    """Return (log_prior_fn, log_transition_fn, log_observation_fn) closures."""
    dim_x = system.A.shape[0]
    dim_y = system.H.shape[0]
    Q, R, transition_function, observation_function, _, _ = lg_env.make_parameters(
        system.A, system.b, system.Omega, system.H, system.e, system.Delta
    )
    transition_model = AdditiveGaussianModel(fun=transition_function, noise=Gaussian(jnp.zeros((dim_x,)), Q))
    observation_model = AdditiveGaussianModel(fun=observation_function, noise=Gaussian(jnp.zeros((dim_y,)), R))

    if family == "GSLR":
        method = GSLR_BACKENDS[backend]
        lp = lambda q: _pl.get_log_prior(system.prior, q, method)
        lt = lambda q, _: _pl.get_log_transition(transition_model, q, method)
        lo = lambda y, q: _pl.get_log_observation(y, observation_model, q, method)
    elif family == "FH":
        method = FH_BACKENDS[backend]
        lp = lambda q: _fh.get_log_prior(system.prior, q, method)
        lt = lambda q, p: _fh.get_log_transition(transition_model, q, p, method)
        lo = lambda y, q: _fh.get_log_observation(y, observation_model, q, method)
    else:
        raise ValueError(f"unknown family {family!r}")
    return lp, lt, lo


# ---- single-pass smoother runners -------------------------------------------
def run_single_pass(direction, model_fns, observations, system, nb_steps, temperature=0.0, init_kwargs=None):
    """One damped pass (temperature -> damping beta).  beta=0 is the full/exact update."""
    lp, lt, lo = model_fns
    init_kwargs = init_kwargs or {}
    if direction == "forward":
        fwd_init = make_forward_init(system, nb_steps, **init_kwargs)
        res = forward_markov_smoother(observations, lp, lt, lo, fwd_init, temperature)
    elif direction == "reverse":
        rev_init = make_reverse_init(system, nb_steps, **init_kwargs)
        res = reverse_markov_smoother(observations, lp, lt, lo, rev_init, temperature)
    elif direction == "hybrid":
        fwd_init = make_forward_init(system, nb_steps, **init_kwargs)
        rev_init = make_reverse_init(system, nb_steps, **init_kwargs)
        res = two_filter_smoother(observations, lp, lt, lo, fwd_init, rev_init, temperature)
    else:
        raise ValueError(direction)
    return get_marginals(direction, res)


def single_pass_elbo(direction, model_fns, observations, system, nb_steps, temperature=0.0, init_kwargs=None):
    """ELBO surrogate L(q) of the single-pass result"""
    lp, lt, lo = model_fns
    init_kwargs = init_kwargs or {}
    if direction == "forward":
        fwd_init = make_forward_init(system, nb_steps, **init_kwargs)
        res = forward_markov_smoother(observations, lp, lt, lo, fwd_init, temperature)
        marg, van = std_forward_message(res), fwd_log_evidence
    elif direction == "reverse":
        rev_init = make_reverse_init(system, nb_steps, **init_kwargs)
        res = reverse_markov_smoother(observations, lp, lt, lo, rev_init, temperature)
        marg, van = std_backward_message(res), rev_log_evidence
    else:
        return None
    lpe, lte, loe = statistical_expansion(observations, lp, lt, lo, res.kernels, marg)
    return float(van(lpe, lte, loe, res))


# ---- reporting helpers ------------------------------------------------------
def mean_se(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0


def fmt_mean_se(mean, se, sig=2):
    """'mean +/- se' with the mantissa in scientific notation when tiny."""
    if not np.isfinite(mean):
        return "--"
    if abs(mean) > 0 and abs(mean) < 1e-3:
        return f"{mean:.{sig}e} $\\pm$ {se:.{sig}e}"
    return f"{mean:.{sig}f} $\\pm$ {se:.{sig}f}"


def save_fig(fig, name):
    return common.save_fig(fig, OUTPUT_DIR / name)
