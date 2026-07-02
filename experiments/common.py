"""Shared helpers for the experiments"""

import contextlib
import csv
import os
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from varsmooth.approximation import cubature_linearization
from varsmooth.approximation import cubature_quadratization
from varsmooth.approximation import extended_linearization
from varsmooth.approximation import gauss_hermite_linearization
from varsmooth.approximation import gauss_hermite_quadratization
from varsmooth.approximation import unscented_linearization
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.smoothers.forward_markov import iterated_forward_markov_smoother
from varsmooth.smoothers.reverse_markov import iterated_reverse_markov_smoother
from varsmooth.smoothers.hybrid_markov import iterated_hybrid_markov_smoother
from varsmooth.smoothers.utils import free_energy
from varsmooth.smoothers.utils import initialize_reverse_with_forward
from varsmooth.smoothers.utils import kl_between_marginals
from varsmooth.smoothers.utils import statistical_expansion
from varsmooth.smoothers.utils import std_backward_message
from varsmooth.smoothers.utils import std_forward_message

# ---- approximation backends (label -> linearization / quadratization method) ----
GSLR_BACKENDS = {
    "extended": extended_linearization,
    "cubature": cubature_linearization,
    "unscented": unscented_linearization,
    "gauss_hermite": gauss_hermite_linearization,
}
FH_BACKENDS = {
    "cubature": cubature_quadratization,
    "gauss_hermite": gauss_hermite_quadratization,
}


# ---- I/O --------------------------------------------------------------------
@contextlib.contextmanager
def silence_stdout():
    """Suppress the smoother's per-iteration `jax.debug.print` (writes to OS fd 1)."""
    sys.stdout.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(1)
    os.dup2(devnull, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def write_csv(path, rows):
    """Write `rows` (list of dicts) to `path` as CSV; columns are the first row's keys."""
    path = Path(path)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}")


def save_fig(fig, path):
    """Save `fig` to `path` (tight bbox) and return the path."""
    path = Path(path)
    fig.savefig(path, bbox_inches="tight")
    print(f"  wrote {path}")
    return path


def set_style(linewidth=1.8, **overrides):
    """Apply the shared matplotlib paper style (matplotlib imported lazily)."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "serif",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 120,
            "lines.linewidth": linewidth,
            **overrides,
        }
    )


# ---- metrics ----------------------------------------------------------------
def make_elbo_evaluator(observations, lp_fn, lt_fn, lo_fn):
    """Factory: jitted ELBO F(q) = E_q[log p] + H(q) for the given log-potential builders."""

    @jax.jit
    def elbo_of(marginals, kernels):
        log_prior, log_transition, log_observation = statistical_expansion(
            observations, lp_fn, lt_fn, lo_fn, kernels, marginals
        )
        return free_energy(log_prior, log_transition, log_observation, marginals, kernels)

    return elbo_of


def rmse(mean, ref):
    """Trajectory RMSE sqrt( mean_k ||m_k - ref_k||^2 )."""
    m = np.asarray(mean)
    r = np.asarray(ref).reshape(m.shape)
    return float(np.sqrt(np.mean(np.sum((m - r) ** 2, axis=-1))))


def avg_kl(q, ref):
    """Mean over k of KL(q_k || ref_k)."""
    return float(jnp.mean(jax.vmap(kl_between_marginals)(q, ref)))


def nlpd(marginals, x_true):
    """Calibration NLPD: mean_k -log N(x*_k; m_k, P_k) of the true path under the marginals."""
    m = jnp.asarray(marginals.mean)
    P = jnp.asarray(marginals.cov)
    x = jnp.asarray(x_true).reshape(m.shape)

    diff = x - m
    quad = jnp.sum(diff * jnp.linalg.solve(P, diff[..., None])[..., 0], axis=-1)
    logdet = jnp.linalg.slogdet(P)[1]
    return jnp.mean(0.5 * (m.shape[-1] * jnp.log(2 * jnp.pi) + logdet + quad))


# ---- smoother runners -------------------------------------------------------


def make_forward_init(system, num_steps, F_scale=0.1, Sigma_scale=1.0, prior=None):
    """Forward Gauss-Markov init: root = prior, kernels = (F_scale*I, 0, Sigma_scale*I)."""
    dim_x = jnp.asarray(system.prior.mean).shape[0]
    F = F_scale * np.eye(dim_x)
    d = np.zeros((dim_x,))
    Sigma = Sigma_scale * np.eye(dim_x)
    return GaussMarkov(
        marginal=system.prior if prior is None else prior,
        kernels=AffineGaussian(
            np.repeat([F], num_steps, axis=0),
            np.repeat([d], num_steps, axis=0),
            np.repeat([Sigma], num_steps, axis=0),
        ),
    )


def make_reverse_init(system, num_steps, **kwargs):
    """Reverse Gauss-Markov init derived from the forward init."""
    return initialize_reverse_with_forward(make_forward_init(system, num_steps, **kwargs))


def get_marginals(direction, result):
    if direction == "forward":
        return std_forward_message(result)
    if direction == "reverse":
        return std_backward_message(result)
    return result  # hybrid already returns marginals


def get_markov_history(diags, rts=None, x_true=None):
    """Per-iteration records from smoother `diags`; adds rmse/kl vs `rts` and rmse vs `x_true` when given."""
    means = np.asarray(diags["marginals"].mean)
    covs = np.asarray(diags["marginals"].cov)
    history = []
    for i in range(len(means)):
        rec = dict(
            iter=i,
            dual=float(diags["dual"][i]),
            damping=float(diags["damping"][i]),
            feasible=bool(diags["feasible"][i]),
            realized_kl=float(diags["realized_kl"][i]),
        )
        q = Gaussian(jnp.asarray(means[i]), jnp.asarray(covs[i]))
        if rts is not None:
            rec["rmse_rts"] = rmse(means[i], rts.mean)
            rec["kl_rts"] = avg_kl(q, rts)
        if x_true is not None:
            rec["rmse_true"] = rmse(means[i], x_true)
        history.append(rec)
    return history


def run_iterated_smoother(
    direction,
    model_fns,
    observations,
    init_fwd_posterior,
    kl_constraint=1.0,
    init_temperature=1e6,
    min_temperature=1e-12,
    max_iterations=100,
    return_history=False,
):
    """KL-constrained iterated smoother (forward / reverse / hybrid) from a forward init."""
    lp, lt, lo = model_fns
    kw = dict(
        kl_constraint=kl_constraint,
        init_temperature=init_temperature,
        min_temperature=min_temperature,
        max_iterations=max_iterations,
        return_history=return_history,
    )
    if direction == "forward":
        out = iterated_forward_markov_smoother(observations, lp, lt, lo, init_fwd_posterior, **kw)
    elif direction == "reverse":
        init_rev_posterior = initialize_reverse_with_forward(init_fwd_posterior)
        out = iterated_reverse_markov_smoother(observations, lp, lt, lo, init_rev_posterior, **kw)
    elif direction == "hybrid":
        init_rev_posterior = initialize_reverse_with_forward(init_fwd_posterior)
        out = iterated_hybrid_markov_smoother(observations, lp, lt, lo, init_fwd_posterior, init_rev_posterior, **kw)
    else:
        raise ValueError(direction)
    if return_history:
        final, diags = out
        return get_marginals(direction, final), diags
    return get_marginals(direction, out)
