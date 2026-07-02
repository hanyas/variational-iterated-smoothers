"""Shared helpers for the experiments."""

import csv
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Headless matplotlib backend for the experiment scripts; set before any pyplot
# import (the scripts import pyplot lazily inside main(), after this harness).
os.environ.setdefault("MPLBACKEND", "Agg")

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
from varsmooth.smoothers.hybrid_markov import iterated_hybrid_markov_smoother
from varsmooth.smoothers.reverse_markov import iterated_reverse_markov_smoother
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
def write_csv(path, rows):
    """Write rows to path as CSV and log the destination.

    Args:
        path: str or Path
            Destination CSV file.
        rows: list
            Rows as dicts sharing the same keys; the columns are the first row's keys.
    """
    path = Path(path)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path}")


def save_fig(fig, path):
    """Save fig with a tight bounding box, close it, and log the destination.

    Args:
        fig: matplotlib.figure.Figure
            The figure to write.
        path: str or Path
            Destination image file.

    Returns:
        Path
            The path written.
    """
    import matplotlib.pyplot as plt

    path = Path(path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
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
    """Build a jitted ELBO F(q) = E_q[log p] + H(q) for the given log-potential builders.

    Args:
        observations: Array
            Observation sequence of leading shape (T,).
        lp_fn: Callable
            Log-prior builder.
        lt_fn: Callable
            Log-transition builder.
        lo_fn: Callable
            Log-likelihood builder.

    Returns:
        Callable
            elbo_of(marginals, kernels) returning the scalar free energy.
    """

    @jax.jit
    def elbo_of(marginals, kernels):
        log_prior, log_transition, log_likelihood = statistical_expansion(
            observations, lp_fn, lt_fn, lo_fn, kernels, marginals
        )
        return free_energy(log_prior, log_transition, log_likelihood, marginals, kernels)

    return elbo_of


def rmse(mean, ref):
    """Return the trajectory RMSE sqrt(mean_k ||m_k - ref_k||^2)."""
    m = np.asarray(mean)
    r = np.asarray(ref).reshape(m.shape)
    return float(np.sqrt(np.mean(np.sum((m - r) ** 2, axis=-1))))


def avg_kl(q, ref):
    """Return the mean over k of KL(q_k || ref_k)."""
    return float(jnp.mean(jax.vmap(kl_between_marginals)(q, ref)))


def nlpd(marginals, x_true):
    """Return the calibration NLPD mean_k -log N(x*_k; m_k, P_k) of the true path."""
    m = jnp.asarray(marginals.mean)
    P = jnp.asarray(marginals.cov)
    x = jnp.asarray(x_true).reshape(m.shape)

    diff = x - m
    quad = jnp.sum(diff * jnp.linalg.solve(P, diff[..., None])[..., 0], axis=-1)
    logdet = jnp.linalg.slogdet(P)[1]
    return jnp.mean(0.5 * (m.shape[-1] * jnp.log(2 * jnp.pi) + logdet + quad))


# ---- smoother runners -------------------------------------------------------
def make_forward_init(system, num_steps, F_scale=0.1, Sigma_scale=1.0, prior=None):
    """Build a forward Gauss-Markov init: root = prior, kernels = (F_scale I, 0, Sigma_scale I).

    Args:
        system: namedtuple
            System carrying a .prior Gaussian.
        num_steps: int
            Number of transitions T.
        F_scale: float
            Scale of the identity transition map in the init kernels.
        Sigma_scale: float
            Scale of the identity conditional covariance in the init kernels.
        prior: Gaussian
            Root marginal to use instead of system.prior when given.

    Returns:
        GaussMarkov
            The forward Gauss-Markov init.
    """
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


def get_marginals(direction, result):
    """Return the standard marginals of a smoother result for the given direction."""
    if direction == "forward":
        return std_forward_message(result)
    if direction == "reverse":
        return std_backward_message(result)
    return result  # hybrid already returns marginals


def get_markov_history(diags, rts=None, x_true=None):
    """Build per-iteration records from smoother diagnostics.

    Args:
        diags: dict
            Smoother return_history diagnostics (marginals, dual, damping, ...).
        rts: Gaussian
            Exact RTS marginals; when given, adds rmse_rts and kl_rts per iteration.
        x_true: Array
            True trajectory; when given, adds rmse_true per iteration.

    Returns:
        list
            One dict per iteration with the recorded quantities.
    """
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
    verbose=False,
):
    """Run the KL-constrained iterated smoother (forward / reverse / hybrid) from a forward init.

    Args:
        direction: str
            One of "forward", "reverse", "hybrid".
        model_fns: tuple
            The (log_prior_fn, log_transition_fn, log_likelihood_fn) builders.
        observations: Array
            Observation sequence of leading shape (T,).
        init_fwd_posterior: GaussMarkov
            Forward Gauss-Markov init; the reverse/hybrid inits are derived from it.
        kl_constraint: float
            Trust-region radius per iteration.
        init_temperature: float
            Initial line-search temperature.
        min_temperature: float
            Early-stopping temperature threshold.
        max_iterations: int
            Maximum number of iterations.
        return_history: bool
            When True, also return the per-iteration diagnostics.
        verbose: bool
            Forwarded to the smoother; False silences its per-iteration prints.

    Returns:
        Gaussian
            The smoothed marginals, or (marginals, diagnostics) when return_history.
    """
    lp, lt, lo = model_fns
    kw = dict(
        kl_constraint=kl_constraint,
        init_temperature=init_temperature,
        min_temperature=min_temperature,
        max_iterations=max_iterations,
        return_history=return_history,
        verbose=verbose,
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
