"""Cubic sensor: ELBO and NLPD per iteration, FEVS (FH/GSLR) vs undamped IPLS."""

import math
import pickle

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import _cubic_common as cs
import common
import matplotlib.pyplot as plt

from varsmooth.environments.cubic_sensor import make_parameters
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.smoothers.forward_markov import iterated_forward_markov_smoother
from varsmooth.smoothers.utils import std_forward_message

# parsmooth predates recent JAX/NumPy API removals; re-add what it still calls
jax.tree_map = jax.tree_util.tree_map  # removed from top-level jax (now jax.tree_util.tree_map)
np.math = math  # removed in NumPy 2.0; parsmooth's Gauss-Hermite weights use np.math.factorial
from parsmooth._base import FunctionalModel
from parsmooth._base import MVNStandard
from parsmooth.linearization import gauss_hermite as _pars_gauss_hermite
from parsmooth.methods import filter_smoother
from parsmooth.methods import filtering

# IPLS linearization: Gauss-Hermite order 5
PARS_GH5 = lambda model, x: _pars_gauss_hermite(model, x, order=5)

DATA_SEED = 37

INIT_TEMP = 1e8
KL_STEPS = [1.0, 5.0, 25.0]

NUM_STEPS = 4096
MAX_ITER = 80


common.set_style(linewidth=1.6)


def parsmooth_models(system):
    """Parsmooth (x0, transition, observation) models for the cubic system (IPLS baseline)."""
    Q, R, tfn, ofn, _, _ = make_parameters(system.phi0, system.mu0, system.Sigma0, system.beta, system.r)
    x0 = MVNStandard(system.prior.mean, system.prior.cov)
    trans = FunctionalModel(tfn, MVNStandard(jnp.zeros(1), Q))
    obs = FunctionalModel(ofn, MVNStandard(jnp.zeros(1), R))
    return x0, trans, obs


SYSTEM = cs.make_cubic_system()
X_TRUE, YS = cs.simulate_data(SYSTEM, NUM_STEPS, np.random.RandomState(DATA_SEED))
ELBO_OF = cs.make_elbo_evaluator(SYSTEM, YS)
NLPD_OF = cs.make_nlpd_evaluator(X_TRUE)
X0, TRANS, OBS = parsmooth_models(SYSTEM)


def nominal_to_gaussmarkov(nominal):
    """Rebuild a forward Gauss-Markov (marginals, kernels) from a parsmooth smoothed trajectory."""
    filt = filtering(YS, X0, TRANS, OBS, PARS_GH5, nominal, parallel=False)
    Pf = filt.cov
    ms, Ps = nominal.mean, nominal.cov

    def kernel(Pf_k, ms_k, Ps_k, ms_k1, Ps_k1):
        F_k, Ql, _ = PARS_GH5(TRANS, MVNStandard(ms_k, Ps_k))
        G = Pf_k @ F_k.T @ jnp.linalg.inv(F_k @ Pf_k @ F_k.T + Ql)
        cross = G @ Ps_k1
        Ffwd = cross.T @ jnp.linalg.inv(Ps_k)
        return Ffwd, ms_k1 - Ffwd @ ms_k, Ps_k1 - Ffwd @ cross

    Ffwd, dfwd, Sfwd = jax.vmap(kernel)(Pf[:-1], ms[:-1], Ps[:-1], ms[1:], Ps[1:])
    return Gaussian(ms, Ps), AffineGaussian(Ffwd, dfwd, Sfwd)


def run_ours(family, eps, init_gm, e0, n0):
    """Our damped smoother from a Gauss-Markov init; ELBO/NLPD prepended with the init's values."""
    backend = cs.GH5_QUAD if family == "FH" else cs.GH5_LIN
    fns = cs.make_model_fns(SYSTEM, family, backend)
    with common.silence_stdout():
        _, hist = iterated_forward_markov_smoother(
            YS,
            *fns,
            init_gm,
            kl_constraint=eps,
            init_temperature=INIT_TEMP,
            max_iterations=MAX_ITER,
            return_history=True,
        )
    feas = np.asarray(hist["feasible"]).astype(bool)
    elbo = np.asarray(jax.vmap(ELBO_OF)(hist["marginals"], hist["kernels"]))[feas]
    nlpd = np.asarray(jax.vmap(NLPD_OF)(hist["marginals"]))[feas]
    return np.concatenate([[e0], elbo]), np.concatenate([[n0], nlpd])


def run_ipls(init_marg, e0, n0):
    """Undamped sequential IPLS from a given init trajectory; ELBO/NLPD prepended with the init's values."""
    nominal = MVNStandard(init_marg.mean, init_marg.cov)
    elbos, nlpds = [e0], [n0]
    for _ in range(MAX_ITER):
        nominal = filter_smoother(YS, X0, TRANS, OBS, PARS_GH5, nominal, parallel=False)
        marg, kern = nominal_to_gaussmarkov(nominal)
        elbos.append(float(ELBO_OF(marg, kern)))
        nlpds.append(float(NLPD_OF(marg)))
    return np.array(elbos), np.array(nlpds)


def compute_or_load():
    """All algorithms from the shared prior-chain init -> {key: (elbo[], nlpd[])}; cached."""
    key = (NUM_STEPS, tuple(KL_STEPS), MAX_ITER, DATA_SEED)
    cache = cs.OUTPUT_DIR / "_results_cache.pkl"
    if cache.exists():
        blob = pickle.load(cache.open("rb"))
        if blob.get("key") == key:
            print("  (loaded cached results; delete outputs/_results_cache.pkl to recompute)")
            return blob["res"]

    print(f"cubic sensor: T={NUM_STEPS}, seed={DATA_SEED}, eps={KL_STEPS}, {MAX_ITER} iters")
    init_gm = cs.make_prior_chain_init(SYSTEM, NUM_STEPS)
    init_marg = std_forward_message(init_gm)
    e0 = float(ELBO_OF(init_marg, init_gm.kernels))
    n0 = float(NLPD_OF(init_marg))

    res = {}
    for family in ["FH", "GSLR"]:
        for eps in KL_STEPS:
            res[(family, eps)] = run_ours(family, eps, init_gm, e0, n0)
    res["IPLS"] = run_ipls(init_marg, e0, n0)

    pickle.dump({"key": key, "res": res}, cache.open("wb"))
    return res


def _curves():
    blues = plt.cm.Blues(np.linspace(0.5, 0.92, len(KL_STEPS)))
    oranges = plt.cm.Oranges(np.linspace(0.5, 0.92, len(KL_STEPS)))
    curves = [(("FH", e), c, "-o", rf"FH ($\varepsilon={e:g}$)") for c, e in zip(blues, KL_STEPS)]
    curves += [(("GSLR", e), c, "--o", rf"GSLR ($\varepsilon={e:g}$)") for c, e in zip(oranges, KL_STEPS)]
    curves += [("IPLS", "0.15", ":s", "IPLS")]
    return curves


def plot(res, elbo_ylim, nlpd_ylim):
    """Two panels: ELBO (left) and calibration NLPD (right) per iteration, FEVS vs IPLS."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for key, col, style, lab in _curves():
        elbo, nlpd = res[key]
        axL.plot(np.arange(len(elbo)), elbo, style, ms=3.5, color=col, label=lab)
        axR.plot(np.arange(len(nlpd)), nlpd, style, ms=3.5, color=col, label=lab)
    axL.set(
        xlabel="iteration",
        ylabel=r"ELBO  $\mathcal{L}$  (FH+GH5, exact)",
        title="(a) Evidence lower bound",
        ylim=elbo_ylim,
    )
    axR.set(
        xlabel="iteration",
        ylabel="NLPD of true state",
        title="(b) Log score",
        ylim=nlpd_ylim,
    )
    axL.legend(frameon=False, loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    cs.save_fig(fig, "fig_cubic_elbo_nlpd.pdf")
    plt.close(fig)


def write_csvs(res, elbo_ylim, nlpd_ylim):
    # one column per curve (FH/GSLR at each eps, plus IPLS)
    columns = (
        [(("FH", e), f"FH_eps{e:g}") for e in KL_STEPS]
        + [(("GSLR", e), f"GSLR_eps{e:g}") for e in KL_STEPS]
        + [("IPLS", "IPLS")]
    )

    def clamp(v, lo, hi):  # keep off-scale undamped-IPLS values within the plotted range
        return min(max(float(v), lo), hi)

    for which, fname, (lo, hi) in [
        (0, "results_cubic_elbo.csv", elbo_ylim),
        (1, "results_cubic_nlpd.csv", nlpd_ylim),
    ]:
        series = {col: res[key][which] for key, col in columns}
        n_iters = max(len(s) for s in series.values())
        rows = []
        for i in range(n_iters):
            row = {"iter": i}
            for col, s in series.items():
                row[col] = clamp(s[i], lo, hi) if i < len(s) else ""
            rows.append(row)
        common.write_csv(cs.OUTPUT_DIR / fname, ["iter", *series], rows)


def main():
    res = compute_or_load()
    e0, n0 = float(res[("FH", 1.0)][0][0]), float(res[("FH", 1.0)][1][0])
    fevs_elbo = np.concatenate([res[k][0] for k in res if k != "IPLS"])
    fevs_nlpd = np.concatenate([res[k][1] for k in res if k != "IPLS"])
    print(
        f"  init ELBO={e0:.1f} NLPD={n0:.3f}  |  FH eps=1 final ELBO={res[('FH', 1.0)][0][-1]:.1f} NLPD={res[('FH', 1.0)][1][-1]:.3f}"
    )
    elbo_ylim = (e0 - 800, float(np.nanmax(fevs_elbo)) + 100)
    nlpd_ylim = (float(np.nanmin(fevs_nlpd)) - 0.03, n0 + 0.08)
    plot(res, elbo_ylim=elbo_ylim, nlpd_ylim=nlpd_ylim)
    write_csvs(res, elbo_ylim, nlpd_ylim)


if __name__ == "__main__":
    main()
