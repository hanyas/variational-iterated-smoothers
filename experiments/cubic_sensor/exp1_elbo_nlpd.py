"""Cubic sensor: ELBO and NLPD per iteration, HEVS (FH/GSLR) vs undamped IPLS."""

import pickle

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import _cubic_common as cs
import common
import matplotlib.pyplot as plt

from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.smoothers.rts_kalman import filtering
from varsmooth.smoothers.rts_kalman import smoothing
from varsmooth.smoothers.two_filter import iterated_two_filter_smoother
from varsmooth.smoothers.utils import initialize_reverse_with_forward
from varsmooth.smoothers.utils import std_forward_message

DATA_SEED = 42

INIT_TEMP = 1e8
KL_STEPS = [1.0, 5.0]

NUM_STEPS = 4096
MAX_ITER = 80


common.set_style(linewidth=1.6)


SYSTEM = cs.make_cubic_system()
X_TRUE, YS = cs.simulate_data(SYSTEM, NUM_STEPS, np.random.RandomState(DATA_SEED))
ELBO_OF = cs.make_elbo_evaluator(SYSTEM, YS)
NLPD_OF = cs.make_nlpd_evaluator(X_TRUE)


def run_ours(family, kl_constraint, init_forward, init_elbo, init_nlpd):
    backend = cs.GH5_QUAD if family == "FH" else cs.GH5_LIN
    log_prior, log_transition, log_observation = cs.make_model_fns(SYSTEM, family, backend)
    init_reverse = initialize_reverse_with_forward(init_forward)
    with common.silence_stdout():
        _, history = iterated_two_filter_smoother(
            YS,
            log_prior,
            log_transition,
            log_observation,
            init_forward,
            init_reverse,
            kl_constraint=kl_constraint,
            init_temperature=INIT_TEMP,
            max_iterations=MAX_ITER,
            return_history=True,
        )
    feasible = np.asarray(history["feasible"]).astype(bool)
    marginals, kernels = history["forward_marginals"], history["forward_kernels"]

    def prepend_init(init_value, per_iteration):
        return np.concatenate([[init_value], np.asarray(per_iteration)[feasible]])

    elbo = prepend_init(init_elbo, jax.vmap(ELBO_OF)(marginals, kernels))
    nlpd = prepend_init(init_nlpd, jax.vmap(NLPD_OF)(marginals))
    return elbo, nlpd


def run_ipls(init_forward, init_elbo, init_nlpd):
    transition_model, observation_model = cs.make_additive_models(SYSTEM)

    def step(forward, _):
        marginals = std_forward_message(forward)
        linear_transition = jax.vmap(lambda marginal: AffineGaussian(*cs.GH5_LIN(transition_model, marginal)))(
            Gaussian(marginals.mean[:-1], marginals.cov[:-1])
        )
        linear_observation = jax.vmap(lambda marginal: AffineGaussian(*cs.GH5_LIN(observation_model, marginal)))(
            Gaussian(marginals.mean[1:], marginals.cov[1:])
        )
        filter_trajectory = filtering(YS, SYSTEM.prior, linear_transition, linear_observation)
        posterior = smoothing(linear_transition, filter_trajectory)
        marginals = std_forward_message(posterior)
        return posterior, (ELBO_OF(marginals, posterior.kernels), NLPD_OF(marginals))

    _, (elbos, nlpds) = jax.lax.scan(step, init_forward, None, length=MAX_ITER)
    return np.concatenate([[init_elbo], np.asarray(elbos)]), np.concatenate([[init_nlpd], np.asarray(nlpds)])


def compute_or_load():
    key = (NUM_STEPS, tuple(KL_STEPS), MAX_ITER, DATA_SEED)
    cache = cs.OUTPUT_DIR / "_results_cache.pkl"
    if cache.exists():
        blob = pickle.load(cache.open("rb"))
        if blob.get("key") == key:
            print("  (loaded cached results; delete outputs/_results_cache.pkl to recompute)")
            return blob["res"]

    print(f"cubic sensor: T={NUM_STEPS}, seed={DATA_SEED}, eps={KL_STEPS}, {MAX_ITER} iters")
    init_forward = cs.make_prior_chain_init(SYSTEM, NUM_STEPS)
    init_marginals = std_forward_message(init_forward)
    init_elbo = float(ELBO_OF(init_marginals, init_forward.kernels))
    init_nlpd = float(NLPD_OF(init_marginals))

    res = {}
    for family in ["FH", "GSLR"]:
        for kl_constraint in KL_STEPS:
            res[(family, kl_constraint)] = run_ours(family, kl_constraint, init_forward, init_elbo, init_nlpd)
    res["IPLS"] = run_ipls(init_forward, init_elbo, init_nlpd)

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
    columns = (
        [(("FH", e), f"FH_eps{e:g}") for e in KL_STEPS]
        + [(("GSLR", e), f"GSLR_eps{e:g}") for e in KL_STEPS]
        + [("IPLS", "IPLS")]
    )

    def clamp(v, lo, hi):
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
    fh_final_elbo = res[("FH", 1.0)][0][-1]
    fh_final_nlpd = res[("FH", 1.0)][1][-1]
    print(f"  init ELBO={e0:.1f} NLPD={n0:.3f}")
    print(f"  FH eps=1 final ELBO={fh_final_elbo:.1f} NLPD={fh_final_nlpd:.3f}")
    elbo_ylim = (e0 - 800, float(np.nanmax(fevs_elbo)) + 100)
    nlpd_ylim = (float(np.nanmin(fevs_nlpd)) - 0.03, n0 + 0.08)
    plot(res, elbo_ylim=elbo_ylim, nlpd_ylim=nlpd_ylim)
    write_csvs(res, elbo_ylim, nlpd_ylim)


if __name__ == "__main__":
    main()
