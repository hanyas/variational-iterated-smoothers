"""Cubic sensor: ELBO and NLPD per iteration, damped forward-markov smoother (FH/GSLR, cubature) vs undamped IPLS.

Outputs:
  outputs/fig_cubic_elbo_nlpd_fwd.pdf   -- ELBO (left) and calibration NLPD (right) per iteration
  outputs/results_cubic_elbo_fwd.csv    -- per-iteration ELBO, one column per curve
  outputs/results_cubic_nlpd_fwd.csv    -- per-iteration NLPD, one column per curve

Run from this directory: python exp2_elbo_fwd.py
"""

import cubic_common as cs
import jax
import numpy as np

from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.smoothers.forward_markov import iterated_forward_markov_smoother
from varsmooth.smoothers.hybrid_markov import iterated_hybrid_markov_smoother
from varsmooth.smoothers.rts_kalman import filtering
from varsmooth.smoothers.rts_kalman import smoothing
from varsmooth.smoothers.utils import initialize_reverse_with_forward
from varsmooth.smoothers.utils import std_forward_message

# ---- config (all that differs between the two cubic scripts) ----
SMOOTHER = "forward"                      # other cubic script: "hybrid"
OUT_SUFFIX = "fwd"                        # other cubic script: "hyb"
HIST_MARGINALS_KEY = "marginals"          # hybrid smoother: "forward_marginals"
HIST_KERNELS_KEY = "kernels"              # hybrid smoother: "forward_kernels"

DATA_SEED = 37
NUM_STEPS = 4096
KL_CONSTRAINTS = [1.0, 5.0]
MAX_ITER = 80


def main():
    import matplotlib.pyplot as plt

    # ---- data ----
    system = cs.make_cubic_system()
    x_true, ys = cs.simulate_data(system, NUM_STEPS, np.random.RandomState(DATA_SEED))

    elbo_of = cs.make_elbo_evaluator(system, ys)
    nlpd_of = cs.make_nlpd_evaluator(x_true)

    init_forward = cs.make_prior_chain_init(system, NUM_STEPS)
    init_reverse = initialize_reverse_with_forward(init_forward) if SMOOTHER == "hybrid" else None
    init_marginals = std_forward_message(init_forward)

    init_elbo = float(elbo_of(init_marginals, init_forward.kernels))
    init_nlpd = float(nlpd_of(init_marginals))
    print(f"cubic sensor ({SMOOTHER}): T={NUM_STEPS}, seed={DATA_SEED}, eps={KL_CONSTRAINTS}, {MAX_ITER} iters")

    # ---- run ----
    results = {}

    # ours: damped markov smoother (FH, GSLR) with cubature, one curve per trust-region eps
    for family in ["FH", "GSLR"]:
        log_prior_fn, log_transition_fn, log_likelihood_fn = cs.make_model_fns(
            system, family, cs.CUB_QUAD if family == "FH" else cs.CUB_LINEAR
        )
        for kl in KL_CONSTRAINTS:
            if SMOOTHER == "hybrid":
                _, h = iterated_hybrid_markov_smoother(
                    observations=ys,
                    log_prior_fn=log_prior_fn,
                    log_transition_fn=log_transition_fn,
                    log_likelihood_fn=log_likelihood_fn,
                    init_forward_posterior=init_forward,
                    init_reverse_posterior=init_reverse,
                    kl_constraint=kl,
                    max_iterations=MAX_ITER,
                    return_history=True,
                    verbose=False,
                )
            else:
                _, h = iterated_forward_markov_smoother(
                    observations=ys,
                    log_prior_fn=log_prior_fn,
                    log_transition_fn=log_transition_fn,
                    log_likelihood_fn=log_likelihood_fn,
                    init_posterior=init_forward,
                    kl_constraint=kl,
                    max_iterations=MAX_ITER,
                    return_history=True,
                    verbose=False,
                )
            keep = np.asarray(h["feasible"]).astype(bool)
            elbo = np.asarray(jax.vmap(elbo_of)(h[HIST_MARGINALS_KEY], h[HIST_KERNELS_KEY]))[keep]
            nlpd = np.asarray(jax.vmap(nlpd_of)(h[HIST_MARGINALS_KEY]))[keep]
            results[(family, kl)] = (np.concatenate([[init_elbo], elbo]), np.concatenate([[init_nlpd], nlpd]))

    # IPLS: undamped native cubature statistical-linear-regression + RTS, re-linearized each iteration
    transition_model, likelihood_model = cs.make_additive_models(system)

    def ipls_step(forward, _):
        m = std_forward_message(forward)
        lin_t = jax.vmap(lambda q: AffineGaussian(*cs.CUB_LINEAR(transition_model, q)))(Gaussian(m.mean[:-1], m.cov[:-1]))
        lin_o = jax.vmap(lambda q: AffineGaussian(*cs.CUB_LINEAR(likelihood_model, q)))(Gaussian(m.mean[1:], m.cov[1:]))
        forward = smoothing(lin_t, filtering(ys, system.prior, lin_t, lin_o))
        m = std_forward_message(forward)
        return forward, (elbo_of(m, forward.kernels), nlpd_of(m))

    _, (ipls_elbo, ipls_nlpd) = jax.lax.scan(ipls_step, init_forward, None, length=MAX_ITER)
    results["IPLS"] = (
        np.concatenate([[init_elbo], np.asarray(ipls_elbo)]),
        np.concatenate([[init_nlpd], np.asarray(ipls_nlpd)]),
    )

    print(f"  init ELBO={init_elbo:.1f} NLPD={init_nlpd:.3f}")
    print(f"  FH eps=1 final ELBO={results[('FH', 1.0)][0][-1]:.1f} NLPD={results[('FH', 1.0)][1][-1]:.3f}")

    # ---- report (figures + csv) ----
    # axis limits from the damped-smoother curves only (off-scale IPLS is clamped in the CSV)
    ours_elbo = np.concatenate([results[k][0] for k in results if k != "IPLS"])
    ours_nlpd = np.concatenate([results[k][1] for k in results if k != "IPLS"])
    elbo_ylim = (init_elbo - 800, float(np.nanmax(ours_elbo)) + 100)
    nlpd_ylim = (float(np.nanmin(ours_nlpd)) - 0.03, init_nlpd + 0.08)

    cs.set_style()
    # figure: ELBO (left) and calibration NLPD (right) per iteration
    grays = plt.cm.gray(np.linspace(0.0, 0.55, len(KL_CONSTRAINTS)))
    curves = [(("FH", kl), c, "-o", rf"FH ($\varepsilon={kl:g}$)") for c, kl in zip(grays, KL_CONSTRAINTS)]
    curves += [(("GSLR", kl), c, "--o", rf"GSLR ($\varepsilon={kl:g}$)") for c, kl in zip(grays, KL_CONSTRAINTS)]
    curves += [("IPLS", "0.15", ":s", "IPLS")]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for key, col, style, lab in curves:
        elbo, nlpd = results[key]
        axL.plot(np.arange(len(elbo)), elbo, style, ms=3.5, color=col, label=lab)
        axR.plot(np.arange(len(nlpd)), nlpd, style, ms=3.5, color=col, label=lab)

    axL.set(xlabel="iteration", ylabel=r"ELBO", title="(a) Evidence lower bound", ylim=elbo_ylim)
    axR.set(xlabel="iteration", ylabel="NLPD", title="(b) Log score", ylim=nlpd_ylim)
    axL.legend(frameon=False, loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    cs.save_fig(fig, f"fig_cubic_elbo_nlpd_{OUT_SUFFIX}.pdf")

    # CSVs: one column per curve, off-scale IPLS clamped to the plotted range
    columns = (
        [(("FH", kl), f"FH_eps{kl:g}") for kl in KL_CONSTRAINTS]
        + [(("GSLR", kl), f"GSLR_eps{kl:g}") for kl in KL_CONSTRAINTS]
        + [("IPLS", "IPLS")]
    )
    for which, fname, (lo, hi) in [
        (0, f"results_cubic_elbo_{OUT_SUFFIX}.csv", elbo_ylim),
        (1, f"results_cubic_nlpd_{OUT_SUFFIX}.csv", nlpd_ylim),
    ]:
        series = {col: results[key][which] for key, col in columns}
        n_iters = max(len(s) for s in series.values())
        rows = []
        for i in range(n_iters):
            row = {"iter": i}
            for col, s in series.items():
                row[col] = min(max(float(s[i]), lo), hi) if i < len(s) else ""
            rows.append(row)
        cs.write_csv(cs.OUTPUT_DIR / fname, rows)


if __name__ == "__main__":
    main()
