"""Cubic sensor (exp2): same as exp1 but the hybrid-markov smoother + IPLS use the cubature backend (CUB_QUAD/CUB_LINEAR); ELBO still evaluated with GH5."""

from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiments.cubic_sensor.cubic_common as cs
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.smoothers.hybrid_markov import iterated_hybrid_markov_smoother
from varsmooth.smoothers.rts_kalman import filtering
from varsmooth.smoothers.rts_kalman import smoothing
from varsmooth.smoothers.utils import initialize_reverse_with_forward
from varsmooth.smoothers.utils import std_forward_message

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


def main():
    data_seed = 37
    num_steps = 4096
    kl_steps = [1.0, 5.0]
    max_iter = 80

    # model, data, evaluators, shared Gauss-Markov init
    system = cs.make_cubic_system()
    x_true, ys = cs.simulate_data(system, num_steps, np.random.RandomState(data_seed))

    elbo_of = cs.make_elbo_evaluator(system, ys)
    nlpd_of = cs.make_nlpd_evaluator(x_true)

    init_forward = cs.make_prior_chain_init(system, num_steps)
    init_reverse = initialize_reverse_with_forward(init_forward)
    init_marginals = std_forward_message(init_forward)

    init_elbo = float(elbo_of(init_marginals, init_forward.kernels))
    init_nlpd = float(nlpd_of(init_marginals))
    print(f"cubic sensor: T={num_steps}, seed={data_seed}, eps={kl_steps}, {max_iter} iters")

    results = {}

    # ours: damped hybrid-markov smoother (FH, GSLR) with cubature, one curve per trust-region eps
    for family in ["FH", "GSLR"]:
        log_prior_fn, log_trans_fn, log_obsrv_fn = cs.make_model_fns(
            system, family, cs.CUB_QUAD if family == "FH" else cs.CUB_LINEAR
        )
        for step in kl_steps:
            with cs.silence_stdout():
                _, h = iterated_hybrid_markov_smoother(
                    observations=ys,
                    log_prior_fn=log_prior_fn,
                    log_transition_fn=log_trans_fn,
                    log_likelihood_fn=log_obsrv_fn,
                    init_forward_posterior=init_forward,
                    init_reverse_posterior=init_reverse,
                    kl_constraint=step,
                    max_iterations=max_iter,
                    return_history=True,
                )
            keep = np.asarray(h["feasible"]).astype(bool)
            elbo = np.asarray(jax.vmap(elbo_of)(h["forward_marginals"], h["forward_kernels"]))[keep]
            nlpd = np.asarray(jax.vmap(nlpd_of)(h["forward_marginals"]))[keep]
            results[(family, step)] = (np.concatenate([[init_elbo], elbo]), np.concatenate([[init_nlpd], nlpd]))

    # IPLS: undamped native cubature statistical-linear-regression + RTS, re-linearized each iteration
    transition, observation = cs.make_additive_models(system)

    def ipls_step(forward, _):
        m = std_forward_message(forward)
        lin_t = jax.vmap(lambda q: AffineGaussian(*cs.CUB_LINEAR(transition, q)))(Gaussian(m.mean[:-1], m.cov[:-1]))
        lin_o = jax.vmap(lambda q: AffineGaussian(*cs.CUB_LINEAR(observation, q)))(Gaussian(m.mean[1:], m.cov[1:]))
        forward = smoothing(lin_t, filtering(ys, system.prior, lin_t, lin_o))
        m = std_forward_message(forward)
        return forward, (elbo_of(m, forward.kernels), nlpd_of(m))

    _, (ipls_elbo, ipls_nlpd) = jax.lax.scan(ipls_step, init_forward, None, length=max_iter)
    results["IPLS"] = (
        np.concatenate([[init_elbo], np.asarray(ipls_elbo)]),
        np.concatenate([[init_nlpd], np.asarray(ipls_nlpd)]),
    )

    print(f"  init ELBO={init_elbo:.1f} NLPD={init_nlpd:.3f}")
    print(f"  FH eps=1 final ELBO={results[('FH', 1.0)][0][-1]:.1f} NLPD={results[('FH', 1.0)][1][-1]:.3f}")

    # axis limits from the FEVS curves only (off-scale IPLS is clamped in the CSV)
    fevs_elbo = np.concatenate([results[k][0] for k in results if k != "IPLS"])
    fevs_nlpd = np.concatenate([results[k][1] for k in results if k != "IPLS"])
    elbo_ylim = (init_elbo - 800, float(np.nanmax(fevs_elbo)) + 100)
    nlpd_ylim = (float(np.nanmin(fevs_nlpd)) - 0.03, init_nlpd + 0.08)

    cs.set_style()
    # figure: ELBO (left) and calibration NLPD (right) per iteration
    grays = plt.cm.gray(np.linspace(0.0, 0.55, len(kl_steps)))
    curves = [(("FH", e), c, "-o", rf"FH ($\varepsilon={e:g}$)") for c, e in zip(grays, kl_steps)]
    curves += [(("GSLR", e), c, "--o", rf"GSLR ($\varepsilon={e:g}$)") for c, e in zip(grays, kl_steps)]
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
    fig.savefig(OUTDIR / "fig_cubic_elbo_nlpd_hyb.pdf", bbox_inches="tight")
    plt.close(fig)

    # CSVs: one column per curve, off-scale IPLS clamped to the plotted range
    columns = (
        [(("FH", e), f"FH_eps{e:g}") for e in kl_steps]
        + [(("GSLR", e), f"GSLR_eps{e:g}") for e in kl_steps]
        + [("IPLS", "IPLS")]
    )
    for which, fname, (lo, hi) in [
        (0, "results_cubic_elbo_hyb.csv", elbo_ylim),
        (1, "results_cubic_nlpd_hyb.csv", nlpd_ylim),
    ]:
        series = {col: results[key][which] for key, col in columns}
        n_iters = max(len(s) for s in series.values())
        rows = []
        for i in range(n_iters):
            row = {"iter": i}
            for col, s in series.items():
                row[col] = min(max(float(s[i]), lo), hi) if i < len(s) else ""
            rows.append(row)
        cs.write_csv(OUTDIR / fname, rows)


if __name__ == "__main__":
    main()
