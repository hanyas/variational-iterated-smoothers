"""Experiment (SV): accuracy and predictive density vs. nonlinearity (vol-of-vol sigma).

Outputs:
  outputs/fig_sv_sigma_sweep.pdf       -- two panels: RMSE | NLPD vs sigma
  outputs/results_sv_sweep.csv         -- per-trial rmse + nlpd for GSLR and FH
  outputs/results_sv_sweep_summary.csv -- per-sigma mean +/- 1 std

Run from this directory: python exp2_sweep.py
"""

import jax
import numpy as np
import sv_common as sv

METHODS = ["GSLR", "FH"]


def fit_all(system, ys, num_steps, kl_constraint, max_iter):
    """Fit the reverse iterated smoother with the GSLR and FH families."""
    gslr_fns = sv.make_model_fns(system, "GSLR", "gauss_hermite")
    fh_fns = sv.make_model_fns(system, "FH", "gauss_hermite")
    init_fwd = sv.make_forward_init(system, num_steps)
    q_gslr = jax.block_until_ready(
        sv.run_iterated_smoother("reverse", gslr_fns, ys, init_fwd, kl_constraint=kl_constraint, max_iterations=max_iter)
    )
    q_fh = jax.block_until_ready(
        sv.run_iterated_smoother("reverse", fh_fns, ys, init_fwd, kl_constraint=kl_constraint, max_iterations=max_iter)
    )
    return q_gslr, q_fh


def main():
    import matplotlib.pyplot as plt

    # ---- config ----
    num_steps = 1000
    sigmas = [0.10, 0.15, 0.20, 0.25, 0.30]
    kl_constraint = 10.0
    num_trials = 10
    max_iter = 50

    # ---- run ----
    rows = []     # per-trial metrics
    summary = []  # per-sigma mean +/- 1 std, reused by the summary CSV and the plot
    for sigma in sigmas:
        system = sv.make_sv_system(sigma=sigma)
        per_trial = {m: {"rmse": [], "nlpd": []} for m in METHODS}
        for seed in range(1, num_trials + 1):
            x_true, ys = sv.simulate_data(system, num_steps, np.random.RandomState(seed))
            x_flat = np.asarray(x_true).reshape(-1)
            q_gslr, q_fh = fit_all(system, ys, num_steps, kl_constraint, max_iter)
            metric = {
                "GSLR": (sv.rmse(q_gslr.mean, x_flat), float(sv.nlpd(q_gslr, x_flat))),
                "FH": (sv.rmse(q_fh.mean, x_flat), float(sv.nlpd(q_fh, x_flat))),
            }
            for m in METHODS:
                per_trial[m]["rmse"].append(metric[m][0])
                per_trial[m]["nlpd"].append(metric[m][1])
            rows.append(
                dict(
                    sigma=sigma,
                    seed=seed,
                    gslr_rmse=metric["GSLR"][0],
                    fh_rmse=metric["FH"][0],
                    gslr_nlpd=metric["GSLR"][1],
                    fh_nlpd=metric["FH"][1],
                )
            )

        stat = {}
        for m in METHODS:
            r = np.asarray(per_trial[m]["rmse"], dtype=float)
            n = np.asarray(per_trial[m]["nlpd"], dtype=float)
            stat[m] = dict(
                rmse=float(np.nanmean(r)),
                rmse_std=float(np.nanstd(r, ddof=1)),  # 1 std across trials
                nlpd=float(np.nanmean(n)),
                nlpd_std=float(np.nanstd(n, ddof=1)),
            )
        summary.append(
            dict(
                sigma=sigma,
                gslr_rmse=stat["GSLR"]["rmse"],
                gslr_rmse_std=stat["GSLR"]["rmse_std"],
                fh_rmse=stat["FH"]["rmse"],
                fh_rmse_std=stat["FH"]["rmse_std"],
                gslr_nlpd=stat["GSLR"]["nlpd"],
                gslr_nlpd_std=stat["GSLR"]["nlpd_std"],
                fh_nlpd=stat["FH"]["nlpd"],
                fh_nlpd_std=stat["FH"]["nlpd_std"],
            )
        )
        print(
            f"  sigma={sigma:.2f} "
            f"GSLR {stat['GSLR']['rmse']:.3f} FH {stat['FH']['rmse']:.3f}  |  "
            f"NLPD GSLR {stat['GSLR']['nlpd']:.3f} FH {stat['FH']['nlpd']:.3f}"
        )

    # ---- report (figure + csv) ----
    sv.write_csv(sv.OUTPUT_DIR / "results_sv_sweep.csv", rows)
    sv.write_csv(sv.OUTPUT_DIR / "results_sv_sweep_summary.csv", summary)

    sig = np.array(sigmas)
    gslr_rmse = [r["gslr_rmse"] for r in summary]
    gslr_rmse_std = [r["gslr_rmse_std"] for r in summary]
    fh_rmse = [r["fh_rmse"] for r in summary]
    fh_rmse_std = [r["fh_rmse_std"] for r in summary]
    gslr_nlpd = [r["gslr_nlpd"] for r in summary]
    gslr_nlpd_std = [r["gslr_nlpd_std"] for r in summary]
    fh_nlpd = [r["fh_nlpd"] for r in summary]
    fh_nlpd_std = [r["fh_nlpd_std"] for r in summary]

    sv.set_style()
    fig, (axR, axN) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axR.errorbar(sig, gslr_rmse, yerr=gslr_rmse_std, fmt="-s", color="0.45", capsize=2, lw=1.6, label="Iterated (GSLR)")
    axR.errorbar(sig, fh_rmse, yerr=fh_rmse_std, fmt="-o", color="0.0", capsize=2, lw=1.9, label="Iterated (FH)")
    axR.set(xlabel=r"vol-of-vol $\sigma$", ylabel="RMSE to true log-volatility", title="(a) Accuracy")
    axR.legend(frameon=False, loc="upper left")
    axN.errorbar(sig, gslr_nlpd, yerr=gslr_nlpd_std, fmt="-s", color="0.45", capsize=2, lw=1.6, label="Iterated (GSLR)")
    axN.errorbar(sig, fh_nlpd, yerr=fh_nlpd_std, fmt="-o", color="0.0", capsize=2, lw=1.9, label="Iterated (FH)")
    axN.set(xlabel=r"vol-of-vol $\sigma$", ylabel="NLPD of the true log-volatility", title="(b) Log score")
    axN.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    sv.save_fig(fig, "fig_sv_sigma_sweep.pdf")


if __name__ == "__main__":
    main()
