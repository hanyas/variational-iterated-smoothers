"""Experiment (SV): accuracy and predictive density vs. nonlinearity (vol-of-vol sigma).

Outputs:
  outputs/fig_sv_sigma_sweep.pdf       -- two panels: RMSE | NLPD vs sigma
  outputs/results_sv_sweep.csv         -- per-trial rmse + nlpd for GSLR and FH
  outputs/results_sv_sweep_summary.csv -- per-sigma mean +/- 1 std
"""

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import _sv_common as sv
import common
import matplotlib.pyplot as plt

NUM_STEPS = 1000

SIGMAS = [0.10, 0.15, 0.20, 0.25, 0.30]

KL_STEP = 10.0
NUM_TRIALS = 10
MAX_ITER = 50

common.set_style()


def fit_all(system, ys):
    """Damped GSLR iteration and our damped FH iteration on one dataset."""
    gslr = sv.make_model_fns(system, "GSLR", "gauss_hermite")
    fh = sv.make_model_fns(system, "FH", "gauss_hermite")
    init = common.make_forward_init(system, NUM_STEPS)
    with common.silence_stdout():
        q_gslr = jax.block_until_ready(
            common.run_iterated_smoother("forward", gslr, ys, init, kl_constraint=KL_STEP, max_iterations=MAX_ITER)
        )
        q_fh = jax.block_until_ready(
            common.run_iterated_smoother("forward", fh, ys, init, kl_constraint=KL_STEP, max_iterations=MAX_ITER)
        )
    return q_gslr, q_fh


def main():
    rows = []
    sig = np.array(SIGMAS)
    rmse_keys = ["GSLR", "FH"]
    nlpd_keys = ["GSLR", "FH"]
    rmse_mean = {k: [] for k in rmse_keys}
    rmse_std = {k: [] for k in rmse_keys}
    nlpd_mean = {k: [] for k in nlpd_keys}
    nlpd_std = {k: [] for k in nlpd_keys}
    for sigma in SIGMAS:
        system = sv.make_sv_system(sigma=sigma)
        pr = {k: [] for k in rmse_keys}
        pn = {k: [] for k in nlpd_keys}
        for seed in range(1, NUM_TRIALS + 1):
            x_true, ys = sv.simulate_data(system, NUM_STEPS, np.random.RandomState(seed))
            x_flat = np.asarray(x_true).reshape(-1)
            q_gslr, q_fh = fit_all(system, ys)
            gslr_rmse, gslr_nlpd = common.rmse(q_gslr.mean, x_flat), float(common.nlpd(q_gslr, x_flat))
            fh_rmse, fh_nlpd = common.rmse(q_fh.mean, x_flat), float(common.nlpd(q_fh, x_flat))
            pr["GSLR"].append(gslr_rmse)
            pr["FH"].append(fh_rmse)
            pn["GSLR"].append(gslr_nlpd)
            pn["FH"].append(fh_nlpd)
            rows.append(
                dict(
                    sigma=sigma,
                    seed=seed,
                    gslr_rmse=gslr_rmse,
                    fh_rmse=fh_rmse,
                    gslr_nlpd=gslr_nlpd,
                    fh_nlpd=fh_nlpd,
                )
            )
        for k in rmse_keys:
            arr = np.asarray(pr[k], dtype=float)
            rmse_mean[k].append(float(np.nanmean(arr)))
            rmse_std[k].append(float(np.nanstd(arr, ddof=1)))  # 1 std across trials
        for k in nlpd_keys:
            arr = np.asarray(pn[k], dtype=float)
            nlpd_mean[k].append(float(np.nanmean(arr)))
            nlpd_std[k].append(float(np.nanstd(arr, ddof=1)))
        print(
            f"  sigma={sigma:.2f} "
            f"GSLR {rmse_mean['GSLR'][-1]:.3f} FH {rmse_mean['FH'][-1]:.3f}  |  "
            f"NLPD GSLR {nlpd_mean['GSLR'][-1]:.3f} FH {nlpd_mean['FH'][-1]:.3f}"
        )

    common.write_csv(
        sv.OUTPUT_DIR / "results_sv_sweep.csv",
        ["sigma", "seed", "gslr_rmse", "fh_rmse", "gslr_nlpd", "fh_nlpd"],
        rows,
    )
    # per-sigma mean +/- 1 std across trials (the plotted curves; ready for pgfplots)
    common.write_csv(
        sv.OUTPUT_DIR / "results_sv_sweep_summary.csv",
        [
            "sigma",
            "gslr_rmse",
            "gslr_rmse_std",
            "fh_rmse",
            "fh_rmse_std",
            "gslr_nlpd",
            "gslr_nlpd_std",
            "fh_nlpd",
            "fh_nlpd_std",
        ],
        [
            dict(
                sigma=sg,
                gslr_rmse=rmse_mean["GSLR"][i],
                gslr_rmse_std=rmse_std["GSLR"][i],
                fh_rmse=rmse_mean["FH"][i],
                fh_rmse_std=rmse_std["FH"][i],
                gslr_nlpd=nlpd_mean["GSLR"][i],
                gslr_nlpd_std=nlpd_std["GSLR"][i],
                fh_nlpd=nlpd_mean["FH"][i],
                fh_nlpd_std=nlpd_std["FH"][i],
            )
            for i, sg in enumerate(SIGMAS)
        ],
    )

    fig, (axR, axN) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    # (a) RMSE: GSLR sits on the data-ignoring baseline; FH grows far slower
    axR.errorbar(
        sig,
        rmse_mean["GSLR"],
        yerr=rmse_std["GSLR"],
        fmt="-s",
        color="0.45",
        capsize=2,
        lw=1.6,
        label="Iterated (GSLR)",
    )
    axR.errorbar(
        sig, rmse_mean["FH"], yerr=rmse_std["FH"], fmt="-o", color="0.0", capsize=2, lw=1.9, label="Iterated (FH)"
    )
    axR.set(xlabel=r"vol-of-vol $\sigma$", ylabel="RMSE to true log-volatility", title="(a) Accuracy")
    axR.legend(frameon=False, loc="upper left")
    # (b) NLPD of the true states under each marginal
    axN.errorbar(
        sig,
        nlpd_mean["GSLR"],
        yerr=nlpd_std["GSLR"],
        fmt="-s",
        color="0.45",
        capsize=2,
        lw=1.6,
        label="Iterated (GSLR)",
    )
    axN.errorbar(
        sig, nlpd_mean["FH"], yerr=nlpd_std["FH"], fmt="-o", color="0.0", capsize=2, lw=1.9, label="Iterated (FH)"
    )
    axN.set(xlabel=r"vol-of-vol $\sigma$", ylabel="NLPD of the true log-volatility", title="(b) Log score")
    axN.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    sv.save_fig(fig, "fig_sv_sigma_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
