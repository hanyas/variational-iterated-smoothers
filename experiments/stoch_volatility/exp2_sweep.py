"""Experiment (SV): accuracy and predictive density vs. nonlinearity (vol-of-vol sigma).

Outputs:
  outputs/fig_sv_sigma_sweep.pdf       -- two panels: RMSE | NLPD vs sigma
  outputs/results_sv_sweep.csv         -- per-trial rmse + nlpd for GSLR and FH
  outputs/results_sv_sweep_summary.csv -- per-sigma mean +/- 1 std
"""

from pathlib import Path

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sv_common as sv

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


def fit_all(system, ys, num_steps, kl_step, max_iter):
    gslr_fns = sv.make_model_fns(system, "GSLR", "gauss_hermite")
    fh_fns = sv.make_model_fns(system, "FH", "gauss_hermite")
    init_fwd = sv.make_forward_init(system, num_steps)

    with sv.silence_stdout():
        q_gslr = jax.block_until_ready(
            sv.run_iterated_smoother(
                direction="reverse",
                model_fns=gslr_fns,
                observations=ys,
                init_fwd_posterior=init_fwd,
                kl_constraint=kl_step,
                max_iterations=max_iter,
            )
        )
        q_fh = jax.block_until_ready(
            sv.run_iterated_smoother(
                direction="reverse",
                model_fns=fh_fns,
                observations=ys,
                init_fwd_posterior=init_fwd,
                kl_constraint=kl_step,
                max_iterations=max_iter,
            )
        )
    return q_gslr, q_fh


def main():
    num_steps = 1000

    sigmas = [0.10, 0.15, 0.20, 0.25, 0.30]

    kl_step = 10.0
    num_trials = 10
    max_iter = 50

    rows = []
    rmse_keys = ["GSLR", "FH"]
    nlpd_keys = ["GSLR", "FH"]
    rmse_mean = {k: [] for k in rmse_keys}
    rmse_std = {k: [] for k in rmse_keys}
    nlpd_mean = {k: [] for k in nlpd_keys}
    nlpd_std = {k: [] for k in nlpd_keys}

    for sigma in sigmas:
        system = sv.make_sv_system(sigma=sigma)
        pr = {k: [] for k in rmse_keys}
        pn = {k: [] for k in nlpd_keys}
        for seed in range(1, num_trials + 1):
            x_true, ys = sv.simulate_data(system, num_steps, np.random.RandomState(seed))
            x_flat = np.asarray(x_true).reshape(-1)
            q_gslr, q_fh = fit_all(system, ys, num_steps, kl_step, max_iter)
            gslr_rmse, gslr_nlpd = sv.rmse(q_gslr.mean, x_flat), float(sv.nlpd(q_gslr, x_flat))
            fh_rmse, fh_nlpd = sv.rmse(q_fh.mean, x_flat), float(sv.nlpd(q_fh, x_flat))
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

    sv.write_csv(OUTDIR / "results_sv_sweep.csv", rows)
    sv.write_csv(
        OUTDIR / "results_sv_sweep_summary.csv",
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
            for i, sg in enumerate(sigmas)
        ],
    )

    sv.set_style()
    fig, (axR, axN) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axR.errorbar(
        np.array(sigmas),
        rmse_mean["GSLR"],
        yerr=rmse_std["GSLR"],
        fmt="-s",
        color="0.45",
        capsize=2,
        lw=1.6,
        label="Iterated (GSLR)",
    )
    axR.errorbar(
        np.array(sigmas),
        rmse_mean["FH"],
        yerr=rmse_std["FH"],
        fmt="-o",
        color="0.0",
        capsize=2,
        lw=1.9,
        label="Iterated (FH)",
    )
    axR.set(xlabel=r"vol-of-vol $\sigma$", ylabel="RMSE to true log-volatility", title="(a) Accuracy")
    axR.legend(frameon=False, loc="upper left")
    axN.errorbar(
        np.array(sigmas),
        nlpd_mean["GSLR"],
        yerr=nlpd_std["GSLR"],
        fmt="-s",
        color="0.45",
        capsize=2,
        lw=1.6,
        label="Iterated (GSLR)",
    )
    axN.errorbar(
        np.array(sigmas),
        nlpd_mean["FH"],
        yerr=nlpd_std["FH"],
        fmt="-o",
        color="0.0",
        capsize=2,
        lw=1.9,
        label="Iterated (FH)",
    )
    axN.set(xlabel=r"vol-of-vol $\sigma$", ylabel="NLPD of the true log-volatility", title="(b) Log score")
    axN.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_sv_sigma_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
