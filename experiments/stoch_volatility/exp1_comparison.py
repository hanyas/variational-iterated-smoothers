"""Experiment (SV): smoothed log-volatility vs. the true latent state.

Outputs:
  outputs/fig_sv_trajectory.pdf     -- smoothed log-vol vs truth
  outputs/results_sv_trajectory.csv -- the plotted trajectory
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


def fit(system, ys, horizon, kl_step, max_iter):
    gslr = sv.make_model_fns(system, "GSLR", "gauss_hermite")
    fh = sv.make_model_fns(system, "FH", "gauss_hermite")
    init = sv.make_forward_init(system, horizon)
    with sv.silence_stdout():
        q_gslr = jax.block_until_ready(
            sv.run_iterated_smoother("reverse", gslr, ys, init, kl_constraint=kl_step, max_iterations=max_iter)
        )
        q_fh = jax.block_until_ready(
            sv.run_iterated_smoother("reverse", fh, ys, init, kl_constraint=kl_step, max_iterations=max_iter)
        )
    return q_gslr, q_fh


def main():
    num_steps = 1000
    data_seed = 1

    kl_step = 10.0
    max_iter = 50

    system = sv.make_sv_system()
    print(f"SV trajectory: T={num_steps}, system mu={system.mu} phi={system.phi} sigma={system.sigma}")

    x_true, ys = sv.simulate_data(system, num_steps, np.random.RandomState(data_seed))
    x_flat = np.asarray(x_true).reshape(-1)
    q_gslr, q_fh = fit(system, ys, num_steps, kl_step, max_iter)
    ks = np.arange(num_steps + 1)

    gslr_m = np.asarray(q_gslr.mean).reshape(-1)
    gslr_s = np.sqrt(np.clip(np.asarray(q_gslr.cov).reshape(-1), 0, None))
    fh_m = np.asarray(q_fh.mean).reshape(-1)
    fh_s = np.sqrt(np.clip(np.asarray(q_fh.cov).reshape(-1), 0, None))

    sv.write_csv(
        OUTDIR / "results_sv_trajectory.csv",
        [
            dict(
                k=int(kk),
                x_true=float(x_flat[i]),
                gslr_mean=float(gslr_m[i]),
                gslr_std=float(gslr_s[i]),
                fh_mean=float(fh_m[i]),
                fh_std=float(fh_s[i]),
            )
            for i, kk in enumerate(ks)
        ],
    )

    sv.set_style()
    fig, ax = plt.subplots(figsize=(13.0, 4.0))

    ax.axhline(system.mu, color="0.7", lw=0.8, zorder=0)
    ax.fill_between(
        ks,
        gslr_m - 2 * gslr_s,
        gslr_m + 2 * gslr_s,
        color="0.8",
        alpha=0.7,
        zorder=1,
        label=r"GSLR $\pm2\sigma$",
    )
    ax.fill_between(ks, fh_m - 2 * fh_s, fh_m + 2 * fh_s, color="0.45", alpha=0.4, zorder=2, label=r"FH $\pm2\sigma$")
    ax.plot(ks, x_flat, "-", color="black", lw=1.0, label="true $x_k$", zorder=4)
    ax.plot(ks, fh_m, "-", color="black", lw=1.8, label="ours (FH)", zorder=5)
    ax.plot(ks, gslr_m, ":", color="black", lw=1.3, label="ours (GSLR)", zorder=3)
    ax.set(
        xlim=(0, num_steps),
        xlabel="time step $k$",
        ylabel="log-volatility $x_k$",
        title="Stochastic volatility: smoothed log-volatility vs. truth",
    )
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_sv_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
