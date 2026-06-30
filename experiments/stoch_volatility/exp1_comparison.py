"""Experiment (SV): smoothed log-volatility vs. the true latent state.

Outputs:
  outputs/fig_sv_trajectory.pdf     -- smoothed log-vol vs truth
  outputs/results_sv_trajectory.csv -- the plotted trajectory
"""

import jax
import matplotlib
import numpy as np

matplotlib.use("Agg")
import _sv_common as sv
import common
import matplotlib.pyplot as plt

# ---- configuration -----------------------------------------------------------
SYSTEM = sv.make_sv_system()

NUM_STEPS = 1000
DATA_SEED = 1

KL_STEP = 10.0
MAX_ITER = 50

common.set_style()


def fit(system, ys, horizon):
    """Smoothed marginals for the damped GSLR and FH iterations on one dataset."""
    gslr = sv.make_model_fns(system, "GSLR", "gauss_hermite")
    fh = sv.make_model_fns(system, "FH", "gauss_hermite")
    init = common.make_forward_init(system, horizon)
    with common.silence_stdout():
        q_gslr = jax.block_until_ready(
            common.run_iterated_smoother("forward", gslr, ys, init, kl_constraint=KL_STEP, max_iterations=MAX_ITER)
        )
        q_fh = jax.block_until_ready(
            common.run_iterated_smoother("forward", fh, ys, init, kl_constraint=KL_STEP, max_iterations=MAX_ITER)
        )
    return q_gslr, q_fh


def main():
    print(f"SV trajectory: T={NUM_STEPS}, system mu={SYSTEM.mu} phi={SYSTEM.phi} sigma={SYSTEM.sigma}")
    x_true, ys = sv.simulate_data(SYSTEM, NUM_STEPS, np.random.RandomState(DATA_SEED))
    x_flat = np.asarray(x_true).reshape(-1)
    q_gslr, q_fh = fit(SYSTEM, ys, NUM_STEPS)
    ks = np.arange(NUM_STEPS + 1)

    gslr_m = np.asarray(q_gslr.mean).reshape(-1)
    gslr_s = np.sqrt(np.clip(np.asarray(q_gslr.cov).reshape(-1), 0, None))
    fh_m = np.asarray(q_fh.mean).reshape(-1)
    fh_s = np.sqrt(np.clip(np.asarray(q_fh.cov).reshape(-1), 0, None))

    common.write_csv(
        sv.OUTPUT_DIR / "results_sv_trajectory.csv",
        ["k", "x_true", "gslr_mean", "gslr_std", "fh_mean", "fh_std"],
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

    fig, ax = plt.subplots(figsize=(13.0, 4.0))
    ax.axhline(SYSTEM.mu, color="0.7", lw=0.8, zorder=0)
    ax.fill_between(ks, gslr_m - 2 * gslr_s, gslr_m + 2 * gslr_s, color="0.8", alpha=0.7, zorder=1, label=r"GSLR $\pm2\sigma$")
    ax.fill_between(ks, fh_m - 2 * fh_s, fh_m + 2 * fh_s, color="0.45", alpha=0.4, zorder=2, label=r"FH $\pm2\sigma$")
    ax.plot(ks, x_flat, "-", color="black", lw=1.0, label="true $x_k$", zorder=4)
    ax.plot(ks, fh_m, "-", color="black", lw=1.8, label="ours (FH)", zorder=5)
    ax.plot(ks, gslr_m, ":", color="black", lw=1.3, label="damped (GSLR)", zorder=3)
    ax.set(
        xlim=(0, NUM_STEPS),
        xlabel="time step $k$",
        ylabel="log-volatility $x_k$",
        title="Stochastic volatility: smoothed log-volatility vs. truth",
    )
    ax.legend(frameon=False, ncol=3, fontsize=9, loc="best")
    fig.tight_layout()
    sv.save_fig(fig, "fig_sv_trajectory.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
