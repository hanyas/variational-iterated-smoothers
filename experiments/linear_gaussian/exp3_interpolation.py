"""Experiment 3 (LG): entropic-proximal beta-interpolation.

Outputs:
  outputs/fig_lg_interpolation.pdf       -- m_k[d] vs k, one subplot per coordinate
  outputs/results_lg_interpolation.csv   -- the plotted trajectories
"""

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import _lg_common as lg
import common
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from varsmooth.objects import Gaussian
from varsmooth.smoothers.utils import std_forward_message

DIM_X, DIM_Y, NUM_STEPS = 2, 2, 100
DATA_SEED = 1

DIRECTION = "forward"
INIT_POST = dict(F_scale=0.1, Sigma_scale=1.0)
BETAS = np.concatenate([np.linspace(0.0, 0.9, 19), [0.95, 0.99, 0.999, 0.9999]])

common.set_style()
GREY = LinearSegmentedColormap.from_list("grey_seq", plt.cm.gray(np.linspace(0.08, 0.72, 256)))


def make_smooth_system(rho=0.985, theta=0.16, q=0.05, r=0.25, r0=4.0):
    """A 2-D damped rotation: a smooth inward spiral with low process noise."""
    c, s = np.cos(theta), np.sin(theta)
    A = rho * np.array([[c, -s], [s, c]])
    b = np.zeros(2)
    Omega = (q**2) * np.eye(2)
    H = np.eye(2)
    e = np.zeros(2)
    Delta = (r**2) * np.eye(2)
    prior = Gaussian(jnp.asarray([r0, 0.0]), jnp.asarray(0.01 * np.eye(2)))
    return lg.LGSystem(prior, A, b, Omega, H, e, Delta)


def plot_interpolation(q_betas, rts, q_init):
    """m_k[d] vs time k, one subplot per state coordinate."""
    fan_colors = GREY(np.linspace(0.0, 1.0, len(BETAS)))
    ks = np.arange(NUM_STEPS + 1)
    fig, axes = plt.subplots(1, DIM_X, figsize=(5.4 * DIM_X, 4.2), sharex=True, constrained_layout=True)

    for comp, ax in enumerate(axes):
        for q, col in zip(q_betas, fan_colors):
            ax.plot(ks, np.asarray(q.mean)[:, comp], color=col, lw=0.9, alpha=0.8)
        ax.plot(ks, np.asarray(rts.mean)[:, comp], "k-", lw=2.4, label=r"RTS ($\beta=0$)")
        ax.plot(ks, np.asarray(q_init.mean)[:, comp], "k--", lw=2.0, label=r"init ($\beta\to1$)")
        ax.set(xlabel="time step $k$", ylabel=rf"smoothed mean $m_k[{comp}]$", title=rf"State dimension {comp}")
        ax.legend(frameon=False, loc="best", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=GREY, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
    cb.set_label(r"damping $\beta$")
    fig.suptitle(r"Geometric $\beta$-interpolation: exact posterior $\leftrightarrow$ init")
    lg.save_fig(fig, "fig_lg_interpolation.pdf")
    plt.close(fig)


def dump_csv(q_betas, rts, q_init):
    """Long-format trajectories behind the figure: one row per (series, beta, k)."""
    ks = np.arange(NUM_STEPS + 1)
    rows = []
    for bi, (beta, q) in enumerate(zip(BETAS, q_betas)):
        m = np.asarray(q.mean)
        for k in ks:
            rows.append(dict(series="sweep", bi=bi, beta=float(beta), k=int(k), m0=float(m[k, 0]), m1=float(m[k, 1])))
    for series, ref in (("rts", rts), ("init", q_init)):
        m = np.asarray(ref.mean)
        for k in ks:
            rows.append(dict(series=series, bi=-1, beta="", k=int(k), m0=float(m[k, 0]), m1=float(m[k, 1])))
    common.write_csv(lg.OUTPUT_DIR / "results_lg_interpolation.csv", ["series", "bi", "beta", "k", "m0", "m1"], rows)


def main():
    system = make_smooth_system()
    _, ys = lg.simulate_data(system, NUM_STEPS, np.random.RandomState(DATA_SEED))
    rts = lg.rts_marginals(system, ys)
    fns = lg.make_model_fns(system, "GSLR", "gauss_hermite")
    q_init = std_forward_message(common.make_forward_init(system, NUM_STEPS, **INIT_POST))

    q_betas, kl_to_rts, kl_to_init = [], [], []
    for beta in BETAS:
        temperature = beta / (1.0 - beta)
        q = lg.run_single_pass(DIRECTION, fns, ys, system, NUM_STEPS, temperature=temperature, init_kwargs=INIT_POST)
        q_betas.append(q)
        kl_to_rts.append(common.avg_kl(q, rts))
        kl_to_init.append(common.avg_kl(q, q_init))
    print(f"  beta=0   : KL->RTS={max(kl_to_rts[0], 1e-18):.2e}  KL->init={kl_to_init[0]:.2e}")
    print(f"  beta=1-  : KL->RTS={kl_to_rts[-1]:.2e}  KL->init={max(kl_to_init[-1], 1e-18):.2e}")

    plot_interpolation(q_betas, rts, q_init)
    dump_csv(q_betas, rts, q_init)


if __name__ == "__main__":
    main()
