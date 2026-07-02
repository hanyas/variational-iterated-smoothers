"""Experiment 3 (LG): entropic-proximal beta-interpolation.

Outputs:
  outputs/fig_lg_interpolation.pdf       -- m_k[d] vs k, one subplot per coordinate
  outputs/results_lg_interpolation.csv   -- the plotted trajectories
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import lg_common as lg
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

from varsmooth.smoothers.utils import std_forward_message

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


def main():
    dim_x, num_steps = 2, 100
    data_seed = 1

    betas = np.concatenate([np.linspace(0.0, 0.9, 19), [0.95, 0.99, 0.999, 0.9999]])

    system = lg.make_linear_system()
    _, ys = lg.simulate_data(system, num_steps, np.random.RandomState(data_seed))
    rts = lg.rts_marginals(system, ys)

    fns = lg.make_model_fns(system, "GSLR", "gauss_hermite")
    init_post_kwargs = dict(F_scale=0.1, Sigma_scale=1.0)
    q_init = std_forward_message(lg.make_forward_init(system, num_steps, **init_post_kwargs))

    q_betas, kl_to_rts, kl_to_init = [], [], []
    for beta in betas:
        temperature = beta / (1.0 - beta)
        q = lg.run_single_pass(
            direction="forward",
            model_fns=fns,
            observations=ys,
            system=system,
            num_steps=num_steps,
            temperature=temperature,
            init_kwargs=init_post_kwargs,
        )
        q_betas.append(q)
        kl_to_rts.append(lg.avg_kl(q, rts))
        kl_to_init.append(lg.avg_kl(q, q_init))

    print(f"  beta=0   : KL->RTS={max(kl_to_rts[0], 1e-18):.2e}  KL->init={kl_to_init[0]:.2e}")
    print(f"  beta=1-  : KL->RTS={kl_to_rts[-1]:.2e}  KL->init={max(kl_to_init[-1], 1e-18):.2e}")

    lg.set_style()
    ks = np.arange(num_steps + 1)
    greys = LinearSegmentedColormap.from_list("grey_seq", plt.cm.gray(np.linspace(0.08, 0.72, 256)))

    fan_colors = greys(np.linspace(0.0, 1.0, len(betas)))
    fig, axes = plt.subplots(1, dim_x, figsize=(5.4 * dim_x, 4.2), sharex=True, constrained_layout=True)
    for comp, ax in enumerate(axes):
        for q, col in zip(q_betas, fan_colors):
            ax.plot(ks, np.asarray(q.mean)[:, comp], color=col, lw=0.9, alpha=0.8)
        ax.plot(ks, np.asarray(rts.mean)[:, comp], "k-", lw=2.4, label=r"RTS ($\beta=0$)")
        ax.plot(ks, np.asarray(q_init.mean)[:, comp], "k--", lw=2.0, label=r"init ($\beta\to1$)")
        ax.set(xlabel="time step $k$", ylabel=rf"smoothed mean $m_k[{comp}]$", title=rf"State dimension {comp}")
        ax.legend(frameon=False, loc="best", fontsize=9)

    sm = plt.cm.ScalarMappable(cmap=greys, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.046, pad=0.02)
    cb.set_label(r"damping $\beta$")
    fig.suptitle(r"Geometric $\beta$-interpolation: exact posterior $\leftrightarrow$ init")
    fig.savefig(OUTDIR / "fig_lg_interpolation.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for bi, (beta, q) in enumerate(zip(betas, q_betas)):
        m = np.asarray(q.mean)
        for k in ks:
            rows.append(dict(series="sweep", bi=bi, beta=float(beta), k=int(k), m0=float(m[k, 0]), m1=float(m[k, 1])))
    for series, ref in (("rts", rts), ("init", q_init)):
        m = np.asarray(ref.mean)
        for k in ks:
            rows.append(dict(series=series, bi=-1, beta="", k=int(k), m0=float(m[k, 0]), m1=float(m[k, 1])))
    lg.write_csv(OUTDIR / "results_lg_interpolation.csv", rows)


if __name__ == "__main__":
    main()
