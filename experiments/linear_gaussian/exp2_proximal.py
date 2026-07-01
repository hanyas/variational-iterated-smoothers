"""Experiment 2 (LG): proximal / trust-region convergence to the exact posterior.

Writes to ./outputs (next to this script):
  fig_lg_convergence.pdf   -- avg. KL->RTS vs iteration, for several radii eps
  fig_lg_damping.pdf       -- adaptive damping beta vs iteration (log y), same eps
  results_lg_proximal.csv  -- per-iteration traces
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import lg_common as lg
import matplotlib.pyplot as plt

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)


def _converged_iter(history):
    return next((h["iter"] for h in history if h["kl_rts"] < 1e-8), None)


def _truncate(history, pad=1):
    conv = _converged_iter(history)
    if conv is None:
        return history
    return [h for h in history if h["iter"] <= conv + pad]


def main():
    dim_x, dim_y, num_steps = 3, 2, 100

    system_seed, data_seed = 0, 1
    kl_steps = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]

    max_iter = 70

    system = lg.make_lg_system(dim_x, dim_y, np.random.RandomState(system_seed))
    _, ys = lg.simulate_data(system, num_steps, np.random.RandomState(data_seed))
    rts = lg.rts_marginals(system, ys)

    fns = lg.make_model_fns(system, "GSLR", "gauss_hermite")
    init = lg.make_forward_init(system, num_steps, F_scale=0.1, Sigma_scale=1.0)
    ablation = {}
    for step in kl_steps:
        _, diags = lg.run_iterated_smoother(
            direction="forward",
            model_fns=fns,
            observations=ys,
            init_fwd_posterior=init,
            kl_constraint=step,
            max_iterations=max_iter,
            return_history=True,
        )
        ablation[step] = lg.get_markov_history(diags, rts=rts)

    conv = {step: _converged_iter(ablation[step]) for step in kl_steps}
    print(f"  iterations to converge per step: {conv}")

    lg.set_style()
    greys = plt.cm.gray(np.linspace(0.0, 0.62, len(kl_steps)))

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for (step, history), col in zip(ablation.items(), greys):
        h = _truncate(history)
        ax.semilogy(
            [x["iter"] for x in h],
            [max(x["kl_rts"], 1e-18) for x in h],
            "-",
            color=col,
            label=rf"$\varepsilon={step:.0f}$",
        )
    ax.axhline(1e-12, ls=":", color="gray", lw=1)
    ax.set(xlabel="iteration", ylabel="avg. marginal KL", title="Convergence to exact posterior")
    ax.legend(frameon=False, ncol=2, fontsize=9, title="trust-region radius")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_lg_convergence.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for (step, history), col in zip(ablation.items(), greys):
        h = _truncate(history, pad=0)
        ax.semilogy(
            [x["iter"] for x in h],
            [max(x["damping"], 1e-16) for x in h],
            "-o",
            ms=2.5,
            color=col,
            label=rf"$\varepsilon={step:.0f}$",
        )
    ax.set(
        xlabel="iteration",
        ylabel=r"damping $\beta=\alpha/(1+\alpha)$",
        title="Adaptive step size: active, then a full step",
    )
    ax.legend(frameon=False, ncol=2, fontsize=9, title="trust-region radius")
    fig.tight_layout()
    fig.savefig(OUTDIR / "fig_lg_damping.pdf", bbox_inches="tight")
    plt.close(fig)

    fields = ["direction", "iter", "eps", "damping", "kl_rts", "rmse_rts", "realized_kl"]
    rows = [dict(direction="forward", eps=eps, **x) for eps, h in ablation.items() for x in h]
    rows = [{k: r.get(k, "") for k in fields} for r in rows]
    lg.write_csv(OUTDIR / "results_lg_proximal.csv", rows)

    print(f"  wrote figures + csv to {OUTDIR}")


if __name__ == "__main__":
    main()
