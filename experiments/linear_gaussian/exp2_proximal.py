"""Experiment 2 (LG): proximal / trust-region convergence to the exact posterior.

Outputs:
  outputs/fig_lg_convergence.pdf   -- avg. KL->RTS vs iteration, for several radii eps
  outputs/fig_lg_damping.pdf       -- adaptive damping beta vs iteration (log y), same eps
  outputs/results_lg_proximal.csv  -- per-iteration traces
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import _lg_common as lg
import common
import matplotlib.pyplot as plt

# ---- configuration -----------------------------------------------------------
DIM_X, DIM_Y, NUM_STEPS = 3, 2, 100
SYSTEM_SEED, DATA_SEED = 0, 1

KL_ABLATION = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]
KL_MAIN = 100.0

MAX_ITER = 70
CONV_TOL = 1e-8

common.set_style()


def _converged_iter(history, tol=CONV_TOL):
    for h in history:
        if h["kl_rts"] < tol:
            return h["iter"]
    return None


def _truncate(history, pad=1):
    c = _converged_iter(history)
    n = (c + 1 + pad) if c is not None else len(history)
    return history[:n]


def run_traces(system, ys, rts):
    """Forward KL-constrained iteration at each trust-region radius eps."""
    fns = lg.make_model_fns(system, "GSLR", "gauss_hermite")
    init = common.make_forward_init(system, NUM_STEPS, F_scale=0.1, Sigma_scale=1.0)
    ablation = {}
    for eps in KL_ABLATION:
        _, diags = common.run_iterated_smoother(
            "forward", fns, ys, init, kl_constraint=eps, max_iterations=MAX_ITER, return_history=True
        )
        ablation[eps] = common.get_markov_history(diags, rts=rts)
    return ablation


def _eps_colors():
    # greyscale ramp: small eps dark, large eps light
    return plt.cm.gray(np.linspace(0.0, 0.62, len(KL_ABLATION)))


def fig_convergence(ablation):
    """The cliff: avg. marginal KL to RTS vs iteration, swept over the radius eps."""
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for eps, col in zip(KL_ABLATION, _eps_colors()):
        h = _truncate(ablation[eps])
        ax.semilogy(
            [x["iter"] for x in h],
            [max(x["kl_rts"], 1e-18) for x in h],
            "-",
            color=col,
            label=rf"$\varepsilon={eps:.0f}$",
        )
    ax.axhline(1e-12, ls=":", color="gray", lw=1)
    ax.set(
        xlabel="iteration",
        ylabel=r"avg. marginal KL",
        title="Convergence to exact posterior",
    )
    ax.legend(frameon=False, ncol=2, fontsize=9, title="trust-region radius")
    fig.tight_layout()
    lg.save_fig(fig, "fig_lg_convergence.pdf")
    plt.close(fig)


def fig_damping(ablation):
    """Adaptive damping beta (log y): near the cap while active, then a full step."""
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for eps, col in zip(KL_ABLATION, _eps_colors()):
        h = _truncate(ablation[eps], pad=0)
        ax.semilogy(
            [x["iter"] for x in h],
            [max(x["damping"], 1e-16) for x in h],
            "-o",
            ms=2.5,
            color=col,
            label=rf"$\varepsilon={eps:.0f}$",
        )
    ax.set(
        xlabel="iteration",
        ylabel=r"damping $\beta=\alpha/(1+\alpha)$",
        title="Adaptive step size: active, then a full step",
    )
    ax.legend(frameon=False, ncol=2, fontsize=9, title="trust-region radius")
    fig.tight_layout()
    lg.save_fig(fig, "fig_lg_damping.pdf")
    plt.close(fig)


def dump_csv(ablation):
    rows = []
    for eps, h in ablation.items():
        for x in h:
            rows.append(dict(direction="forward", eps=eps, **x))
    fields = [
        "direction",
        "iter",
        "eps",
        "damping",
        "kl_rts",
        "rmse_rts",
        "realized_kl",
    ]
    rows = [{k: r.get(k, "") for k in fields} for r in rows]
    common.write_csv(lg.OUTPUT_DIR / "results_lg_proximal.csv", fields, rows)


def main():
    system = lg.make_lg_system(DIM_X, DIM_Y, np.random.RandomState(SYSTEM_SEED))
    _, ys = lg.simulate_data(system, NUM_STEPS, np.random.RandomState(DATA_SEED))
    rts = lg.rts_marginals(system, ys)

    ablation = run_traces(system, ys, rts)

    main_h = ablation[KL_MAIN]
    print(f"  final kl-to-RTS (eps={KL_MAIN:.0f}): {_truncate(main_h)[-1]['kl_rts']:.2e}")
    conv = {eps: _converged_iter(ablation[eps]) for eps in KL_ABLATION}
    print(f"  iterations to converge per eps: {conv}")

    fig_convergence(ablation)
    fig_damping(ablation)
    dump_csv(ablation)


if __name__ == "__main__":
    main()
