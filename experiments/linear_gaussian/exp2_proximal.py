"""Experiment 2 (LG): proximal / trust-region convergence to the exact posterior.

Outputs:
  outputs/fig_lg_convergence.pdf   -- avg. marginal KL->RTS vs iteration, per trust-region radius
  outputs/fig_lg_damping.pdf       -- adaptive damping beta vs iteration (log y), per radius
  outputs/results_lg_proximal.csv  -- per-iteration traces

Run from this directory: python exp2_proximal.py
"""

import lg_common as lg
import numpy as np

CONV_THRESHOLD = 1e-8  # avg. marginal KL->RTS below which an iteration counts as converged
KL_FLOOR = 1e-18       # semilog floor for the KL->RTS curve
DAMPING_FLOOR = 1e-16  # semilog floor for the damping curve


def _converged_iter(history):
    return next((h["iter"] for h in history if h["kl_rts"] < CONV_THRESHOLD), None)


def _truncate(history, pad=1):
    conv = _converged_iter(history)
    if conv is None:
        return history
    return [h for h in history if h["iter"] <= conv + pad]


def main():
    import matplotlib.pyplot as plt

    # ---- config ----
    num_steps = 100
    data_seed = 1
    kl_constraints = [5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0]
    max_iter = 70

    # ---- data ----
    system = lg.make_linear_system()
    _, ys = lg.simulate_data(system, num_steps, np.random.RandomState(data_seed))
    rts = lg.rts_marginals(system, ys)

    # ---- run ----
    fns = lg.make_model_fns(system, "GSLR", "gauss_hermite")
    init = lg.make_forward_init(system, num_steps, F_scale=0.1, Sigma_scale=1.0)
    ablation = {}
    for kl in kl_constraints:
        _, diags = lg.run_iterated_smoother(
            direction="forward",
            model_fns=fns,
            observations=ys,
            init_fwd_posterior=init,
            kl_constraint=kl,
            max_iterations=max_iter,
            return_history=True,
        )
        ablation[kl] = lg.get_markov_history(diags, rts=rts)

    conv = {kl: _converged_iter(ablation[kl]) for kl in kl_constraints}
    print(f"  iterations to converge per radius: {conv}")

    # ---- report (figures + csv) ----
    lg.set_style()
    greys = plt.cm.gray(np.linspace(0.0, 0.62, len(kl_constraints)))

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for (kl, history), col in zip(ablation.items(), greys):
        h = _truncate(history)
        ax.semilogy(
            [x["iter"] for x in h],
            [max(x["kl_rts"], KL_FLOOR) for x in h],
            "-",
            color=col,
            label=rf"$\varepsilon={kl:.0f}$",
        )
    ax.set(xlabel="iteration", ylabel="avg. marginal KL", title="Convergence to exact posterior")
    ax.legend(frameon=False, ncol=2, fontsize=9, title="trust-region radius")
    fig.tight_layout()
    lg.save_fig(fig, "fig_lg_convergence.pdf")

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for (kl, history), col in zip(ablation.items(), greys):
        h = _truncate(history, pad=0)
        ax.semilogy(
            [x["iter"] for x in h],
            [max(x["damping"], DAMPING_FLOOR) for x in h],
            "-o",
            ms=2.5,
            color=col,
            label=rf"$\varepsilon={kl:.0f}$",
        )
    ax.set(
        xlabel="iteration",
        ylabel=r"damping $\beta=\alpha/(1+\alpha)$",
        title="Adaptive step size: active, then a full step",
    )
    ax.legend(frameon=False, ncol=2, fontsize=9, title="trust-region radius")
    fig.tight_layout()
    lg.save_fig(fig, "fig_lg_damping.pdf")

    fields = ["direction", "iter", "eps", "damping", "kl_rts", "rmse_rts", "realized_kl"]
    rows = [dict(direction="forward", eps=kl, **x) for kl, h in ablation.items() for x in h]
    rows = [{k: r.get(k, "") for k in fields} for r in rows]
    lg.write_csv(lg.OUTPUT_DIR / "results_lg_proximal.csv", rows)


if __name__ == "__main__":
    main()
