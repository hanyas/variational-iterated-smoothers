"""Experiment 1 (LG): exact recovery of the RTS smoother across directions and backends.

Outputs:
  outputs/results_linear_gaussian.csv   -- one row per (trial, direction, family, backend)

Run from this directory: python exp1_recovery.py
"""

import lg_common as lg
import numpy as np

BACKENDS = [("GSLR", b) for b in lg.GSLR_BACKENDS] + [("FH", b) for b in lg.FH_BACKENDS]
DIRECTIONS = ("forward", "reverse", "hybrid")


def main():
    # ---- config ----
    num_steps = 100
    num_trials = 10
    data_seeds = range(1, num_trials + 1)

    # ---- data ----
    system = lg.make_linear_system()
    model_fns = {combo: lg.make_model_fns(system, *combo) for combo in BACKENDS}
    init = lg.make_forward_init(system, num_steps)

    # ---- run ----
    results = []
    for seed in data_seeds:
        x_true, ys = lg.simulate_data(system, num_steps, np.random.RandomState(seed))
        rts = lg.rts_marginals(system, ys)
        logZ = lg.kalman_log_evidence(system, ys)

        for family, backend in BACKENDS:
            fns = model_fns[(family, backend)]

            for direction in DIRECTIONS:
                q = lg.run_single_pass(direction, fns, ys, init, temperature=0.0)
                finite = bool(np.all(np.isfinite(q.mean)) and np.all(np.isfinite(q.cov)))
                kl_rts = lg.avg_kl(q, rts) if finite else float("nan")
                rmse_rts = lg.rmse(q.mean, rts.mean) if finite else float("nan")
                rmse_true = lg.rmse(q.mean, x_true) if finite else float("nan")

                results.append(
                    dict(
                        experiment="lg_recovery",
                        model="linear_gaussian",
                        direction=direction,
                        family=family,
                        backend=backend,
                        seed=seed,
                        kl_rts=kl_rts,
                        rmse_rts=rmse_rts,
                        rmse_true=rmse_true,
                        log_evidence=logZ,
                    )
                )

    # ---- report (csv) ----
    lg.write_csv(lg.OUTPUT_DIR / "results_linear_gaussian.csv", results)
    print_summary(results)


def _select(rows, **kw):
    # rows matching every key=value filter in kw
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def print_summary(results):
    """Print the mean RMSE-to-RTS and KL-to-RTS per (direction, family, backend)."""
    print("\n=== recovery summary ===")
    print(f"{'direction':8s} {'family':5s} {'backend':14s} {'rmse_rts':>10s} {'kl_rts':>10s}")
    for direction in DIRECTIONS:
        for family, backend in BACKENDS:
            sub = _select(results, direction=direction, family=family, backend=backend)
            r = np.nanmean([x["rmse_rts"] for x in sub])
            kl = np.nanmean([abs(x["kl_rts"]) for x in sub])
            print(f"{direction:8s} {family:5s} {backend:14s} {r:10.2e} {kl:10.2e}")


if __name__ == "__main__":
    main()
