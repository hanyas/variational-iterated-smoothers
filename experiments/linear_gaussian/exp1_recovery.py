"""Experiment 1 (LG): exact recovery + modularity.

Outputs:
  outputs/results_linear_gaussian.csv   -- every (trial, direction, backend) row
  outputs/table_linear_gaussian.tex     -- condensed mean +/- SE paper table
"""

import _lg_common as lg
import common
import numpy as np

# ---- configuration -----------------------------------------------------------
DIM_X, DIM_Y, NUM_STEPS = 3, 2, 100

NUM_TRIALS = 10
SYSTEM_SEED = 0
DATA_SEEDS = list(range(1, NUM_TRIALS + 1))

APPROX_BACKENDS = [("GSLR", b) for b in common.GSLR_BACKENDS] + [("FH", b) for b in common.FH_BACKENDS]
DIRECTION_LABELS = {"forward": "Forward", "reverse": "Reverse", "hybrid": "Hybrid"}


def main():
    system = lg.make_lg_system(DIM_X, DIM_Y, np.random.RandomState(SYSTEM_SEED))
    model_fns = {combo: lg.make_model_fns(system, *combo) for combo in APPROX_BACKENDS}

    rows = []
    for seed in DATA_SEEDS:
        x_true, ys = lg.simulate_data(system, NUM_STEPS, np.random.RandomState(seed))
        rts = lg.rts_marginals(system, ys)
        logZ = lg.kalman_log_evidence(system, ys)

        for family, backend in APPROX_BACKENDS:
            fns = model_fns[(family, backend)]
            for direction in common.DIRECTIONS:

                q = lg.run_single_pass(direction, fns, ys, system, NUM_STEPS, temperature=0.0)
                finite = bool(np.all(np.isfinite(q.mean)) and np.all(np.isfinite(q.cov)))
                kl_rts = common.avg_kl(q, rts) if finite else float("nan")
                rmse_rts = common.rmse(q.mean, rts.mean) if finite else float("nan")
                rmse_true = common.rmse(q.mean, x_true) if finite else float("nan")

                rows.append(
                    dict(
                        experiment="lg_recovery",
                        model="linear_gaussian",
                        direction=direction,
                        family=family,
                        backend=backend,
                        seed=seed,
                        num_steps=NUM_STEPS,
                        kl_rts=kl_rts,
                        rmse_rts=rmse_rts,
                        rmse_true=rmse_true,
                        log_evidence=logZ,
                    )
                )

    fields = [
        "experiment",
        "model",
        "direction",
        "family",
        "backend",
        "seed",
        "num_steps",
        "kl_rts",
        "rmse_rts",
        "rmse_true",
        "log_evidence",
    ]
    common.write_csv(lg.OUTPUT_DIR / "results_linear_gaussian.csv", fields, rows)
    _summary(rows)


def _select(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def _summary(rows):
    print("\n=== recovery summary ===")
    print(f"{'direction':8s} {'family':5s} {'backend':14s} {'rmse_rts':>10s} {'kl_rts':>10s}")
    for direction in common.DIRECTIONS:
        for family, backend in APPROX_BACKENDS:
            sub = _select(rows, direction=direction, family=family, backend=backend)
            r = np.nanmean([x["rmse_rts"] for x in sub])
            kl = np.nanmean([abs(x["kl_rts"]) for x in sub])
            print(f"{direction:8s} {family:5s} {backend:14s} " f"{r:10.2e} {kl:10.2e}")


if __name__ == "__main__":
    main()
