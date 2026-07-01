"""Experiment 1 (LG): exact recovery + modularity.

Outputs:
  outputs/results_linear_gaussian.csv   -- every (trial, direction, backend) row
  outputs/table_linear_gaussian.tex     -- condensed mean +/- SE paper table
"""

from pathlib import Path

import lg_common as lg
import numpy as np

OUTDIR = Path(__file__).resolve().parent / "outputs"
OUTDIR.mkdir(exist_ok=True)

BACKENDS = [("GSLR", b) for b in lg.GSLR_BACKENDS] + [("FH", b) for b in lg.FH_BACKENDS]
DIRECTIONS = ("forward", "reverse", "hybrid")


def main():
    dim_x, dim_y, num_steps = 3, 2, 100

    num_trials = 10
    system_seed = 0
    data_seeds = range(1, num_trials + 1)

    system = lg.make_lg_system(dim_x, dim_y, np.random.RandomState(system_seed))
    model_fns = {combo: lg.make_model_fns(system, *combo) for combo in BACKENDS}

    results = []
    for seed in data_seeds:
        x_true, ys = lg.simulate_data(system, num_steps, np.random.RandomState(seed))
        rts = lg.rts_marginals(system, ys)
        logZ = lg.kalman_log_evidence(system, ys)

        for family, backend in BACKENDS:
            fns = model_fns[(family, backend)]

            for direction in DIRECTIONS:
                q = lg.run_single_pass(direction, fns, ys, system, num_steps, temperature=0.0)
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

    lg.write_csv(OUTDIR / "results_linear_gaussian.csv", results)
    get_summary(results)


def _select(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


def get_summary(results):
    print("\n=== recovery summary ===")
    print(f"{'direction':8s} {'family':5s} {'backend':14s} {'rmse_rts':>10s} {'kl_rts':>10s}")
    for direction in DIRECTIONS:
        for family, backend in BACKENDS:
            sub = _select(results, direction=direction, family=family, backend=backend)
            r = np.nanmean([x["rmse_rts"] for x in sub])
            kl = np.nanmean([abs(x["kl_rts"]) for x in sub])
            print(f"{direction:8s} {family:5s} {backend:14s} " f"{r:10.2e} {kl:10.2e}")


if __name__ == "__main__":
    main()
