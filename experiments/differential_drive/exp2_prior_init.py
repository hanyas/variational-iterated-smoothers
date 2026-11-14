"""Differential drive: REVS, Barfoot, and Courts with observation-independent initialization."""

import numpy as np

from experiments.differential_drive import dd_common as dd
from varsmooth.baselines import barfoot
from varsmooth.baselines import courts
from varsmooth.baselines.convergence import make_stationarity_evaluator

DATA_SEED = 1
NUM_STEPS = 2000
KL_CONSTRAINTS = [5.0, 25.0]
LEARNING_RATES = [0.25, 0.5]
MAX_ITER = 300
STATIONARITY_TOLERANCE = 1e-6
FEASIBILITY_TOLERANCE = 1e-6


def main():
    """Run the experiment and write its scores and comparison plot."""
    Q, R, *_ = dd.make_parameters(**dd.PHYSICAL)
    prior = dd.make_prior()
    x_true, ys, controls = dd.simulate_data(prior, Q, R, NUM_STEPS, np.random.RandomState(DATA_SEED))
    log_prior_fn, log_transition_fn, log_likelihood_fn = dd.make_model_fns(prior, Q, R)
    elbo_of = dd.make_elbo_evaluator(ys, controls, log_prior_fn, log_transition_fn, log_likelihood_fn)
    evaluator, _ = make_stationarity_evaluator(elbo_of, NUM_STEPS, dd.STATE_SCALE)
    initial = dd.make_prior_init(prior, Q, controls)
    stopping = dict(
        max_iterations=MAX_ITER,
        evaluator=evaluator,
        stationarity_tolerance=STATIONARITY_TOLERANCE,
        feasibility_tolerance=FEASIBILITY_TOLERANCE,
    )
    results = {}
    print(
        f"Differential drive: T={NUM_STEPS}, seed={DATA_SEED}, observation-independent initialization",
        flush=True,
    )

    revs_fns = dd.make_revs_fns(prior, Q, R, controls)
    for radius in KL_CONSTRAINTS:
        print(f"  Running REVS, eps={radius:g}", flush=True)
        posterior, history = dd.run_revs(
            ys,
            *revs_fns,
            initial,
            kl_constraint=radius,
            **stopping,
        )
        results[f"REVS_eps{radius:g}"] = posterior, history

    step = barfoot.make_step(ys, controls, prior, R, log_transition_fn)
    for rate in LEARNING_RATES:
        print(f"  Running Barfoot, alpha={rate:g}", flush=True)
        posterior, history = barfoot.fit(
            observations=ys,
            controls=controls,
            prior=prior,
            R=R,
            log_prior_fn=log_prior_fn,
            log_transition_fn=log_transition_fn,
            log_likelihood_fn=log_likelihood_fn,
            initial=initial,
            step=step,
            learning_rate=rate,
            **stopping,
        )
        results[f"Barfoot_alpha{rate:g}"] = posterior, history

    print("  Running Courts", flush=True)
    posterior, history = courts.fit(
        observations=ys,
        controls=controls,
        log_prior_fn=log_prior_fn,
        log_transition_fn=log_transition_fn,
        log_likelihood_fn=log_likelihood_fn,
        initial=initial,
        state_scale=dd.STATE_SCALE,
        **stopping,
    )
    results["Courts"] = posterior, history

    dd.save_results(
        results, "prior", DATA_SEED, NUM_STEPS, MAX_ITER, STATIONARITY_TOLERANCE, FEASIBILITY_TOLERANCE
    )
    dd.plot_comparison(dd.OUTPUT_DIR)
    if not all(history[-1]["converged"] for _, history in results.values()):
        raise SystemExit("At least one method did not converge; all outcomes were saved.")


if __name__ == "__main__":
    main()
