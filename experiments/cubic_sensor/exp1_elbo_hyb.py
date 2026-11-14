"""Cubic sensor: HEVS (FH/GSLR, cubature), IPLS, and LM-IPLS."""

if __package__:
    from . import cubic_common as cs
else:
    import cubic_common as cs

import jax
import numpy as np

from varsmooth.baselines import lm_ipls_smoother
from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.smoothers.hybrid_markov import iterated_hybrid_markov_smoother
from varsmooth.smoothers.rts_kalman import filtering
from varsmooth.smoothers.rts_kalman import smoothing
from varsmooth.smoothers.utils import initialize_reverse_with_forward
from varsmooth.smoothers.utils import std_forward_message

DATA_SEED = 37
NUM_STEPS = 4096
KL_CONSTRAINTS = [1.0, 5.0]
MAX_ITER = 80


def main():
    """Run the experiment and write its results."""
    import matplotlib.pyplot as plt

    system = cs.make_cubic_system()
    x_true, ys = cs.simulate_data(system, NUM_STEPS, np.random.RandomState(DATA_SEED))
    elbo_of = cs.make_elbo_evaluator(system, ys)
    nlpd_of = cs.make_nlpd_evaluator(x_true)

    init_forward = cs.make_prior_chain_init(system, NUM_STEPS)
    init_reverse = initialize_reverse_with_forward(init_forward)
    init_marginals = std_forward_message(init_forward)
    init_elbo = float(elbo_of(init_marginals, init_forward.kernels))
    init_nlpd = float(nlpd_of(init_marginals))
    print(f"cubic sensor: T={NUM_STEPS}, seed={DATA_SEED}, {MAX_ITER} iterations per method", flush=True)
    results = {}

    # HEVS: original cubature FH/GSLR updates and KL radii.
    for family in ["FH", "GSLR"]:
        log_prior_fn, log_transition_fn, log_likelihood_fn = cs.make_model_fns(
            system, family, cs.CUB_QUAD if family == "FH" else cs.CUB_LINEAR
        )
        for kl in KL_CONSTRAINTS:
            print(f"  Running HEVS + {family}, eps={kl:g}", flush=True)
            _, history = iterated_hybrid_markov_smoother(
                observations=ys,
                log_prior_fn=log_prior_fn,
                log_transition_fn=log_transition_fn,
                log_likelihood_fn=log_likelihood_fn,
                init_forward_posterior=init_forward,
                init_reverse_posterior=init_reverse,
                kl_constraint=kl,
                max_iterations=MAX_ITER,
                return_history=True,
                verbose=False,
            )
            keep = np.asarray(history["feasible"]).astype(bool)
            elbo = np.asarray(jax.vmap(elbo_of)(history["forward_marginals"], history["forward_kernels"]))[
                keep
            ]
            nlpd = np.asarray(jax.vmap(nlpd_of)(history["forward_marginals"]))[keep]
            results[(family, kl)] = (np.r_[init_elbo, elbo], np.r_[init_nlpd, nlpd])

    # IPLS: undamped cubature statistical linearization followed by RTS.
    print("  Running IPLS", flush=True)
    transition_model, likelihood_model = cs.make_additive_models(system)

    def ipls_step(forward, _):
        """Apply one IPLS update and score its posterior.

        Args:
            forward: GaussMarkov
                Current forward Gaussian Markov posterior.
            _: object
                Unused callback or scan argument.

        Returns:
            posterior: GaussMarkov
                Forward Gaussian Markov posterior.
            scores: tuple
                Total ELBO and mean marginal NLPD.
        """
        m = std_forward_message(forward)
        lin_t = jax.vmap(lambda q: AffineGaussian(*cs.CUB_LINEAR(transition_model, q)))(
            Gaussian(m.mean[:-1], m.cov[:-1])
        )
        lin_o = jax.vmap(lambda q: AffineGaussian(*cs.CUB_LINEAR(likelihood_model, q)))(
            Gaussian(m.mean[1:], m.cov[1:])
        )
        forward = smoothing(lin_t, filtering(ys, system.prior, lin_t, lin_o))
        m = std_forward_message(forward)
        return forward, (elbo_of(m, forward.kernels), nlpd_of(m))

    _, (ipls_elbo, ipls_nlpd) = jax.lax.scan(ipls_step, init_forward, None, length=MAX_ITER)
    results["IPLS"] = (np.r_[init_elbo, np.asarray(ipls_elbo)], np.r_[init_nlpd, np.asarray(ipls_nlpd)])

    # LM-IPLS: same cubature rule, initial Gaussian, and scoring functions.
    # One accepted mean update per covariance update, as in Lindqvist et al.
    print("  Running LM-IPLS", flush=True)
    lm_scores = []

    @jax.jit
    def score_lm_ipls(posterior):
        """Evaluate the LM-IPLS posterior ELBO and NLPD.

        Args:
            posterior: GaussMarkov
                Forward Gaussian Markov posterior.

        Returns:
            elbo: float
                Total trajectory ELBO.
            nlpd: float
                Mean marginal negative log predictive density.
        """
        marginal = std_forward_message(posterior)
        return elbo_of(marginal, posterior.kernels), nlpd_of(marginal)

    def record_lm_ipls(_, posterior):
        # Score each native accepted RTS posterior, including its covariance.
        """Append the LM-IPLS posterior ELBO and NLPD.

        Args:
            _: object
                Unused callback or scan argument.
            posterior: GaussMarkov
                Current forward Gaussian Markov posterior.
        """
        lm_scores.append(tuple(map(float, score_lm_ipls(posterior))))

    _, lm_history = lm_ipls_smoother(
        ys,
        system.prior,
        transition_model,
        likelihood_model,
        init_forward,
        linearization_method=cs.CUB_LINEAR,
        max_iterations=MAX_ITER,
        initial_damping=0.01,
        damping_factor=10.0,
        stop_on_convergence=False,
        callback=record_lm_ipls,
    )
    scores = np.asarray(lm_scores)
    results["LM-IPLS"] = (scores[:, 0], scores[:, 1])
    final = lm_history[-1]
    print(
        f"  LM-IPLS: {final['iterations']} accepted updates, {final['total_trials']} LM trials, "
        f"converged={final['converged']}, IPLS fixed-point error={final['fixed_point_error']:.3g}; "
        f"{final['message']}",
        flush=True,
    )
    for key, (elbo, nlpd) in results.items():
        print(f"  {key}: final ELBO={elbo[-1]:.6f}, NLPD={nlpd[-1]:.6f}", flush=True)

    # Figure limits include LM-IPLS; off-scale IPLS values are clipped only by axes.
    elbos = np.concatenate([results[k][0] for k in results if k != "IPLS"])
    nlpds = np.concatenate([results[k][1] for k in results if k != "IPLS"])
    elbo_ylim = (min(init_elbo - 800, float(np.nanmin(elbos)) - 100), float(np.nanmax(elbos)) + 100)
    nlpd_ylim = (float(np.nanmin(nlpds)) - 0.03, max(init_nlpd + 0.08, float(np.nanmax(nlpds)) + 0.03))
    cs.set_style()
    grays = plt.cm.gray(np.linspace(0.0, 0.55, len(KL_CONSTRAINTS)))
    curves = [
        (("FH", kl), c, "-o", rf"HEVS + FH ($\varepsilon={kl:g}$)") for c, kl in zip(grays, KL_CONSTRAINTS)
    ]
    curves += [
        (("GSLR", kl), c, "--o", rf"HEVS + GSLR ($\varepsilon={kl:g}$)")
        for c, kl in zip(grays, KL_CONSTRAINTS)
    ]
    curves += [("IPLS", "0.15", ":s", "IPLS"), ("LM-IPLS", "tab:blue", "-^", "LM-IPLS")]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    for key, color, style, label in curves:
        elbo, nlpd = results[key]
        axL.plot(np.arange(len(elbo)), elbo, style, ms=3.5, color=color, label=label)
        axR.plot(np.arange(len(nlpd)), nlpd, style, ms=3.5, color=color, label=label)
    axL.set(xlabel="iteration", ylabel="ELBO", title="(a) Evidence lower bound", ylim=elbo_ylim)
    axR.set(xlabel="iteration", ylabel="NLPD", title="(b) Log score", ylim=nlpd_ylim)
    axL.legend(frameon=False, loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    cs.save_fig(fig, "fig_cubic_elbo_nlpd_hyb.pdf")

    # Raw scores, one column per curve; undefined Gaussian scores stay blank.
    columns = (
        [(("FH", kl), f"FH_eps{kl:g}") for kl in KL_CONSTRAINTS]
        + [(("GSLR", kl), f"GSLR_eps{kl:g}") for kl in KL_CONSTRAINTS]
        + [("IPLS", "IPLS"), ("LM-IPLS", "LM_IPLS")]
    )
    for which, metric in enumerate(("elbo", "nlpd")):
        series = {column: results[key][which] for key, column in columns}
        rows = []
        for i in range(max(len(values) for values in series.values())):
            row = {"iter": i}
            for column, values in series.items():
                row[column] = float(values[i]) if i < len(values) and np.isfinite(values[i]) else ""
            rows.append(row)
        cs.write_csv(cs.OUTPUT_DIR / f"results_cubic_{metric}_hyb.csv", rows)


if __name__ == "__main__":
    main()
