from functools import partial

import jax
from jax import numpy as jnp

from varsmooth.smoothers.utils import _run_iterations
from varsmooth.smoothers.utils import free_energy
from varsmooth.smoothers.utils import line_search
from varsmooth.smoothers.utils import statistical_expansion


def make_smoother_suite(log_message_fn, std_marginal_fn, kl_fn):
    """Build a direction's smoother suite from its message-passing primitives.

    Args:
        log_message_fn: Callable
            log_forward_message (reverse smoother) or log_backward_message
            (forward smoother). Maps (log_prior, log_transition,
            log_observation, reference, damping) to (posterior, log_marg_norm,
            value_fns, log_msgs, feasible).
        std_marginal_fn: Callable
            std_backward_message (reverse) or std_forward_message (forward);
            the marginals of a Gauss-Markov chain.
        kl_fn: Callable
            kl_between_reverse_gauss_markovs (reverse) or
            kl_between_forward_gauss_markovs (forward).

    Returns:
        single_pass_smoother: Callable
            One damped message pass around a reference posterior.
        dual_objective: Callable
            The trust-region dual value at a given damping.
        log_evidence: Callable
            The undamped marginal log-evidence at the reference root.
        iterated_smoother: Callable
            The full iterated KL-constrained smoother.
    """

    def single_pass_smoother(
        observations,
        log_prior_fn,
        log_transition_fn,
        log_observation_fn,
        reference_posterior,
        temperature,
    ):
        """Run one damped message pass around a reference posterior.

        Args:
            observations: Array
                Batched observations of leading shape (T,).
            log_prior_fn: Callable
                Maps the root marginal to the quadratic log-prior over x_0.
            log_transition_fn: Callable
                Maps reference kernels and marginals to the pairwise quadratic
                log-transitions.
            log_observation_fn: Callable
                Maps observations and marginals to the quadratic
                log-observations.
            reference_posterior: GaussMarkov
                The Gauss-Markov posterior to expand around.
            temperature: float
                Trust-region temperature t; damping = t / (1 + t).

        Returns:
            GaussMarkov
                The updated Gauss-Markov posterior after a single pass.
        """
        marginals = std_marginal_fn(reference_posterior)
        log_prior, log_transition, log_observation = statistical_expansion(
            observations,
            log_prior_fn,
            log_transition_fn,
            log_observation_fn,
            reference_posterior.kernels,
            marginals,
        )
        damping = temperature / (1.0 + temperature)
        posterior, _, _, _, _ = log_message_fn(
            log_prior,
            log_transition,
            log_observation,
            reference_posterior,
            damping,
        )
        return posterior

    def dual_objective(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        kl_constraint,
        damping,
    ):
        """Evaluate the trust-region dual objective at a given damping.

        Runs one message pass and, if every step is feasible, returns
        damping * kl_constraint plus the root log-normalizer, rescaled by
        1 / (1 - damping); returns +inf if any step is infeasible.

        Args:
            log_prior: LogPrior
                Quadratic log-prior over the boundary state.
            log_transition: LogTransition
                Batched pairwise quadratic log-transitions of leading shape (T,).
            log_observation: LogObservation
                Batched quadratic log-observations of leading shape (T,).
            reference_posterior: GaussMarkov
                The Gauss-Markov posterior to expand around.
            kl_constraint: float
                Trust-region KL bound entering the dual.
            damping: float
                Trust-region damping in [0, 1); damping = t / (1 + t).

        Returns:
            Array
                The scalar dual objective value, or +inf if infeasible.
        """
        _, log_norm, _, _, feasible = log_message_fn(
            log_prior,
            log_transition,
            log_observation,
            reference_posterior,
            damping,
        )

        def _feasible_objective():
            U, u, eta = log_norm
            m, _ = reference_posterior.marginal
            dual_value = damping * kl_constraint
            dual_value += -0.5 * m.T @ U @ m + m.T @ u + eta
            return dual_value / (1.0 - damping)

        return jax.lax.cond(jnp.all(feasible), _feasible_objective, lambda: jnp.inf)

    def log_normalizer(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        damping,
    ):
        """Evaluate the marginal log-normalizer at the reference root.

        Args:
            log_prior: LogPrior
                Quadratic log-prior over the boundary state.
            log_transition: LogTransition
                Batched pairwise quadratic log-transitions of leading shape (T,).
            log_observation: LogObservation
                Batched quadratic log-observations of leading shape (T,).
            reference_posterior: GaussMarkov
                The Gauss-Markov posterior to expand around.
            damping: float
                Trust-region damping in [0, 1); damping = t / (1 + t).

        Returns:
            Array
                The scalar log-normalizer -0.5 m^T U m + m^T u + eta evaluated
                at the reference root mean m.
        """
        _, log_norm, _, _, _ = log_message_fn(
            log_prior,
            log_transition,
            log_observation,
            reference_posterior,
            damping,
        )
        U, u, eta = log_norm
        m, _ = reference_posterior.marginal
        return -0.5 * m.T @ U @ m + m.T @ u + eta

    def log_evidence(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
    ):
        """Return the undamped marginal log-evidence (log-normalizer at damping 0).

        Args:
            log_prior: LogPrior
                Quadratic log-prior over the boundary state.
            log_transition: LogTransition
                Batched pairwise quadratic log-transitions of leading shape (T,).
            log_observation: LogObservation
                Batched quadratic log-observations of leading shape (T,).
            reference_posterior: GaussMarkov
                The Gauss-Markov posterior to expand around.

        Returns:
            Array
                The scalar log-evidence at the reference root mean.
        """
        return log_normalizer(
            log_prior,
            log_transition,
            log_observation,
            reference_posterior,
            0.0,
        )

    @partial(
        jax.jit,
        static_argnames=[
            "log_prior_fn",
            "log_transition_fn",
            "log_observation_fn",
            "max_iterations",
            "return_history",
            "verbose",
        ],
    )
    def iterated_smoother(
        observations,
        log_prior_fn,
        log_transition_fn,
        log_observation_fn,
        init_posterior,
        kl_constraint,
        init_temperature=1e12,
        min_temperature=1e-12,
        max_iterations=1000,
        return_history=False,
        verbose=True,
    ):
        """Run the iterated KL-constrained smoother to convergence.

        Alternates a statistical expansion around the current posterior with a
        trust-region step, stopping early once the line-search temperature drops
        to min_temperature or max_iterations is reached.

        Args:
            observations: Array
                Batched observations of leading shape (T,).
            log_prior_fn: Callable
                Maps the root marginal to the quadratic log-prior over x_0.
            log_transition_fn: Callable
                Maps reference kernels and marginals to the pairwise quadratic
                log-transitions.
            log_observation_fn: Callable
                Maps observations and marginals to the quadratic
                log-observations.
            init_posterior: GaussMarkov
                Initial Gauss-Markov posterior to start the iterations from.
            kl_constraint: float
                Per-iteration trust-region KL bound.
            init_temperature: float
                Initial line-search temperature.
            min_temperature: float
                Early-stopping threshold on the temperature.
            max_iterations: int
                Maximum number of iterations.
            return_history: bool
                If True, run a fixed-length scan and also return stacked
                per-iteration diagnostics; disables verbose logging.
            verbose: bool
                If True (and not return_history), print per-iteration
                diagnostics.

        Returns:
            posterior: GaussMarkov
                The converged Gauss-Markov posterior.
            history: dict
                Stacked per-iteration diagnostics, returned only when
                return_history is True.
        """

        def single_iteration(reference, iteration_idx):
            marginals = std_marginal_fn(reference)
            log_prior, log_transition, log_observation = statistical_expansion(
                observations,
                log_prior_fn,
                log_transition_fn,
                log_observation_fn,
                reference.kernels,
                marginals,
            )

            def dual_objective_fn(temperature):
                damping = temperature / (1.0 + temperature)
                return dual_objective(
                    log_prior,
                    log_transition,
                    log_observation,
                    reference,
                    kl_constraint,
                    damping,
                )

            def constraint_slack_fn(temperature):
                """Constraint slack on the proposed move: kl_constraint - realized_KL."""
                damping = temperature / (1.0 + temperature)
                posterior, _, _, _, feasible_flags = log_message_fn(
                    log_prior,
                    log_transition,
                    log_observation,
                    reference,
                    damping,
                )

                def compute_slack():
                    kl_div = kl_fn(
                        marginals=std_marginal_fn(posterior),
                        gauss_markov=posterior,
                        ref_gauss_markov=reference,
                    )
                    return kl_constraint - kl_div

                return jax.lax.cond(
                    jnp.all(feasible_flags),
                    lambda _: compute_slack(),
                    lambda _: jnp.inf,
                    operand=None,
                )

            # Undamped (full) step: the proximal optimum when the trust region is
            # inactive. If it already satisfies the KL constraint, take it directly
            # -- this avoids post-convergence line-search jitter and lands cleanly
            # on the fixed point.
            full_posterior, _, _, _, full_feasible = log_message_fn(
                log_prior,
                log_transition,
                log_observation,
                reference,
                0.0,
            )
            full_feasible = jnp.all(full_feasible)
            full_kl = jax.lax.cond(
                full_feasible,
                lambda: kl_fn(
                    marginals=std_marginal_fn(full_posterior),
                    gauss_markov=full_posterior,
                    ref_gauss_markov=reference,
                ),
                lambda: jnp.inf,
            )
            take_full_step = jnp.logical_and(full_feasible, full_kl <= kl_constraint)

            temperature, dual_value, _, line_search_feasible = line_search(
                init_temperature,
                dual_objective_fn,
                constraint_slack_fn,
                rtol=0.1 * kl_constraint,
            )

            damping = temperature / (1.0 + temperature)
            candidate, _, _, _, _ = log_message_fn(
                log_prior,
                log_transition,
                log_observation,
                reference,
                damping,
            )
            ls_posterior = jax.lax.cond(
                line_search_feasible,
                lambda _: candidate,
                lambda _: reference,
                operand=None,
            )

            posterior = jax.lax.cond(
                take_full_step,
                lambda _: full_posterior,
                lambda _: ls_posterior,
                operand=None,
            )
            temperature = jnp.where(take_full_step, min_temperature, temperature)
            damping = jnp.where(take_full_step, 0.0, jnp.where(line_search_feasible, damping, 0.0))
            feasible = jnp.logical_or(take_full_step, line_search_feasible)

            new_marginals = std_marginal_fn(posterior)
            kl_div = kl_fn(
                marginals=new_marginals,
                gauss_markov=posterior,
                ref_gauss_markov=reference,
            )
            elbo_value = free_energy(
                log_prior,
                log_transition,
                log_observation,
                marginals,
                reference.kernels,
            )

            def _log_feasible(_):
                jax.debug.print(
                    "iter {iter:>4d} | damping {damp:>8.2e} | kl {kl:>8.3f} "
                    "| dual {dual:>12.3f} | elbo {elbo:>12.3f}",
                    iter=iteration_idx,
                    damp=damping,
                    kl=kl_div,
                    dual=dual_value,
                    elbo=elbo_value,
                )
                return 0

            def _log_infeasible(_):
                jax.debug.print(
                    "iter {iter:>4d} | not feasible, process might have converged",
                    iter=iteration_idx,
                )
                return 0

            if verbose and not return_history:
                jax.lax.cond(feasible, _log_feasible, _log_infeasible, operand=None)

            diagnostics = {
                "dual": dual_value,
                "elbo": elbo_value,
                "damping": damping,
                "feasible": feasible,
                "realized_kl": kl_div,
                "marginals": new_marginals,
                "kernels": posterior.kernels,
            }
            return posterior, temperature, diagnostics

        final_posterior, history = _run_iterations(
            single_iteration,
            init_posterior,
            init_temperature,
            min_temperature,
            max_iterations,
            return_history,
        )
        if return_history:
            return final_posterior, history
        return final_posterior

    return (
        single_pass_smoother,
        dual_objective,
        log_evidence,
        iterated_smoother,
    )
