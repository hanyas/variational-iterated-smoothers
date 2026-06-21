from functools import partial

import jax
from jax import numpy as jnp

from varsmooth.smoothers.utils import line_search, statistical_expansion
from varsmooth.utils import bounded_while_loop


def make_smoother_suite(log_message_fn, std_marginal_fn, kl_fn):
    """Build a direction's smoother suite from its message-passing primitives.

    The forward and reverse iterated smoothers share identical orchestration and
    differ only in three primitives. This factory captures those and returns the
    full set of public entry points for one direction.

    Args:
        log_message_fn: ``log_forward_message`` (reverse smoother) or
            ``log_backward_message`` (forward smoother). Maps
            ``(log_prior, log_transition, log_observation, reference, damping)``
            to ``(posterior, log_marg_norm, value_fns, log_msgs, feasible)``.
        std_marginal_fn: ``std_backward_message`` (reverse) or
            ``std_forward_message`` (forward); marginals of a Gauss-Markov chain.
        kl_fn: ``kl_between_reverse_gauss_markovs`` (reverse) or
            ``kl_between_forward_gauss_markovs`` (forward).

    Returns:
        Tuple ``(single_pass_smoother, dual_objective, vanilla_objective,
        iterated_smoother, undamped_iterated_smoother)``.
    """

    def single_pass_smoother(
        observations,
        log_prior_fn,
        log_transition_fn,
        log_observation_fn,
        reference_posterior,
        temperature,
    ):
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
        posterior, _, _, _, _ = log_message_fn(log_prior, log_transition, log_observation, reference_posterior, damping)
        return posterior

    def dual_objective(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
        kl_constraint,
        damping,
    ):
        _, log_norm, _, _, feasible = log_message_fn(
            log_prior, log_transition, log_observation, reference_posterior, damping
        )

        def _feasible_objective():
            U, u, eta = log_norm
            m, _ = reference_posterior.marginal
            dual_value = damping * kl_constraint
            dual_value += -0.5 * m.T @ U @ m + m.T @ u + eta
            return dual_value / (1.0 - damping)

        return jax.lax.cond(jnp.all(feasible), _feasible_objective, lambda: jnp.inf)

    def vanilla_objective(
        log_prior,
        log_transition,
        log_observation,
        reference_posterior,
    ):
        _, log_norm, _, _, _ = log_message_fn(log_prior, log_transition, log_observation, reference_posterior, 0.0)
        U, u, eta = log_norm
        m, _ = reference_posterior.marginal
        return -0.5 * m.T @ U @ m + m.T @ u + eta

    @partial(
        jax.jit,
        static_argnames=[
            "log_prior_fn",
            "log_transition_fn",
            "log_observation_fn",
            "max_iterations",
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
    ):
        """Iterated KL-constrained smoother with temperature-based early stopping.

        Iterations stop when ``max_iterations`` is reached or the line-search
        temperature drops below ``min_temperature`` (i.e. convergence).
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

            def dual_gradient_fn(temperature):
                damping = temperature / (1.0 + temperature)
                posterior, _, _, _, feasible_pass = log_message_fn(
                    log_prior,
                    log_transition,
                    log_observation,
                    reference,
                    damping,
                )

                def compute_gradient():
                    kl_div = kl_fn(
                        marginals=std_marginal_fn(posterior),
                        gauss_markov=posterior,
                        ref_gauss_markov=reference,
                    )
                    return kl_constraint - kl_div

                return jax.lax.cond(
                    jnp.all(feasible_pass),
                    lambda _: compute_gradient(),
                    lambda _: jnp.inf,
                    operand=None,
                )

            temperature, dual_value, _, line_search_feasible = line_search(
                init_temperature,
                dual_objective_fn,
                dual_gradient_fn,
                rtol=0.1 * kl_constraint,
            )

            def apply_optimal_solution():
                damping = temperature / (1.0 + temperature)
                posterior, _, _, _, _ = log_message_fn(
                    log_prior,
                    log_transition,
                    log_observation,
                    reference,
                    damping,
                )
                kl_div = kl_fn(
                    marginals=std_marginal_fn(posterior),
                    gauss_markov=posterior,
                    ref_gauss_markov=reference,
                )
                obj_value = vanilla_objective(
                    log_prior,
                    log_transition,
                    log_observation,
                    posterior,
                )
                jax.debug.print(
                    "iter {iter:>4d} | damping {damp:>8.2e} | kl {kl:>8.3f} "
                    "| dual {dual:>12.3f} | val {val:>12.3f}",
                    iter=iteration_idx,
                    damp=damping,
                    kl=kl_div,
                    dual=dual_value,
                    val=obj_value,
                )
                return posterior

            def use_reference():
                jax.debug.print(
                    "iter {iter:>4d} | not feasible, process might have converged",
                    iter=iteration_idx,
                )
                return reference

            posterior = jax.lax.cond(
                line_search_feasible,
                lambda _: apply_optimal_solution(),
                lambda _: use_reference(),
                operand=None,
            )
            return posterior, temperature

        def iteration_body(carry):
            current_posterior, iteration_count, _ = carry
            next_posterior, next_temperature = single_iteration(current_posterior, iteration_count)
            return next_posterior, iteration_count + 1, next_temperature

        def iteration_condition(carry):
            _, iteration_count, next_temperature = carry
            return jnp.logical_and(
                iteration_count < max_iterations,
                next_temperature > min_temperature,
            )

        optimal_posterior, _, _ = bounded_while_loop(
            cond_fun=iteration_condition,
            body_fun=iteration_body,
            init_val=(init_posterior, 0, init_temperature),
            maxiter=max_iterations,
        )
        return optimal_posterior

    @partial(
        jax.jit,
        static_argnames=[
            "log_prior_fn",
            "log_transition_fn",
            "log_observation_fn",
            "max_iterations",
        ],
    )
    def undamped_iterated_smoother(
        observations,
        log_prior_fn,
        log_transition_fn,
        log_observation_fn,
        init_posterior,
        max_iterations=1000,
    ):
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
            optimal_posterior, _, _, _, _ = log_message_fn(
                log_prior,
                log_transition,
                log_observation,
                reference,
                0.0,
            )
            kl_div = kl_fn(
                marginals=std_marginal_fn(optimal_posterior),
                gauss_markov=optimal_posterior,
                ref_gauss_markov=reference,
            )
            obj_val = vanilla_objective(
                log_prior,
                log_transition,
                log_observation,
                optimal_posterior,
            )
            jax.debug.print(
                "iter {iter:>4d} | kl {kl:>8.3f} | val {val:>12.3f}",
                iter=iteration_idx,
                kl=kl_div,
                val=obj_val,
            )
            return optimal_posterior, optimal_posterior

        optimal_posterior, _ = jax.lax.scan(single_iteration, init_posterior, xs=jnp.arange(max_iterations))
        return optimal_posterior

    return (
        single_pass_smoother,
        dual_objective,
        vanilla_objective,
        iterated_smoother,
        undamped_iterated_smoother,
    )
