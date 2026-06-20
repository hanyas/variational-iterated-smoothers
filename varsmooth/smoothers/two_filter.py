from typing import Callable
from functools import partial

import jax
from jax import Array
from jax import numpy as jnp

from varsmooth.objects import (
    Gaussian,
    GaussMarkov,
    ValueFn,
    LogMessage
)
from varsmooth.smoothers.utils import (
    statistical_expansion,
    log_to_std_form,
    std_to_log_form,
    merge_messages,
    kl_between_reverse_gauss_markovs,
    kl_between_forward_gauss_markovs,
    line_search,
)
from varsmooth.utils import (
    none_or_concat,
    none_or_shift,
    bounded_while_loop,
)

from varsmooth.smoothers.forward_markov import log_backward_message
from varsmooth.smoothers.forward_markov import std_forward_message
from varsmooth.smoothers.reverse_markov import log_forward_message
from varsmooth.smoothers.reverse_markov import std_backward_message

from varsmooth.smoothers.reverse_markov import dual_objective


def two_filter_smoother(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    forward_reference: GaussMarkov,
    reverse_reference: GaussMarkov,
    temperature: float
) -> Gaussian:

    marginals = std_forward_message(forward_reference)

    log_prior, log_transition, log_observation = \
        statistical_expansion(
            observations,
            log_prior_fn,
            log_transition_fn,
            log_observation_fn,
            forward_reference.kernels,
            marginals,
        )

    damping = temperature / (1.0 + temperature)
    forward_posterior, _, _, backward_message, _ = log_backward_message(
        log_prior,
        log_transition,
        log_observation,
        forward_reference,
        damping,
    )

    reverse_posterior, _, forward_message, _, _ = log_forward_message(
        log_prior,
        log_transition,
        log_observation,
        reverse_reference,
        damping,
    )

    fwd_kl_div = kl_between_forward_gauss_markovs(
        std_forward_message(forward_posterior),
        forward_posterior,
        forward_reference
    )

    rvs_kl_div = kl_between_reverse_gauss_markovs(
        std_backward_message(reverse_posterior),
        reverse_posterior,
        reverse_reference
    )

    marginals = update_marginals(
        marginals,
        forward_message,
        backward_message,
        forward_posterior.marginal,
        reverse_posterior.marginal,
        damping
    )
    return marginals


def update_marginals(
    marginals: Gaussian,
    forward_message: ValueFn,
    backward_message: LogMessage,
    first_boundary: Gaussian,
    last_boundary: Gaussian,
    damping: float
):
    log_marginals = jax.vmap(std_to_log_form)(marginals)
    log_messages = jax.vmap(merge_messages)(
        none_or_shift(none_or_shift(forward_message, -1), 1),
        none_or_shift(backward_message, 1),

    )
    log_first_boundary = std_to_log_form(first_boundary)
    log_last_boundary = std_to_log_form(last_boundary)

    # update all but last marginal
    value_fns = ValueFn(
        R=(1.0 - damping) * log_messages.R + damping * log_marginals.R[1:-1],
        r=(1.0 - damping) * log_messages.r + damping * log_marginals.r[1:-1],
        rho=(1.0 - damping) * log_messages.rho + damping * log_marginals.rho[1:-1]
    )

    # append first marginal
    value_fns = none_or_concat(
        value_fns,
        ValueFn(
            R=log_first_boundary.R,
            r=log_first_boundary.r,
            rho=log_first_boundary.rho,
        ),
    )

    # append last marginal
    value_fns = none_or_concat(
        value_fns,
        ValueFn(
            R=log_last_boundary.R,
            r=log_last_boundary.r,
            rho=log_last_boundary.rho,
        ),
        position=-1
    )

    return jax.vmap(log_to_std_form)(value_fns)


@partial(jax.jit, static_argnames=[
    'log_prior_fn',
    'log_transition_fn',
    'log_observation_fn',
    'kl_constraint',
    'init_temperature',
    'min_temperature',
    'max_iterations'
])
def iterated_two_filter_smoother(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    init_forward_posterior: GaussMarkov,
    init_reverse_posterior: GaussMarkov,
    kl_constraint: float,
    init_temperature: float = 1e12,
    min_temperature: float = 1e-12,
    max_iterations: int = 1000,
):
    """
    Iterated two-filter smoother with early stopping based on temperature.

    This function performs variational inference using both forward and reverse
    message passing until convergence. The iterations stop when either:
    1. Maximum iterations are reached, or
    2. Temperature drops below min_temperature (indicating convergence)

    Args:
        observations: Array of observations
        log_prior_fn: Function to compute log prior
        log_transition_fn: Function to compute log transition
        log_observation_fn: Function to compute log observation likelihood
        init_forward_posterior: Initial forward posterior estimate
        init_reverse_posterior: Initial reverse posterior estimate
        kl_constraint: KL divergence constraint for the optimization
        init_temperature: Initial temperature for line search
        min_temperature: Minimum temperature threshold for early stopping
        max_iterations: Maximum number of iterations

    Returns:
        Optimal marginals after convergence
    """

    def single_iteration(carry, iteration_idx):
        """
        Perform a single iteration of the two-filter update.

        Args:
            carry: Tuple of (reference_marginals, forward_reference, reverse_reference)
            iteration_idx: Current iteration index (for logging)

        Returns:
            updated_carry: Updated state tuple
            final_temperature: Temperature from line search
        """
        reference_marginals, forward_reference, reverse_reference = carry

        # Step 1: Compute statistical expansion
        log_prior, log_transition, log_observation = statistical_expansion(
            observations,
            log_prior_fn,
            log_transition_fn,
            log_observation_fn,
            forward_reference.kernels,
            reference_marginals,
        )

        # Step 2: Define dual objective function for line search
        def dual_objective_fn(temperature):
            """Dual objective function for temperature optimization."""
            damping = temperature / (1.0 + temperature)
            return dual_objective(
                log_prior,
                log_transition,
                log_observation,
                reverse_reference,
                kl_constraint,
                damping,
            )

        # Step 3: Define gradient function for line search
        def dual_gradient_fn(temperature):
            """Gradient of dual objective with respect to temperature."""
            damping = temperature / (1.0 + temperature)
            posterior, _, _, _, feasible_pass = log_forward_message(
                log_prior,
                log_transition,
                log_observation,
                reverse_reference,
                damping,
            )

            def compute_gradient():
                """Compute gradient when forward pass is feasible."""
                kl_div = kl_between_reverse_gauss_markovs(
                    marginals=std_backward_message(posterior),
                    gauss_markov=posterior,
                    ref_gauss_markov=reverse_reference
                )
                return kl_constraint - kl_div

            def inf_gradient():
                """Return infinity when forward pass is not feasible."""
                return jnp.inf

            return jax.lax.cond(
                pred=jnp.all(feasible_pass),
                true_fun=lambda _: compute_gradient(),
                false_fun=lambda _: inf_gradient(),
                operand=None
            )

        # Step 4: Perform line search to find optimal temperature
        temperature, dual_value, _, line_search_feasible = line_search(
            init_temperature,
            dual_objective_fn,
            dual_gradient_fn,
            rtol=0.1 * kl_constraint
        )

        # Step 5: Apply the optimal temperature to get final result
        def apply_optimal_solution():
            """Apply the optimal temperature to compute final marginals."""
            damping = temperature / (1.0 + temperature)

            # Forward pass
            forward_posterior, _, _, backward_message, _ = log_backward_message(
                log_prior,
                log_transition,
                log_observation,
                forward_reference,
                damping,
            )

            # Reverse pass
            reverse_posterior, _, forward_message, _, _ = log_forward_message(
                log_prior,
                log_transition,
                log_observation,
                reverse_reference,
                damping,
            )

            # Compute KL divergences for logging
            fwd_kl_div = kl_between_forward_gauss_markovs(
                std_forward_message(forward_posterior),
                forward_posterior,
                forward_reference
            )

            rvs_kl_div = kl_between_reverse_gauss_markovs(
                std_backward_message(reverse_posterior),
                reverse_posterior,
                reverse_reference
            )

            # Update marginals
            updated_marginals = update_marginals(
                reference_marginals,
                forward_message,
                backward_message,
                forward_posterior.marginal,
                reverse_posterior.marginal,
                damping
            )

            # Log progress
            jax.debug.print(
                "iter {iter:>4d} | damping {damp:>8.2e} | fwd_kl {fwd_kl:>8.3f} "
                "| rvs_kl {rvs_kl:>8.3f} | dual {dual:>12.3f}",
                iter=iteration_idx,
                damp=damping,
                fwd_kl=fwd_kl_div,
                rvs_kl=rvs_kl_div,
                dual=dual_value
            )

            return updated_marginals, forward_posterior, reverse_posterior

        def use_reference():
            """Use reference when line search fails."""
            jax.debug.print(
                "iter {iter:>4d} | not feasible, process might have converged",
                iter=iteration_idx
            )
            return reference_marginals, forward_reference, reverse_reference

        # Choose between optimal solution and reference based on feasibility
        marginals, forward_post, reverse_post = jax.lax.cond(
            pred=line_search_feasible,
            true_fun=lambda _: apply_optimal_solution(),
            false_fun=lambda _: use_reference(),
            operand=None
        )

        return (marginals, forward_post, reverse_post), temperature

    def iteration_body(carry):
        """Body function for the while loop."""
        current_state, iteration_count, _ = carry
        next_state, next_temperature = single_iteration(current_state, iteration_count)
        return next_state, iteration_count + 1, next_temperature

    def iteration_condition(carry):
        """Condition function for the while loop."""
        _, iteration_count, next_temperature = carry
        # Continue if: not reached max iterations AND temperature is above minimum
        return jnp.logical_and(iteration_count < max_iterations, next_temperature > min_temperature)

    # Initialize state
    init_marginals = std_forward_message(init_forward_posterior)
    init_state = (init_marginals, init_forward_posterior, init_reverse_posterior)

    # Run the iterative optimization
    final_state, _, _ = bounded_while_loop(
        cond_fun=iteration_condition,
        body_fun=iteration_body,
        init_val=(init_state, 0, init_temperature),
        maxiter=max_iterations,
    )

    # Extract final marginals
    optimal_marginals, _, _ = final_state
    return optimal_marginals
