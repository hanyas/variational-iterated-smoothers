from functools import partial
from typing import Callable

import jax
from jax import Array
from jax import numpy as jnp

from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import LogMessage
from varsmooth.objects import ValueFn
from varsmooth.smoothers.forward_markov import log_backward_message
from varsmooth.smoothers.forward_markov import std_forward_message
from varsmooth.smoothers.reverse_markov import dual_objective
from varsmooth.smoothers.reverse_markov import log_forward_message
from varsmooth.smoothers.reverse_markov import std_backward_message
from varsmooth.smoothers.utils import kl_between_forward_gauss_markovs
from varsmooth.smoothers.utils import kl_between_reverse_gauss_markovs
from varsmooth.smoothers.utils import line_search
from varsmooth.smoothers.utils import log_to_std_form
from varsmooth.smoothers.utils import merge_messages
from varsmooth.smoothers.utils import statistical_expansion
from varsmooth.smoothers.utils import std_to_log_form
from varsmooth.utils import bounded_while_loop
from varsmooth.utils import none_or_concat
from varsmooth.utils import none_or_shift


def two_filter_smoother(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    forward_reference: GaussMarkov,
    reverse_reference: GaussMarkov,
    temperature: float,
) -> Gaussian:

    marginals = std_forward_message(forward_reference)

    log_prior, log_transition, log_observation = statistical_expansion(
        observations,
        log_prior_fn,
        log_transition_fn,
        log_observation_fn,
        forward_reference.kernels,  # or reverse_reference.kernels
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

    marginals = update_marginals(
        marginals,
        forward_message,
        backward_message,
        forward_posterior.marginal,
        reverse_posterior.marginal,
        damping,
    )
    return marginals


def update_marginals(
    marginals: Gaussian,
    forward_message: ValueFn,
    backward_message: LogMessage,
    first_boundary: Gaussian,
    last_boundary: Gaussian,
    damping: float,
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
        rho=(1.0 - damping) * log_messages.rho + damping * log_marginals.rho[1:-1],
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
        position=-1,
    )

    return jax.vmap(log_to_std_form)(value_fns)


@partial(
    jax.jit,
    static_argnames=[
        "log_prior_fn",
        "log_transition_fn",
        "log_observation_fn",
        "kl_constraint",
        "init_temperature",
        "min_temperature",
        "max_iterations",
        "return_history",
    ],
)
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
    return_history: bool = False,
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

        # Step 1: Compute statistical expansion (around the merged marginals)
        log_prior, log_transition, log_observation = statistical_expansion(
            observations,
            log_prior_fn,
            log_transition_fn,
            log_observation_fn,
            forward_reference.kernels,  # or reverse_reference.kernels
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

        # Step 3: Define gradient function for line search.
        def dual_gradient_fn(temperature):
            """Constraint slack on the merged-marginal move: kl_constraint - KL."""
            damping = temperature / (1.0 + temperature)
            forward_posterior, _, _, _, fwd_feasible = log_backward_message(
                log_prior,
                log_transition,
                log_observation,
                forward_reference,
                damping,
            )
            reverse_posterior, _, _, _, rev_feasible = log_forward_message(
                log_prior,
                log_transition,
                log_observation,
                reverse_reference,
                damping,
            )

            fwd_kl_div = kl_between_forward_gauss_markovs(
                std_forward_message(forward_posterior), forward_posterior, forward_reference
            )
            rvs_kl_div = kl_between_reverse_gauss_markovs(
                std_backward_message(reverse_posterior), reverse_posterior, reverse_reference
            )

            def compute_gradient():
                return kl_constraint - 0.5 * (fwd_kl_div + rvs_kl_div)

            def inf_gradient():
                """Return infinity when either pass is not feasible."""
                return jnp.inf

            return jax.lax.cond(
                pred=jnp.logical_and(jnp.all(fwd_feasible), jnp.all(rev_feasible)),
                true_fun=lambda _: compute_gradient(),
                false_fun=lambda _: inf_gradient(),
                operand=None,
            )

        # Undamped (full) step: the proximal optimum when the trust region is
        # inactive. If the merged marginals already move within the constraint,
        # take the full step directly.
        full_fwd_post, _, _, full_bwd_msg, full_fwd_feasible = log_backward_message(
            log_prior, log_transition, log_observation, forward_reference, 0.0
        )
        full_rev_post, _, full_fwd_msg, _, full_rev_feasible = log_forward_message(
            log_prior, log_transition, log_observation, reverse_reference, 0.0
        )
        full_marginals = update_marginals(
            reference_marginals,
            full_fwd_msg,
            full_bwd_msg,
            full_fwd_post.marginal,
            full_rev_post.marginal,
            0.0,
        )
        full_feasible = jnp.logical_and(jnp.all(full_fwd_feasible), jnp.all(full_rev_feasible))
        full_kl = jax.lax.cond(
            full_feasible,
            lambda: 0.5
            * (
                kl_between_forward_gauss_markovs(std_forward_message(full_fwd_post), full_fwd_post, forward_reference)
                + kl_between_reverse_gauss_markovs(
                    std_backward_message(full_rev_post), full_rev_post, reverse_reference
                )
            ),
            lambda: jnp.inf,
        )
        take_full_step = jnp.logical_and(full_feasible, full_kl <= kl_constraint)

        # Step 4: Perform line search to find optimal temperature
        temperature, dual_value, _, line_search_feasible = line_search(
            init_temperature, dual_objective_fn, dual_gradient_fn, rtol=0.1 * kl_constraint
        )

        # Step 5: Apply the optimal temperature
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
        candidate_marginals = update_marginals(
            reference_marginals,
            forward_message,
            backward_message,
            forward_posterior.marginal,
            reverse_posterior.marginal,
            damping,
        )

        ls_marginals, ls_fwd_post, ls_rev_post = jax.lax.cond(
            pred=line_search_feasible,
            true_fun=lambda _: (candidate_marginals, forward_posterior, reverse_posterior),
            false_fun=lambda _: (reference_marginals, forward_reference, reverse_reference),
            operand=None,
        )

        marginals, forward_post, reverse_post = jax.lax.cond(
            pred=take_full_step,
            true_fun=lambda _: (full_marginals, full_fwd_post, full_rev_post),
            false_fun=lambda _: (ls_marginals, ls_fwd_post, ls_rev_post),
            operand=None,
        )
        temperature = jnp.where(take_full_step, min_temperature, temperature)
        damping = jnp.where(take_full_step, 0.0, jnp.where(line_search_feasible, damping, 0.0))
        feasible = jnp.logical_or(take_full_step, line_search_feasible)

        fwd_kl_div = kl_between_forward_gauss_markovs(
            std_forward_message(forward_post), forward_post, forward_reference
        )
        rvs_kl_div = kl_between_reverse_gauss_markovs(
            std_backward_message(reverse_post), reverse_post, reverse_reference
        )

        def _log_feasible(_):
            jax.debug.print(
                "iter {iter:>4d} | damping {damp:>8.2e} | fwd_kl {fwd_kl:>8.3f} "
                "| rvs_kl {rvs_kl:>8.3f} | dual {dual:>12.3f}",
                iter=iteration_idx,
                damp=damping,
                fwd_kl=fwd_kl_div,
                rvs_kl=rvs_kl_div,
                dual=dual_value,
            )
            return 0

        def _log_infeasible(_):
            jax.debug.print(
                "iter {iter:>4d} | not feasible, process might have converged",
                iter=iteration_idx,
            )
            return 0

        if not return_history:
            jax.lax.cond(feasible, _log_feasible, _log_infeasible, operand=None)

        diagnostics = {
            "damping": damping,
            "fwd_kl": fwd_kl_div,
            "rvs_kl": rvs_kl_div,
            "realized_kl": 0.5 * (fwd_kl_div + rvs_kl_div),
            "dual": dual_value,
            "feasible": feasible,
            "marginals": marginals,
        }
        return (marginals, forward_post, reverse_post), temperature, diagnostics

    # Initialize state
    init_marginals = std_forward_message(init_forward_posterior)
    init_state = (init_marginals, init_forward_posterior, init_reverse_posterior)

    if return_history:

        def scan_step(state, iteration_idx):
            next_state, _temperature, diagnostics = single_iteration(state, iteration_idx)
            return next_state, diagnostics

        final_state, history = jax.lax.scan(scan_step, init_state, xs=jnp.arange(max_iterations))
        optimal_marginals, _, _ = final_state
        return optimal_marginals, history

    def iteration_body(carry):
        """Body function for the while loop."""
        current_state, iteration_count, _ = carry
        next_state, next_temperature, _ = single_iteration(current_state, iteration_count)
        return next_state, iteration_count + 1, next_temperature

    def iteration_condition(carry):
        """Condition function for the while loop."""
        _, iteration_count, next_temperature = carry
        # Continue if: not reached max iterations AND temperature is above minimum
        return jnp.logical_and(iteration_count < max_iterations, next_temperature > min_temperature)

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
