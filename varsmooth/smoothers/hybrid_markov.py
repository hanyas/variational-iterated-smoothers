from functools import partial
from typing import Callable

import jax
from jax import Array
from jax import numpy as jnp

from varsmooth.objects import GaussMarkov
from varsmooth.objects import Gaussian
from varsmooth.objects import LogMessage
from varsmooth.objects import ValueFn
from varsmooth.smoothers.forward_markov import log_backward_message
from varsmooth.smoothers.forward_markov import std_forward_message
from varsmooth.smoothers.reverse_markov import log_forward_message
from varsmooth.smoothers.reverse_markov import reverse_dual_objective
from varsmooth.smoothers.reverse_markov import std_backward_message
from varsmooth.smoothers.utils import _run_iterations
from varsmooth.smoothers.utils import kl_between_forward_gauss_markovs
from varsmooth.smoothers.utils import kl_between_reverse_gauss_markovs
from varsmooth.smoothers.utils import line_search
from varsmooth.smoothers.utils import log_to_std_form
from varsmooth.smoothers.utils import merge_messages
from varsmooth.smoothers.utils import statistical_expansion
from varsmooth.smoothers.utils import std_to_log_form
from varsmooth.utils import none_or_concat
from varsmooth.utils import none_or_shift


def hybrid_markov_smoother(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_likelihood_fn: Callable,
    forward_reference: GaussMarkov,
    reverse_reference: GaussMarkov,
    temperature: float,
) -> Gaussian:
    """Run one hybrid-Markov pass, merging a forward and a reverse message pass.

    Linearizes the model around the forward reference marginals, runs the
    backward pass of the forward smoother and the forward pass of the reverse
    smoother at the damping induced by temperature, then merges their messages
    and boundaries into updated marginals.

    Args:
        observations: Array
            Batched observations of leading shape (T,).
        log_prior_fn: Callable
            Maps the root marginal to the quadratic log-prior over x_0.
        log_transition_fn: Callable
            Maps reference kernels and marginals to the pairwise quadratic
            log-transitions.
        log_likelihood_fn: Callable
            Maps observations and marginals to the quadratic log-likelihoods.
        forward_reference: GaussMarkov
            The forward Gauss-Markov posterior to expand around.
        reverse_reference: GaussMarkov
            The reverse Gauss-Markov posterior to expand around.
        temperature: float
            Trust-region temperature t; damping = t / (1 + t).

    Returns:
        Gaussian
            The updated per-marginal Gaussians of leading shape (T + 1,).
    """
    marginals = std_forward_message(forward_reference)

    log_prior, log_transition, log_likelihood = statistical_expansion(
        observations=observations,
        log_prior_fn=log_prior_fn,
        log_transition_fn=log_transition_fn,
        log_likelihood_fn=log_likelihood_fn,
        kernels=forward_reference.kernels,  # or reverse_reference.kernels
        marginals=marginals,
    )

    damping = temperature / (1.0 + temperature)
    forward_posterior, _, _, backward_message, _ = log_backward_message(
        log_prior=log_prior,
        log_transition=log_transition,
        log_likelihood=log_likelihood,
        forward_reference=forward_reference,
        damping=damping,
    )

    reverse_posterior, _, forward_message, _, _ = log_forward_message(
        log_prior=log_prior,
        log_transition=log_transition,
        log_likelihood=log_likelihood,
        reverse_reference=reverse_reference,
        damping=damping,
    )

    marginals = update_marginals(
        marginals=marginals,
        forward_message=forward_message,
        backward_message=backward_message,
        first_boundary=forward_posterior.marginal,
        last_boundary=reverse_posterior.marginal,
        damping=damping,
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
    """Merge forward and backward messages into damped, updated marginals.

    Combines the forward and backward messages at the interior marginals, damps
    them against the current log-marginals, overrides the first and last
    marginals with the supplied boundaries, and converts the result back to
    standard (moment) form.

    Args:
        marginals: Gaussian
            Current per-marginal Gaussians of leading shape (T + 1,).
        forward_message: ValueFn
            Per-marginal forward value functions produced by log_forward_message.
        backward_message: LogMessage
            Per-step backward messages produced by log_backward_message.
        first_boundary: Gaussian
            Updated first marginal (root x_0) from the forward posterior.
        last_boundary: Gaussian
            Updated last marginal (leaf x_T) from the reverse posterior.
        damping: float
            Trust-region damping in [0, 1); damping = t / (1 + t).

    Returns:
        Gaussian
            The updated per-marginal Gaussians of leading shape (T + 1,).
    """
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
        "log_likelihood_fn",
        "max_iterations",
        "return_history",
        "verbose",
    ],
)
def iterated_hybrid_markov_smoother(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_likelihood_fn: Callable,
    init_forward_posterior: GaussMarkov,
    init_reverse_posterior: GaussMarkov,
    kl_constraint: float,
    init_temperature: float = 1e12,
    min_temperature: float = 1e-12,
    max_iterations: int = 1000,
    return_history: bool = False,
    verbose: bool = True,
):
    """Run the iterated hybrid-Markov smoother to convergence.

    Alternates a statistical expansion around the merged marginals with a
    trust-region step that merges a forward and a reverse message pass, stopping
    early once the line-search temperature drops to min_temperature or
    max_iterations is reached.

    Args:
        observations: Array
            Batched observations of leading shape (T,).
        log_prior_fn: Callable
            Maps the root marginal to the quadratic log-prior over x_0.
        log_transition_fn: Callable
            Maps reference kernels and marginals to the pairwise quadratic
            log-transitions.
        log_likelihood_fn: Callable
            Maps observations and marginals to the quadratic log-likelihoods.
        init_forward_posterior: GaussMarkov
            Initial forward Gauss-Markov posterior.
        init_reverse_posterior: GaussMarkov
            Initial reverse Gauss-Markov posterior.
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
            If True (and not return_history), print per-iteration diagnostics.

    Returns:
        marginals: Gaussian
            The converged per-marginal Gaussians of leading shape (T + 1,).
        history: dict
            Stacked per-iteration diagnostics, returned only when
            return_history is True.
    """

    def single_iteration(carry, iteration_idx):
        """Perform a single trust-region iteration of the hybrid-Markov update."""
        reference_marginals, forward_reference, reverse_reference = carry

        # Step 1: Compute statistical expansion (around the merged marginals)
        log_prior, log_transition, log_likelihood = statistical_expansion(
            observations=observations,
            log_prior_fn=log_prior_fn,
            log_transition_fn=log_transition_fn,
            log_likelihood_fn=log_likelihood_fn,
            kernels=forward_reference.kernels,  # or reverse_reference.kernels
            marginals=reference_marginals,
        )

        # Step 2: Define dual objective function for line search
        def dual_objective_fn(temperature):
            """Dual objective function for temperature optimization."""
            damping = temperature / (1.0 + temperature)
            return reverse_dual_objective(
                log_prior=log_prior,
                log_transition=log_transition,
                log_likelihood=log_likelihood,
                reference_posterior=reverse_reference,
                kl_constraint=kl_constraint,
                damping=damping,
            )

        # Step 3: Define the constraint-slack function for the line search.
        def constraint_slack_fn(temperature):
            """Constraint slack on the merged-marginal move: kl_constraint - KL."""
            damping = temperature / (1.0 + temperature)
            forward_posterior, _, _, _, fwd_feasible = log_backward_message(
                log_prior=log_prior,
                log_transition=log_transition,
                log_likelihood=log_likelihood,
                forward_reference=forward_reference,
                damping=damping,
            )
            reverse_posterior, _, _, _, rev_feasible = log_forward_message(
                log_prior=log_prior,
                log_transition=log_transition,
                log_likelihood=log_likelihood,
                reverse_reference=reverse_reference,
                damping=damping,
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
            log_prior=log_prior,
            log_transition=log_transition,
            log_likelihood=log_likelihood,
            forward_reference=forward_reference,
            damping=0.0,
        )
        full_rev_post, _, full_fwd_msg, _, full_rev_feasible = log_forward_message(
            log_prior=log_prior,
            log_transition=log_transition,
            log_likelihood=log_likelihood,
            reverse_reference=reverse_reference,
            damping=0.0,
        )
        full_marginals = update_marginals(
            marginals=reference_marginals,
            forward_message=full_fwd_msg,
            backward_message=full_bwd_msg,
            first_boundary=full_fwd_post.marginal,
            last_boundary=full_rev_post.marginal,
            damping=0.0,
        )
        full_feasible = jnp.logical_and(jnp.all(full_fwd_feasible), jnp.all(full_rev_feasible))

        def _full_step_kl():
            forward_marginals = std_forward_message(full_fwd_post)
            forward_kl = kl_between_forward_gauss_markovs(
                marginals=forward_marginals,
                gauss_markov=full_fwd_post,
                ref_gauss_markov=forward_reference,
            )
            reverse_marginals = std_backward_message(full_rev_post)
            reverse_kl = kl_between_reverse_gauss_markovs(
                marginals=reverse_marginals,
                gauss_markov=full_rev_post,
                ref_gauss_markov=reverse_reference,
            )
            return 0.5 * (forward_kl + reverse_kl)

        full_kl = jax.lax.cond(full_feasible, _full_step_kl, lambda: jnp.inf)

        take_full_step = jnp.logical_and(full_feasible, full_kl <= kl_constraint)

        # Step 4: Perform line search to find optimal temperature
        temperature, dual_value, _, line_search_feasible = line_search(
            init_temperature, dual_objective_fn, constraint_slack_fn, rtol=0.1 * kl_constraint
        )

        # Step 5: Apply the optimal temperature
        damping = temperature / (1.0 + temperature)

        forward_posterior, _, _, backward_message, _ = log_backward_message(
            log_prior=log_prior,
            log_transition=log_transition,
            log_likelihood=log_likelihood,
            forward_reference=forward_reference,
            damping=damping,
        )
        reverse_posterior, _, forward_message, _, _ = log_forward_message(
            log_prior=log_prior,
            log_transition=log_transition,
            log_likelihood=log_likelihood,
            reverse_reference=reverse_reference,
            damping=damping,
        )
        candidate_marginals = update_marginals(
            marginals=reference_marginals,
            forward_message=forward_message,
            backward_message=backward_message,
            first_boundary=forward_posterior.marginal,
            last_boundary=reverse_posterior.marginal,
            damping=damping,
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
            marginals=std_forward_message(forward_post),
            gauss_markov=forward_post,
            ref_gauss_markov=forward_reference,
        )
        rvs_kl_div = kl_between_reverse_gauss_markovs(
            marginals=std_backward_message(reverse_post),
            gauss_markov=reverse_post,
            ref_gauss_markov=reverse_reference,
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

        if verbose and not return_history:
            jax.lax.cond(feasible, _log_feasible, _log_infeasible, operand=None)

        diagnostics = {
            "damping": damping,
            "fwd_kl": fwd_kl_div,
            "rvs_kl": rvs_kl_div,
            "realized_kl": 0.5 * (fwd_kl_div + rvs_kl_div),
            "dual": dual_value,
            "feasible": feasible,
            "marginals": marginals,
            "forward_marginals": std_forward_message(forward_post),
            "forward_kernels": forward_post.kernels,
            "reverse_marginals": std_backward_message(reverse_post),
            "reverse_kernels": reverse_post.kernels,
        }
        return (marginals, forward_post, reverse_post), temperature, diagnostics

    # Initialize state
    init_marginals = std_forward_message(init_forward_posterior)
    init_state = (init_marginals, init_forward_posterior, init_reverse_posterior)

    final_state, history = _run_iterations(
        single_iteration,
        init_state,
        init_temperature,
        min_temperature,
        max_iterations,
        return_history,
    )
    optimal_marginals, _, _ = final_state
    if return_history:
        return optimal_marginals, history
    return optimal_marginals
