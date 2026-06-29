from functools import partial

import jax
from jax import numpy as jnp

from varsmooth.smoothers.utils import line_search
from varsmooth.smoothers.utils import statistical_expansion
from varsmooth.utils import bounded_while_loop


def free_energy(log_prior, log_transition, log_observation, marginals, kernels):
    """Variational free energy (ELBO)

    Args:
        log_prior: ``LogPrior`` over x_0.
        log_transition: ``LogTransition`` over the joint (x_{k+1}, x_k), stacked over k.
        log_observation: ``LogObservation`` over x_{k+1}, stacked over k.
        marginals: Gaussian marginals N(m_k, P_k), k = 0..T.
        kernels: forward ``AffineGaussian`` kernels q(x_{k+1}|x_k), k = 0..T-1.
    """
    m, P = marginals.mean, marginals.cov
    F, _, Sigma = kernels

    def _expected_quadratic(M, v, c, mk, Pk):
        # E_{N(mk, Pk)}[ -0.5 x^T M x + v^T x + c ]
        return -0.5 * (mk @ M @ mk + jnp.trace(M @ Pk)) + v @ mk + c

    def _entropy(cov):
        return 0.5 * (cov.shape[0] * (jnp.log(2.0 * jnp.pi) + 1.0) + jnp.linalg.slogdet(cov)[1])

    # prior:  E_{q(x_0)}[log p(x_0)]
    prior_term = _expected_quadratic(log_prior.L, log_prior.l, log_prior.nu, m[0], P[0])

    # observations:  sum_k E_{q(x_k)}[log p(y_k | x_k)]
    obs_terms = jax.vmap(lambda lo, mk, Pk: _expected_quadratic(lo.L, lo.l, lo.nu, mk, Pk))(
        log_observation, m[1:], P[1:]
    )

    # transitions:  sum_k E_{q(x_k, x_{k+1})}[log p(x_{k+1} | x_k)]  over the twin marginal
    def _transition_term(lt, mk, Pk, Fk, mk1, Pk1):
        cross = Fk @ Pk  # Cov(x_{k+1}, x_k)
        joint_precision = jnp.block([[lt.C11, -lt.C12], [-lt.C21, lt.C22]])
        mean = jnp.concatenate([mk1, mk])  # z = [x_{k+1}, x_k]
        cov = jnp.block([[Pk1, cross], [cross.T, Pk]])
        linear = jnp.concatenate([lt.c1, lt.c2])
        return -0.5 * (mean @ joint_precision @ mean + jnp.trace(joint_precision @ cov)) + linear @ mean + lt.kappa

    trans_terms = jax.vmap(_transition_term)(log_transition, m[:-1], P[:-1], F, m[1:], P[1:])

    # entropy of the chain:  H(x_0) + sum_k H(x_{k+1} | x_k)
    entropy = _entropy(P[0]) + jnp.sum(jax.vmap(_entropy)(Sigma))
    return prior_term + jnp.sum(obs_terms) + jnp.sum(trans_terms) + entropy


def make_smoother_suite(log_message_fn, std_marginal_fn, kl_fn):
    """Build a direction's smoother suite from its message-passing primitives.

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
        Tuple ``(single_pass_smoother, dual_objective, log_evidence,
        iterated_smoother)``.
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
    ):
        """Iterated KL-constrained smoother with temperature-based early stopping."""

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
                dual_gradient_fn,
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

            if not return_history:
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

        if return_history:

            def scan_step(reference, iteration_idx):
                next_posterior, _temperature, diagnostics = single_iteration(reference, iteration_idx)
                return next_posterior, diagnostics

            final_posterior, history = jax.lax.scan(scan_step, init_posterior, xs=jnp.arange(max_iterations))
            return final_posterior, history

        def iteration_body(carry):
            current_posterior, iteration_count, _ = carry
            next_posterior, next_temperature, _ = single_iteration(current_posterior, iteration_count)
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

    return (
        single_pass_smoother,
        dual_objective,
        log_evidence,
        iterated_smoother,
    )
