from typing import Tuple

from jax import Array

from varsmooth.objects import GaussMarkov
from varsmooth.objects import LogMarginalNorm
from varsmooth.objects import LogMessage
from varsmooth.objects import LogObservation
from varsmooth.objects import LogPrior
from varsmooth.objects import LogTransition
from varsmooth.objects import ValueFn
from varsmooth.smoothers.core import make_smoother_suite
from varsmooth.smoothers.utils import _log_message_pass
from varsmooth.smoothers.utils import kl_between_reverse_gauss_markovs
from varsmooth.smoothers.utils import std_backward_message


def log_forward_message(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    reverse_reference: GaussMarkov,
    damping: float,
) -> Tuple[GaussMarkov, LogMarginalNorm, ValueFn, LogMessage, Array]:
    """Forward message pass of the reverse Gauss-Markov smoother.

    Eliminates the past state of every pairwise log-transition, accumulating a
    forward value function from the root to the leaf. Thin wrapper over the
    shared _log_message_pass with reverse=False.

    Args:
        log_prior: LogPrior
            Quadratic log-prior over the root state x_0.
        log_transition: LogTransition
            Batched pairwise quadratic log-transitions of leading shape (T,).
        log_observation: LogObservation
            Batched quadratic log-observations of leading shape (T,).
        reverse_reference: GaussMarkov
            The reverse Gauss-Markov posterior to expand around.
        damping: float
            Trust-region damping in [0, 1); damping = t / (1 + t).

    Returns:
        posterior: GaussMarkov
            The updated reverse Gauss-Markov posterior (leaf marginal + backward
            kernels).
        log_marg_norm: LogMarginalNorm
            The quadratic marginal log-normalizer at the leaf.
        value_fns: ValueFn
            The per-marginal forward value functions of leading shape (T + 1,).
        log_fwd_msgs: LogMessage
            The per-step forward messages of leading shape (T,).
        feasible_flags: Array
            Boolean array of shape (T,) marking feasible steps.
    """
    return _log_message_pass(
        log_prior,
        log_transition,
        log_observation,
        reverse_reference,
        damping,
        reverse=False,
    )


(
    reverse_markov_smoother,
    dual_objective,
    log_evidence,
    iterated_reverse_markov_smoother,
) = make_smoother_suite(
    log_message_fn=log_forward_message,
    std_marginal_fn=std_backward_message,
    kl_fn=kl_between_reverse_gauss_markovs,
)
