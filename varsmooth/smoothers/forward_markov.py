from jax import Array

from varsmooth.objects import GaussMarkov
from varsmooth.objects import LogLikelihood
from varsmooth.objects import LogMessage
from varsmooth.objects import LogNormalizer
from varsmooth.objects import LogPrior
from varsmooth.objects import LogTransition
from varsmooth.objects import ValueFn
from varsmooth.smoothers.core import make_smoother_suite
from varsmooth.smoothers.utils import _log_message_pass
from varsmooth.smoothers.utils import kl_between_forward_gauss_markovs
from varsmooth.smoothers.utils import std_forward_message


def log_backward_message(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_likelihood: LogLikelihood,
    forward_reference: GaussMarkov,
    damping: float,
) -> tuple[GaussMarkov, LogNormalizer, ValueFn, LogMessage, Array]:
    """Backward message pass of the forward Gauss-Markov smoother.

    Eliminates the future state of every pairwise log-transition, accumulating
    a backward value function from the leaf to the root. Thin wrapper over the
    shared _log_message_pass with reverse=True.

    Args:
        log_prior: LogPrior
            Quadratic log-prior over the root state x_0.
        log_transition: LogTransition
            Batched pairwise quadratic log-transitions of leading shape (T,).
        log_likelihood: LogLikelihood
            Batched quadratic log-likelihoods of leading shape (T,).
        forward_reference: GaussMarkov
            The forward Gauss-Markov posterior to expand around.
        damping: float
            Trust-region damping in [0, 1); damping = t / (1 + t).

    Returns:
        posterior: GaussMarkov
            The updated forward Gauss-Markov posterior (root marginal + forward
            kernels).
        log_marg_norm: LogNormalizer
            The quadratic marginal log-normalizer at the root.
        value_fns: ValueFn
            The per-marginal backward value functions of leading shape (T + 1,).
        log_bwd_msgs: LogMessage
            The per-step backward messages of leading shape (T,).
        feasible_flags: Array
            Boolean array of shape (T,) marking feasible steps.
    """
    return _log_message_pass(
        log_prior,
        log_transition,
        log_likelihood,
        forward_reference,
        damping,
        reverse=True,
    )


(
    forward_markov_smoother,
    forward_dual_objective,
    forward_log_evidence,
    iterated_forward_markov_smoother,
) = make_smoother_suite(
    log_message_fn=log_backward_message,
    std_marginal_fn=std_forward_message,
    kl_fn=kl_between_forward_gauss_markovs,
)
