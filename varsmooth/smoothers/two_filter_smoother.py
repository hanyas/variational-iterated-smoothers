from typing import Callable
from functools import partial

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import (
    Gaussian,
    AffineGaussian,
    GaussMarkov,
    LogPrior,
    LogTransition,
    LogObservation,
    Potential,
    LogMarginalNorm
)
from varsmooth.smoothers.utils import (
    kl_between_marginals,
    statistical_expansion,
    merge_messages,
    log_to_std_form
)
from varsmooth.utils import (
    none_or_concat,
    none_or_shift,
    none_or_idx,
    symmetrize,
    eig
)

from varsmooth.smoothers.forward_markov import backward_log_message
from varsmooth.smoothers.forward_markov import forward_std_message
from varsmooth.smoothers.reverse_markov import forward_log_message


def log_two_filter_smoother(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    ref_forward: GaussMarkov,
    ref_reverse: GaussMarkov,
    temperature: float
) -> Gaussian:

    marginals = forward_std_message(ref_forward)

    log_prior, log_transition, log_observation = \
        statistical_expansion(
            observations,
            log_prior_fn,
            log_transition_fn,
            log_observation_fn,
            ref_forward.kernels,
            marginals,
        )

    damping = temperature / (1.0 + temperature)
    _, _, _, bwd_msg = backward_log_message(
        log_prior,
        log_transition,
        log_observation,
        ref_forward,
        damping,
    )

    rvs_post, _, fwd_msg, _ = forward_log_message(
        log_prior,
        log_transition,
        log_observation,
        ref_reverse,
        damping,
    )

    log_marginals = jax.vmap(merge_messages)(
        none_or_shift(fwd_msg, -1), bwd_msg
    )
    marginals = jax.vmap(log_to_std_form)(log_marginals)
    return none_or_concat(marginals, rvs_post.marginal, position=-1)
