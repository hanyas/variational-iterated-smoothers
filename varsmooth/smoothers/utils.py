from typing import Callable, Tuple, NamedTuple
from functools import partial

import jax
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import (
    Gaussian,
    AffineGaussian,
    GaussMarkov,
    LogPrior,
    LogTransition,
    LogObservation,
    ValueFn,
    LogMessage
)
from varsmooth.utils import (
    none_or_idx,
    none_or_shift,
    none_or_concat,
    logdet,
    bounded_while_loop,
)


def kl_between_marginals(p, q):
    dim = p.mean.shape[0]
    return 0.5 * (
        jnp.trace(jsc.linalg.inv(q.cov) @ p.cov) - dim
        + (q.mean - p.mean).T @ jsc.linalg.solve(q.cov, q.mean - p.mean)
        + logdet(q.cov) - logdet(p.cov)
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def statistical_expansion(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    posterior_kernels: AffineGaussian,
    posterior_marginals: Gaussian
) -> Tuple[LogPrior, LogTransition, LogObservation]:

    init_marginal = none_or_idx(posterior_marginals, 0)
    prev_marginals = none_or_shift(posterior_marginals, -1)
    next_marginals = none_or_shift(posterior_marginals, 1)

    log_prior = log_prior_fn(init_marginal)
    log_transition = log_transition_fn(prev_marginals, posterior_kernels)
    log_observation = log_observation_fn(observations, next_marginals)
    return log_prior, log_transition, log_observation


def std_forward_message(posterior: GaussMarkov) -> Gaussian:
    """Marginals of a forward Gauss-Markov chain (root marginal + forward kernels)."""
    init_marginal, kernels = posterior

    def _forward_step(carry, kernel):
        m, P = carry
        F, d, Sigma = kernel
        qn = Gaussian(mean=F @ m + d, cov=F @ P @ F.T + Sigma)
        return qn, qn

    _, marginals = jax.lax.scan(_forward_step, init_marginal, kernels)
    return none_or_concat(marginals, init_marginal, position=1)


def std_backward_message(posterior: GaussMarkov) -> Gaussian:
    """Marginals of a reverse Gauss-Markov chain (leaf marginal + backward kernels)."""
    last_marginal, kernels = posterior

    def _backward_step(carry, kernel):
        m, P = carry
        F, d, Sigma = kernel
        qn = Gaussian(mean=F @ m + d, cov=F @ P @ F.T + Sigma)
        return qn, qn

    _, marginals = jax.lax.scan(_backward_step, last_marginal, kernels, reverse=True)
    return none_or_concat(marginals, last_marginal, position=-1)


def initialize_reverse_with_forward(
    forward_markov: GaussMarkov
) -> GaussMarkov:
    forward_marginals = std_forward_message(forward_markov)

    # reverse kernels q(x_k | x_{k+1}) from forward marginals + forward kernels
    kernels = jax.vmap(get_reverse_kernel)(
        none_or_shift(forward_marginals, -1),   # marginals 0 .. T-1
        forward_markov.kernels,                 # forward kernels k+1 | k
        none_or_shift(forward_marginals, 1),    # marginals 1 .. T
    )

    return GaussMarkov(
        marginal=Gaussian(
            mean=forward_marginals.mean[-1],
            cov=forward_marginals.cov[-1],
        ),
        kernels=kernels,
    )


def get_marginal(
    marginal: Gaussian,
    kernel: AffineGaussian
):
    m, P = marginal
    F, d, Sigma = kernel
    return Gaussian(
        mean=F @ m + d,
        cov=F @ P @ F.T + Sigma
    )


def get_pairwise_marginal(
    marginal: Gaussian,
    kernel: AffineGaussian
):
    m, P = marginal
    F, d, Sigma = kernel

    q = Gaussian(
        mean=jnp.hstack((F @ m + d, m)),
        cov=jnp.vstack((
            jnp.hstack((F @ P @ F.T + Sigma, F @ P)),
            jnp.hstack((P.T @ F.T, P))
        ))
    )
    return q


def get_conditional(
    marginal: Gaussian,
    pairwise: Gaussian
):
    dim = marginal.mean.shape[0]

    a = pairwise.mean[:dim]
    b = pairwise.mean[dim:]

    A = pairwise.cov[:dim, :dim]
    B = pairwise.cov[dim:, dim:]
    C = pairwise.cov[:dim, dim:]

    return AffineGaussian(
        F=jsc.linalg.solve(A, C).T,
        d=b - C.T @ jsc.linalg.solve(A, a),
        Sigma=B - C.T @ jsc.linalg.solve(A, C)
    )


def get_reverse_kernel(
    marginal: Gaussian,
    kernel: AffineGaussian,
    next_marginal: Gaussian
):
    pairwise = get_pairwise_marginal(marginal, kernel)
    return get_conditional(next_marginal, pairwise)


def merge_messages(
    fwd_message: ValueFn,
    bwd_message: LogMessage,
) -> ValueFn:
    return ValueFn(
        R=(fwd_message.R + bwd_message.S),
        r=(fwd_message.r + bwd_message.s),
        rho=(fwd_message.rho + bwd_message.xi),
    )


def log_to_std_form(
    potential: ValueFn
) -> Gaussian:
    return Gaussian(
        mean=jsc.linalg.inv(potential.R) @ potential.r,
        cov=jsc.linalg.inv(potential.R)
    )


def std_to_log_form(
    dist: Gaussian
) -> ValueFn:
    return ValueFn(
        R=jsc.linalg.inv(dist.cov),
        r=jsc.linalg.solve(dist.cov, dist.mean),
        rho=(
            - 0.5 * logdet(2 * jnp.pi * dist.cov)
            - 0.5 * dist.mean.T @ jsc.linalg.solve(dist.cov, dist.mean)
        )
    )


def _kl_between_gauss_markovs(
    marginals: Gaussian,
    gauss_markov: GaussMarkov,
    ref_gauss_markov: GaussMarkov,
    reverse: bool = False,
):
    dim = gauss_markov.marginal.mean.shape[0]

    def body(carry, args):
        kl_value = carry
        m, P, \
            F, d, Sigma, \
            ref_F, ref_d, ref_Sigma = args

        diff_F = (ref_F - F).T @ jsc.linalg.solve(ref_Sigma, ref_F - F)
        diff_d = (ref_d - d).T @ jsc.linalg.solve(ref_Sigma, ref_d - d)
        diff_cross = (ref_F - F).T @ jsc.linalg.solve(ref_Sigma, ref_d - d)

        kl_value += (
            0.5 * jnp.trace(diff_F @ P)
            + 0.5 * m.T @ diff_F @ m
            + m.T @ diff_cross
            + 0.5 * diff_d
            + 0.5 * jnp.trace(jsc.linalg.solve(ref_Sigma, Sigma))
            - 0.5 * dim
            + 0.5 * logdet(ref_Sigma) - 0.5 * logdet(Sigma)
        )
        return kl_value, kl_value

    init_kl_value = kl_between_marginals(
        gauss_markov.marginal, ref_gauss_markov.marginal
    )

    kl_value, _ = jax.lax.scan(
        f=body,
        init=init_kl_value,
        xs=(
            *none_or_shift(marginals, 1),
            *gauss_markov.kernels,
            *ref_gauss_markov.kernels
        ),
        reverse=reverse,
    )
    return kl_value


def kl_between_reverse_gauss_markovs(
    marginals, gauss_markov, ref_gauss_markov
):
    return _kl_between_gauss_markovs(marginals, gauss_markov, ref_gauss_markov, True)


def kl_between_forward_gauss_markovs(
    marginals, gauss_markov, ref_gauss_markov
):
    return _kl_between_gauss_markovs(marginals, gauss_markov, ref_gauss_markov, False)


class ParamStruct(NamedTuple):
    val: float
    min: float
    max: float


class LineSearchState(NamedTuple):
    param: ParamStruct
    fn_val: float
    gd_val: float
    feasible: bool


def line_search(
    init_param: float,
    fun: Callable,
    grad: Callable,
    rtol=0.1,
    min_param=1e-14,
    max_param=1e14,
    max_iter=100,
) -> Tuple[float, float, float, bool]:

    state = LineSearchState(
        param=ParamStruct(
            val=init_param,
            min=min_param,
            max=max_param,
        ),
        fn_val=jnp.inf,
        gd_val=jnp.inf,
        feasible=False,
    )

    param = ParamStruct(
        val=init_param,
        min=min_param,
        max=max_param,
    )

    def regularize(args):
        param, state = args
        return increase_param(param), state

    def update(args):
        param, state = args

        fn_val = fun(param.val)
        gd_val = grad(param.val)

        state = jax.lax.cond(
            jnp.abs(gd_val) < jnp.abs(state.gd_val),
            lambda _: LineSearchState(param, fn_val, gd_val, True),
            lambda _: state,
            None
        )

        param = jax.lax.cond(
            pred=gd_val > 0.0,
            true_fun=reduce_param,
            false_fun=increase_param,
            operand=param
        )
        return param, state

    def _iteration(carry):
        param, state = carry

        fn_val = fun(param.val)
        gd_val = grad(param.val)

        nan_condition = jnp.logical_or(jnp.isnan(fn_val), jnp.isnan(gd_val))
        inf_condition = jnp.logical_or(jnp.isinf(fn_val), jnp.isinf(gd_val))

        return jax.lax.cond(
            pred=jnp.logical_or(nan_condition, inf_condition),
            true_fun=regularize,
            false_fun=update,
            operand=(param, state)
        )

    _, state = bounded_while_loop(
        cond_fun=lambda x: jnp.abs(x[-1].gd_val) > rtol,
        body_fun=_iteration,
        init_val=(param, state),
        maxiter=max_iter,
    )
    return state.param.val, state.fn_val, state.gd_val, state.feasible


def reduce_param(param) -> ParamStruct:
    # set max to current value
    return ParamStruct(
        val=jnp.sqrt(param.min * param.val),
        min=param.min,
        max=param.val
    )


def increase_param(param) -> ParamStruct:
    # set min to current value
    return ParamStruct(
        val=jnp.sqrt(param.val * param.max),
        min=param.val,
        max=param.max
    )


def sample_from_forward_markov(
    rng_key: Array,
    gauss_markov: GaussMarkov,
    num_samples: int
) -> Array:
    """Sample trajectories from forward Markov smoother conditional posteriors.

    The forward Markov smoother result contains conditional posteriors p(x_t | x_{t-1})
    which we can use to sample complete trajectories efficiently using scan.

    Args:
        rng_key: JAX random key
        gauss_markov: GaussMarkov object from forward_markov_smoother
        num_samples: Number of trajectory samples to generate

    Returns:
        Sampled trajectories of shape (num_samples, num_steps, state_dim)
    """

    # Extract components
    marginal = gauss_markov.marginal  # Gaussian
    kernels = gauss_markov.kernels  # AffineGaussian

    num_time_steps, state_dim, _ = kernels.F.shape

    def sample_single_trajectory(sample_key: Array) -> Array:
        """Sample a single trajectory using scan."""

        def sample_step(carry, args):
            key, prev_state = carry
            F_t, d_t, Sigma_t = args

            # Sample next state: x_t | x_{t-1} ~ N(F_t @ x_{t-1} + d_t, Sigma_t)
            sample_key, next_key = jax.random.split(key)
            conditional_mean = F_t @ prev_state + d_t
            next_state = jax.random.multivariate_normal(
                sample_key, conditional_mean, Sigma_t
            )

            return (next_key, next_state), next_state

        # Sample initial state
        init_key, traj_key = jax.random.split(sample_key)
        initial_state = jax.random.multivariate_normal(
            init_key, marginal.mean, marginal.cov
        )

        # Sample trajectory using conditional posteriors
        _, trajectory = jax.lax.scan(
            sample_step,
            (traj_key, initial_state),
            (kernels.F, kernels.d, kernels.Sigma)
        )

        # Prepend initial state to trajectory
        def concat_trees(x, y):
            return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

        trajectory = concat_trees(initial_state, trajectory)
        return trajectory

    # Generate multiple trajectories
    sample_keys = jax.random.split(rng_key, num_samples)
    trajectories = jax.vmap(sample_single_trajectory)(sample_keys)

    return trajectories
