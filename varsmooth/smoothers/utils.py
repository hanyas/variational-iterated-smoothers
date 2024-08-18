from typing import Callable, Tuple, NamedTuple
from functools import partial

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from jaxopt._src.loop import while_loop as while_with_maxiter

from varsmooth.objects import (
    Gaussian,
    AffineGaussian,
    GaussMarkov,
    LogPrior,
    LogTransition,
    LogObservation,
    Potential,
    LogConditionalNorm
)
from varsmooth.utils import none_or_idx, none_or_shift, logdet


def kl_between_marginals(p, q):
    dim = p.mean.shape[0]
    return 0.5 * (
        jnp.trace(jsc.linalg.inv(q.cov) @ p.cov) - dim
        + (q.mean - p.mean).T @ jsc.linalg.solve(q.cov, q.mean - p.mean)
        + logdet(q.cov) - logdet(p.cov)
    )


@partial(jax.jit, static_argnums=(1, 2, 3))
def statistical_expansion(
    observations: jnp.ndarray,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    posterior_kernels: AffineGaussian,
    posterior_marginals: Gaussian
) -> (LogPrior, LogTransition, LogObservation):

    init_marginal = none_or_idx(posterior_marginals, 0)
    prev_marginals = none_or_shift(posterior_marginals, -1)
    next_marginals = none_or_shift(posterior_marginals, 1)

    log_prior = log_prior_fn(init_marginal)
    log_transition = log_transition_fn(prev_marginals, posterior_kernels)
    log_observation = log_observation_fn(observations, next_marginals)
    return log_prior, log_transition, log_observation


def initialize_reverse_with_forward(
    forward_markov: GaussMarkov
):
    from varsmooth.smoothers.forward_markov import forward_std_message

    forward_marginals = forward_std_message(forward_markov)

    Fs = jnp.zeros_like(forward_markov.kernels.F)
    ds = jnp.zeros_like(forward_markov.kernels.d)
    Sigmas = jnp.zeros_like(forward_markov.kernels.Sigma)

    nb_steps = forward_markov.kernels.F.shape[0]
    for k in range(nb_steps):

        marginal = none_or_idx(forward_marginals, k)  # marginal at k
        kernel = none_or_idx(forward_markov.kernels, k)  # kernel k+1 | k
        next_marginal = none_or_idx(forward_marginals, k+1)  # marginal at k+1

        _kernel = get_reverse_kernel(marginal, kernel, next_marginal)

        Fs = Fs.at[k].set(_kernel.F)
        ds = ds.at[k].set(_kernel.d)
        Sigmas = Sigmas.at[k].set(_kernel.Sigma)

    reverse_markov = GaussMarkov(
        Gaussian(
            mean=forward_marginals.mean[-1],
            cov=forward_marginals.cov[-1],
        ),
        kernels=AffineGaussian(
            F=Fs, d=ds, Sigma=Sigmas
        )
    )
    return reverse_markov


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
    fwd_message: Potential,
    bwd_message: LogConditionalNorm,
) -> Potential:
    return Potential(
        R=(fwd_message.R + bwd_message.S),
        r=(fwd_message.r + bwd_message.s),
        rho=(fwd_message.rho + bwd_message.xi),
    )


def log_to_std_form(
    potential: Potential
) -> Gaussian:
    return Gaussian(
        mean=jsc.linalg.inv(potential.R) @ potential.r,
        cov=jsc.linalg.inv(potential.R)
    )


def std_to_log_form(
    dist: Gaussian
) -> Potential:
    return Potential(
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
    min_param=1e-4,
    max_param=1e14,
    max_iter=100,
) -> (float, float, float):

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

    _, state = while_with_maxiter(
        cond_fun=lambda x: jnp.abs(x[-1].gd_val) > rtol,
        body_fun=_iteration,
        init_val=(param, state),
        maxiter=max_iter,
        jit=True,
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


# def line_search(
#     param: float,
#     fun: Callable,
#     grad: Callable,
#     rtol=0.1,
#     max_iters=100,
#     verbose=False,
# ) -> (float, float, float):
#     min_param = 1e-4
#     max_param = 1e14
#
#     best_param = param
#     best_fun_val = jnp.inf
#     best_grad_val = jnp.inf
#     for k in range(max_iters):
#         fun_val, grad_val = fun(param), grad(param)
#
#         if verbose:
#             print(param, min_param, max_param, fun_val, grad_val)
#
#         if jnp.logical_or(
#             jnp.logical_or(jnp.isnan(fun_val), jnp.isnan(grad_val)),
#             jnp.logical_or(jnp.isinf(fun_val), jnp.isinf(grad_val))
#         ):
#             min_param = param
#             param = jnp.sqrt(min_param * max_param)
#         else:
#             if grad_val < best_grad_val:
#                 best_param = param
#                 best_fun_val = fun_val
#                 best_grad_val = grad_val
#             if jnp.abs(grad_val) <= rtol:
#                 return param, fun_val, grad_val
#             else:
#                 if grad_val > 0.0:  # param too large
#                     max_param = param
#                     param = jnp.sqrt(min_param * max_param)
#                 else:               # param too small
#                     min_param = param
#                     param = jnp.sqrt(min_param * max_param)
#
#     return best_param, best_fun_val, best_grad_val
