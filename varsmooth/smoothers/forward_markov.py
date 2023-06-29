from typing import Callable
from functools import partial

import jax
from jax import numpy as jnp

from varsmooth.objects import (
    StdGaussian,
    FunctionalModel,
    StdLinearGaussian,
    ForwardGaussMarkov,
    QuadraticFunction,
)
from varsmooth.utils import (
    none_or_concat,
    none_or_shift,
    none_or_idx,
    symmetrize
)

_logdet = lambda x: jnp.linalg.slogdet(x)[1]


@partial(jax.vmap, in_axes=(None, 0, None))
def linearization(
    f: FunctionalModel, x: StdGaussian, method
) -> StdLinearGaussian:
    return StdLinearGaussian(*method(f, x))


def compute_stage_costs(
    observations: jnp.ndarray,
    prior_dist: StdGaussian,
    linear_observation: StdLinearGaussian,
) -> QuadraticFunction:
    def body(_, args):
        y, obsrv_mdl = args

        H, c, R = obsrv_mdl

        Cxx = H.T @ jnp.linalg.solve(R, H)
        cx = H.T @ jnp.linalg.solve(R, y - c)
        c0 = -0.5 * (
            _logdet(2.0 * jnp.pi * R)
            + jnp.dot(y, jnp.linalg.solve(R, y - c))
            - jnp.dot(c, jnp.linalg.solve(R, y - c))
        )
        return _, QuadraticFunction(Cxx, cx, c0)

    _, stage_cost = jax.lax.scan(body, (), (observations, linear_observation))

    m0, P0 = prior_dist
    Cxx = jnp.linalg.inv(P0)
    cx = m0.T @ jnp.linalg.inv(P0)
    c0 = -0.5 * (
        _logdet(2.0 * jnp.pi * P0) + jnp.dot(m0, jnp.linalg.solve(P0, m0))
    )
    init_cost = QuadraticFunction(Cxx, cx, c0)

    return none_or_concat(stage_cost, init_cost, 1)


def backward_pass(
    stage_cost: QuadraticFunction,
    linear_transition: StdLinearGaussian,
    nominal_posterior: ForwardGaussMarkov,
    damping: float,
) -> (ForwardGaussMarkov, jnp.ndarray):

    def _backwards(carry, args):
        vfunc = carry
        cost, trans_mdl, nominal = args

        Vxx, vx, v0 = vfunc
        Cxx, cx, c0 = cost

        A, b, Q = trans_mdl
        pF, pr, pS = nominal

        Psi = symmetrize(Vxx + jnp.linalg.inv(Q) + damping * jnp.linalg.inv(pS))
        Gamma = jnp.linalg.solve(Q, A) + damping * jnp.linalg.solve(pS, pF)
        tau = vx + jnp.linalg.solve(Q, b) + damping * jnp.linalg.solve(pS, pr)

        def _feasible(args):
            Vxx, vx, v0, Cxx, cx, c0, A, b, Q, pF, pr, pS, Psi, Gamma, tau = args

            nF = jnp.linalg.inv(Psi) @ Gamma
            nr = jnp.linalg.inv(Psi) @ tau
            nS = (1.0 + damping) * jnp.linalg.inv(Psi)

            Vxx = symmetrize(
                Cxx
                + A.T @ jnp.linalg.solve(Q, A)
                - Gamma.T @ jnp.linalg.solve(Psi, Gamma)
                + damping * pF.T @ jnp.linalg.solve(pS, pF)
            )
            vx = (
                cx
                - A.T @ jnp.linalg.solve(Q, b)
                + Gamma.T @ jnp.linalg.solve(Psi, tau)
                - damping * pF.T @ jnp.linalg.solve(pS, pr)
            )
            v0 = (
                c0
                + v0
                + 0.5 * (1.0 + damping) * _logdet(2.0 * jnp.pi * jnp.linalg.inv(Psi))
                - 0.5 * _logdet(2.0 * jnp.pi * Q)
                - 0.5 * jnp.dot(b, jnp.linalg.solve(Q, b))
                + 0.5 * jnp.dot(tau, jnp.linalg.solve(Psi, tau))
                - 0.5 * damping * _logdet(2.0 * jnp.pi * pS)
                - 0.5 * damping * jnp.dot(pr, jnp.linalg.solve(pS, pr))
            )

            return QuadraticFunction(Vxx, vx, v0), (StdLinearGaussian(nF, nr, nS), True)

        def _not_feasible(args):
            Vxx, vx, v0, Cxx, cx, c0, A, b, Q, pF, pr, pS, Psi, Gamma, tau = args
            return QuadraticFunction(Vxx, vx, v0), (StdLinearGaussian(pF, pr, pS), False)

        return jax.lax.cond(
            jnp.all(jnp.linalg.eigvals(Psi) >= 0.0),
            _feasible,
            _not_feasible,
            (Vxx, vx, v0, Cxx, cx, c0, A, b, Q, pF, pr, pS, Psi, Gamma, tau),
        )

    vfunc, (next_kernels, feasible) = jax.lax.scan(
        f=_backwards,
        init=none_or_idx(stage_cost, -1),
        xs=(
            none_or_shift(stage_cost, -1),
            linear_transition,
            nominal_posterior.kernels,
        ),
        reverse=True,
    )

    Vxx, vx, v0 = vfunc
    pm, pP = nominal_posterior.init

    Psi = symmetrize(Vxx + damping * jnp.linalg.inv(pP))
    tau = vx + damping * jnp.linalg.solve(pP, pm)

    def _feasible(args):
        nm = jnp.linalg.inv(Psi) @ tau
        nP = (1.0 + damping) * jnp.linalg.inv(Psi)
        return StdGaussian(nm, nP), True

    def _not_feasible(args):
        return StdGaussian(pm, pP), False

    next_init, init_feasible = jax.lax.cond(
        jnp.all(jnp.linalg.eigvals(Psi) > 0.0),
        _feasible,
        _not_feasible,
        (),
    )

    feasible = jnp.hstack((init_feasible, feasible))
    return ForwardGaussMarkov(next_init, next_kernels), jnp.all(feasible)


def forward_pass(posterior: ForwardGaussMarkov) -> StdGaussian:
    x0, kernels = posterior

    def _forwards(carry, args):
        x = carry
        kernel = args

        last_m, last_P = x
        F, r, S = kernel

        next_m = F @ last_m + r
        next_P = F @ last_P @ F.T + S
        next_x = StdGaussian(next_m, next_P)

        return next_x, next_x

    _, xs = jax.lax.scan(_forwards, x0, kernels)
    xs = none_or_concat(xs, x0, 1)
    return xs


def dual_objective(posterior, damping):
    return (1.0 + damping) * (-0.5 * _logdet(2.0 * jnp.pi * posterior.init.cov))


def iterated_smoother(
    observations: jnp.ndarray,
    prior_dist: StdGaussian,
    transition_model: FunctionalModel,
    observation_model: FunctionalModel,
    linearization_method: Callable,
    initial_posterior: ForwardGaussMarkov,
    min_damping: float,
    max_damping: float,
    init_damping: float,
    damping_mult: float,
    max_iter: int
) -> (ForwardGaussMarkov, StdGaussian):

    def loop(carry, args):
        nominal_posterior, damping = carry

        # compute posterior marginals
        posterior_marginals = forward_pass(nominal_posterior)

        # linearize models around marginals
        trans_marginals = none_or_shift(posterior_marginals, -1)
        linear_transition = linearization(
            transition_model, trans_marginals, linearization_method
        )

        obsrv_marginals = none_or_shift(posterior_marginals, 1)
        linear_observation = linearization(
            observation_model, obsrv_marginals, linearization_method
        )

        # pre-compute stage cost (log-obs and log-prior)
        stage_cost = compute_stage_costs(
            observations, prior_dist, linear_observation
        )

        # update posterior kernels
        _posterior, feasible = backward_pass(
            stage_cost,
            linear_transition,
            nominal_posterior,
            damping,
        )

        def accept(args):
            damping = args
            return _posterior, damping / damping_mult

        def reject(args):
            damping = args
            return nominal_posterior, damping * damping_mult

        next_posterior, next_damping = jax.lax.cond(feasible, accept, reject, damping)
        return (next_posterior, next_damping), (next_posterior, next_damping)

    (next_posterior, _), (_, dampings) = jax.lax.scan(
        loop, (initial_posterior, init_damping), jnp.arange(max_iter)
    )
    return next_posterior, forward_pass(next_posterior), dampings


# TESTING

def _vanilla_backward_pass(
    stage_cost: QuadraticFunction,
    linear_transition: StdLinearGaussian,
) -> ForwardGaussMarkov:

    def _backwards(carry, args):
        vfunc = carry
        cost, trans_mdl = args

        Vxx, vx, v0 = vfunc
        Cxx, cx, c0 = cost

        A, b, Q = trans_mdl

        Psi = symmetrize(Vxx + jnp.linalg.inv(Q))
        Gamma = jnp.linalg.solve(Q, A)
        tau = vx + jnp.linalg.solve(Q, b)

        F = jnp.linalg.inv(Psi) @ Gamma
        r = jnp.linalg.inv(Psi) @ tau
        S = jnp.linalg.inv(Psi)

        Vxx = symmetrize(
            Cxx
            + A.T @ jnp.linalg.solve(Q, A)
            - Gamma.T @ jnp.linalg.solve(Psi, Gamma)
        )
        vx = (
            cx
            - A.T @ jnp.linalg.solve(Q, b)
            + Gamma.T @ jnp.linalg.solve(Psi, tau)
        )
        v0 = (
            c0
            + v0
            + 0.5 * _logdet(2.0 * jnp.pi * jnp.linalg.inv(Psi))
            - 0.5 * _logdet(2.0 * jnp.pi * Q)
            - 0.5 * jnp.dot(b, jnp.linalg.solve(Q, b))
            + 0.5 * jnp.dot(tau, jnp.linalg.solve(Psi, tau))
        )

        return QuadraticFunction(Vxx, vx, v0), StdLinearGaussian(F, r, S)

    vfunc, next_kernels = jax.lax.scan(
        f=_backwards,
        init=none_or_idx(stage_cost, -1),
        xs=(
            none_or_shift(stage_cost, -1),
            linear_transition,
        ),
        reverse=True,
    )

    Vxx, vx, v0 = vfunc

    m = jnp.linalg.inv(Vxx) @ vx
    P = jnp.linalg.inv(Vxx)

    next_init = StdGaussian(m, P)
    return ForwardGaussMarkov(next_init, next_kernels)


def _var_smoother(
    observations: jnp.ndarray,
    initial_dist: StdGaussian,
    linear_transition: StdLinearGaussian,
    linear_observation: StdLinearGaussian,
    nominal_posterior: ForwardGaussMarkov,
    damping: float
) -> StdGaussian:
    # pre-compute stage cost
    stage_cost = compute_stage_costs(
        observations, initial_dist, linear_observation
    )

    # # update posterior kernels
    # posterior = _vanilla_backward_pass(
    #     stage_cost,
    #     linear_transition,
    # )

    # update posterior kernels
    posterior, _ = backward_pass(
        stage_cost,
        linear_transition,
        nominal_posterior,
        damping
    )

    return forward_pass(posterior)
