import jax
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import Gaussian, AffineGaussian
from varsmooth.utils import none_or_concat, none_or_shift


def filtering(
    observations: jnp.ndarray,
    prior_dist: Gaussian,
    linear_transition: AffineGaussian,
    linear_observation: AffineGaussian,
):
    def _predict(F, b, Omega, q):
        m, P = q

        m = F @ m + b
        P = Omega + F @ P @ F.T
        return Gaussian(m, P)

    def _update(H, e, Delta, q, y):
        m, P = q

        S = Delta + H @ P @ H.T
        G = jsc.linalg.solve(S.T, H @ P.T).T

        m = m + G @ (y - H @ m - e)
        P = P - G @ S @ G.T
        return Gaussian(m, P)

    def body(carry, args):
        qf = carry
        y, (A, b, Omega), (H, e, Delta) = args

        qp = _predict(A, b, Omega, qf)
        qf = _update(H, e, Delta, qp, y)
        return qf, qf

    _, filter_marginals = jax.lax.scan(
        body,
        prior_dist,
        (observations, linear_transition, linear_observation)
    )
    return none_or_concat(filter_marginals, prior_dist, 1)


def smoothing(
    linear_transition: AffineGaussian,
    filter_trajectory: Gaussian,
):
    def _smooth(F, b, Omega, qf, qs):
        mf, Pf = qf
        ms, Ps = qs

        S = F @ Pf @ F.T + Omega

        gain = Pf @ jnp.linalg.solve(S, F).T
        ms = mf + gain @ (ms - b - F @ mf)
        Ps = Pf + gain @ (Ps - S) @ gain.T
        return Gaussian(ms, Ps)

    def body(carry, args):
        qs = carry
        qf, (F, b, Omega) = args

        qs = _smooth(F, b, Omega, qf, qs)
        return qs, qs

    last_marginal = jax.tree_map(lambda z: z[-1], filter_trajectory)
    rest_marginals = none_or_shift(filter_trajectory, -1)

    _, smoothed_marginals = jax.lax.scan(
        body,
        last_marginal,
        (rest_marginals, linear_transition),
        reverse=True
    )
    return none_or_concat(smoothed_marginals, last_marginal, -1)


def rts_smoother(
    observations: jnp.ndarray,
    prior_dist: Gaussian,
    linear_transition: AffineGaussian,
    linear_observation: AffineGaussian,
) -> Gaussian:

    filter_trajectory = filtering(observations, prior_dist, linear_transition, linear_observation)
    return smoothing(linear_transition, filter_trajectory)
