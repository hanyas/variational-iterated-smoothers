"""Kalman (RTS) smoother for affine Gaussian state-space models."""

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import AffineGaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import Gaussian
from varsmooth.smoothers.utils import std_forward_message
from varsmooth.utils import none_or_concat
from varsmooth.utils import none_or_shift


def filtering(observations, prior_dist, linear_transition, linear_likelihood):
    """Run the Kalman filter forward pass over an affine-Gaussian model.

    Scans the predict-update recursion from the prior, producing the filtering
    marginals p(x_k | y_1..y_k) with the prior prepended as the root.

    Args:
        observations: Array
            Observation sequence of leading shape (T,).
        prior_dist: Gaussian
            The prior marginal over the root state x_0.
        linear_transition: AffineGaussian
            Batched affine-Gaussian transition kernels of leading shape (T,).
        linear_likelihood: AffineGaussian
            Batched affine-Gaussian observation models of leading shape (T,).

    Returns:
        Gaussian
            The filtering marginals of leading shape (T + 1,), root first.
    """
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

    _, filter_marginals = jax.lax.scan(body, prior_dist, (observations, linear_transition, linear_likelihood))
    return none_or_concat(filter_marginals, prior_dist, 1)


def smoothing(linear_transition: AffineGaussian, filter_trajectory: Gaussian) -> GaussMarkov:
    """Run the RTS backward pass to build a forward Gauss-Markov posterior.

    Scans the Rauch-Tung-Striebel recursion backward over the filtering
    marginals, returning the smoothing posterior as a forward Gauss-Markov chain
    rooted at x_0 with the backward-derived forward kernels.

    Args:
        linear_transition: AffineGaussian
            Batched affine-Gaussian transition kernels of leading shape (T,).
        filter_trajectory: Gaussian
            The filtering marginals of leading shape (T + 1,) from filtering.

    Returns:
        GaussMarkov
            The smoothing posterior (root marginal x_0 + forward kernels).
    """

    def _smooth(F, b, Omega, qf, qs):
        mf, Pf = qf
        ms1, Ps1 = qs  # smoothed marginal at k+1

        S = F @ Pf @ F.T + Omega
        gain = Pf @ jnp.linalg.solve(S, F).T

        ms = mf + gain @ (ms1 - b - F @ mf)
        Ps = Pf + gain @ (Ps1 - S) @ gain.T

        cross = gain @ Ps1
        Ffwd = jnp.linalg.solve(Ps, cross).T
        kernel = AffineGaussian(F=Ffwd, d=ms1 - Ffwd @ ms, Sigma=Ps1 - Ffwd @ cross)
        return Gaussian(ms, Ps), kernel

    def body(carry, args):
        qs = carry
        qf, (F, b, Omega) = args

        qs, kernel = _smooth(F, b, Omega, qf, qs)
        return qs, (qs, kernel)

    last_marginal = jax.tree.map(lambda z: z[-1], filter_trajectory)
    rest_marginals = none_or_shift(filter_trajectory, -1)

    _, (smoothed_marginals, kernels) = jax.lax.scan(
        body, last_marginal, (rest_marginals, linear_transition), reverse=True
    )
    root = jax.tree.map(lambda z: z[0], smoothed_marginals)
    return GaussMarkov(marginal=root, kernels=kernels)


def rts_smoother(
    observations,
    prior_dist: Gaussian,
    linear_transition: AffineGaussian,
    linear_likelihood: AffineGaussian,
) -> Gaussian:
    """Run the full Kalman (RTS) smoother and return its marginals.

    Filters the observations, runs the RTS backward pass, and expands the
    resulting forward Gauss-Markov posterior into standalone smoothing marginals.

    Args:
        observations: Array
            Observation sequence of leading shape (T,).
        prior_dist: Gaussian
            The prior marginal over the root state x_0.
        linear_transition: AffineGaussian
            Batched affine-Gaussian transition kernels of leading shape (T,).
        linear_likelihood: AffineGaussian
            Batched affine-Gaussian observation models of leading shape (T,).

    Returns:
        Gaussian
            The smoothing marginals of leading shape (T + 1,).
    """
    filter_trajectory = filtering(observations, prior_dist, linear_transition, linear_likelihood)
    return std_forward_message(smoothing(linear_transition, filter_trajectory))
