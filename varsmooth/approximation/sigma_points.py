from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel
from varsmooth.objects import Gaussian


def get_sqrt(x: Gaussian):
    """Return the mean and lower-Cholesky factor of a Gaussian's covariance.

    Args:
        x: Gaussian
            Gaussian to factor.

    Returns:
        mean: Array
            Mean vector of shape (dx,).
        chol: Array
            Lower-triangular Cholesky factor of the covariance, shape (dx, dx).
    """
    m_x, cov_x = x
    return m_x, jnp.linalg.cholesky(cov_x)


def get_cov(wc, x_pts, x_mean, y_pts, y_mean):
    """Compute the weighted cross-covariance between two sigma-point sets.

    Args:
        wc: Array
            Covariance weights of shape (n_points,).
        x_pts: Array
            First point set of shape (n_points, dx).
        x_mean: Array
            Mean of the first set, shape (dx,).
        y_pts: Array
            Second point set of shape (n_points, dy).
        y_mean: Array
            Mean of the second set, shape (dy,).

    Returns:
        Array
            The weighted cross-covariance
            sum_n wc_n (x_pts_n - x_mean) (y_pts_n - y_mean)^T, shape (dx, dy).
    """
    tmp = (x_pts - x_mean[None, :]).T * wc[None, :]
    aux = y_pts - y_mean[None, :]
    return jnp.dot(tmp, aux)


class SigmaPoints(NamedTuple):
    """A set of sigma points with mean and covariance weights.

    Attributes:
        points: Array
            Transformed sigma points of shape (n_points, dx), i.e. the unit
            points mapped through the current mean and Cholesky factor.
        wm: Array
            Mean weights of shape (n_points,).
        wc: Array
            Covariance weights of shape (n_points,).
        xi: Array
            The unit (pre-transform) sigma points of shape (dx, n_points),
            retained for reuse; not read by the linearize/quadratize routines,
            which operate on the transformed points above.
    """

    points: Array
    wm: Array
    wc: Array
    xi: Array


def linearize_additive(fun, noise, q, get_sigma_points):
    """Statistically linearize an additive-noise map fun(x) + noise under q.

    Fits an affine-Gaussian approximation x -> N(F x + d, Sigma) by sigma-point
    regression of fun at the expansion point q.

    Args:
        fun: Callable
            Deterministic map applied to the state.
        noise: Gaussian
            Additive Gaussian noise offsetting and spreading the output.
        q: Gaussian
            Expansion point at which the regression is taken.
        get_sigma_points: Callable
            Sigma-point rule (m, chol_P) -> SigmaPoints.

    Returns:
        F: Array
            Linear map of the affine-Gaussian approximation.
        d: Array
            Offset vector of the approximation.
        Sigma: Array
            Covariance of the approximation.
    """
    m_x, chol_x = get_sqrt(q)
    x_pts = get_sigma_points(m_x, chol_x)

    f_pts = jax.vmap(fun)(x_pts.points)
    m_f = jnp.dot(x_pts.wm, f_pts)

    Psi = get_cov(x_pts.wc, x_pts.points, m_x, f_pts, m_f)
    F = cho_solve((chol_x, True), Psi).T

    m_x, cov_x = q
    Phi = get_cov(x_pts.wc, f_pts, m_f, f_pts, m_f)
    L = Phi - F @ cov_x @ F.T + noise.cov
    return F, m_f - F @ m_x + noise.mean, 0.5 * (L + L.T)


def linearize_conditional(cond_mean, cond_cov, q, get_sigma_points):
    """Statistically linearize a conditional-moments model under q.

    Fits an affine-Gaussian approximation x -> N(F x + d, Sigma) by sigma-point
    regression of the state-dependent moments at the expansion point q.

    Args:
        cond_mean: Callable
            Conditional mean E[y | x] as a function of x.
        cond_cov: Callable
            Conditional covariance Cov[y | x] as a function of x.
        q: Gaussian
            Expansion point at which the regression is taken.
        get_sigma_points: Callable
            Sigma-point rule (m, chol_P) -> SigmaPoints.

    Returns:
        F: Array
            Linear map of the affine-Gaussian approximation.
        d: Array
            Offset vector of the approximation.
        Sigma: Array
            Covariance of the approximation.
    """
    m_x, chol_x = get_sqrt(q)
    x_pts = get_sigma_points(m_x, chol_x)

    cm_pts = jax.vmap(cond_mean)(x_pts.points)
    m_cm = jnp.dot(x_pts.wm, cm_pts)

    Psi = get_cov(x_pts.wc, x_pts.points, m_x, cm_pts, m_cm)
    F = cho_solve((chol_x, True), Psi).T

    cc_pts = jax.vmap(cond_cov)(x_pts.points)
    m_cc = jnp.sum(x_pts.wc[:, None, None] * cc_pts, 0)

    Phi = get_cov(x_pts.wc, cm_pts, m_cm, cm_pts, m_cm)
    L = Phi - (F @ chol_x) @ (F @ chol_x).T + m_cc
    return F, m_cm - F @ m_x, L


def make_linearize(get_sigma_points):
    """Build a model-dispatching linearize from a sigma-point rule.

    Args:
        get_sigma_points: Callable
            Sigma-point rule (m, chol_P) -> SigmaPoints for the chosen scheme.

    Returns:
        linearize: Callable
            A function (model, q) -> (F, d, Sigma) that dispatches on the model
            type (AdditiveGaussianModel or ConditionalMomentsModel) and raises
            TypeError for anything else.
    """

    def linearize(model, q):
        if isinstance(model, AdditiveGaussianModel):
            return linearize_additive(model.fun, model.noise, q, get_sigma_points)
        if isinstance(model, ConditionalMomentsModel):
            return linearize_conditional(model.mean_fn, model.cov_fn, q, get_sigma_points)
        raise TypeError(f"Unsupported model type: {type(model).__name__}")

    return linearize


def quadratize_any(f, q, get_sigma_points):
    """Quadratize a scalar function f under q from sigma-point-averaged derivatives.

    Averages the Jacobian and Hessian of f over the sigma points of q to build
    the second-order expansion -0.5 x^T M x + v^T x + c.

    Args:
        f: Callable
            Scalar function to quadratize (typically a log-density).
        q: Gaussian
            Expansion point under which the derivatives are averaged.
        get_sigma_points: Callable
            Sigma-point rule (m, chol_P) -> SigmaPoints.

    Returns:
        M: Array
            Quadratic-form matrix of the expansion, shape (dx, dx).
        v: Array
            Linear coefficient vector of shape (dx,).
        c: Array
            Scalar constant offset.
    """
    m_x, chol_x = get_sqrt(q)
    x_pts = get_sigma_points(m_x, chol_x)

    H_fn = lambda x: jax.jacrev(jax.jacrev(f))(x)
    J_fn = lambda x: jax.jacrev(f)(x)

    Hs = jax.vmap(H_fn)(x_pts.points)
    E_H = jnp.einsum("n,nkh->kh", x_pts.wm, Hs)
    Fxx = -E_H

    Js = jax.vmap(J_fn)(x_pts.points)
    E_J = jnp.einsum("n,nk->k", x_pts.wm, Js)
    Fx = E_J - E_H @ m_x

    fs = jax.vmap(f)(x_pts.points)
    E_f = jnp.dot(x_pts.wm, fs)
    f0 = E_f - jnp.dot(E_J, m_x) + 0.5 * m_x.T @ E_H @ m_x - 0.5 * jnp.trace(E_H @ (chol_x @ chol_x.T))
    return Fxx, Fx, f0
