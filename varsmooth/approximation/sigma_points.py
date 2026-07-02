from typing import NamedTuple

import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel
from varsmooth.objects import Gaussian


def get_sqrt(x: Gaussian):
    m_x, cov_x = x
    return m_x, jnp.linalg.cholesky(cov_x)


def get_cov(wc, x_pts, x_mean, y_pts, y_mean):
    tmp = (x_pts - x_mean[None, :]).T * wc[None, :]
    aux = y_pts - y_mean[None, :]
    return jnp.dot(tmp, aux)


class SigmaPoints(NamedTuple):
    points: Array
    wm: Array
    wc: Array
    xi: Array


def linearize_additive(fun, noise, q, get_sigma_points):
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
