from typing import Tuple

import jax
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import LogMarginalNorm
from varsmooth.objects import LogMessage
from varsmooth.objects import LogObservation
from varsmooth.objects import LogPrior
from varsmooth.objects import LogTransition
from varsmooth.objects import ValueFn
from varsmooth.smoothers.core import make_smoother_suite
from varsmooth.smoothers.utils import kl_between_forward_gauss_markovs
from varsmooth.smoothers.utils import std_forward_message
from varsmooth.utils import logdet
from varsmooth.utils import none_or_concat
from varsmooth.utils import none_or_idx
from varsmooth.utils import none_or_shift
from varsmooth.utils import symmetrize


def log_backward_message(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    forward_reference: GaussMarkov,
    damping: float,
) -> Tuple[GaussMarkov, LogMarginalNorm, ValueFn, LogMessage, bool]:

    def _backward_step(carry, args):
        R, r, rho = carry
        C11, C12, C21, C22, c1, c2, kappa, L, l, nu, F, d, Sigma = args

        G11 = (1.0 - damping) * (C11 + R) + damping * jsc.linalg.inv(Sigma)
        G22 = (1.0 - damping) * C22 + damping * F.T @ jsc.linalg.solve(Sigma, F)
        G12 = (1.0 - damping) * C12 + damping * jsc.linalg.solve(Sigma, F)
        g1 = (1.0 - damping) * (c1 + r) + damping * jsc.linalg.solve(Sigma, d)
        g2 = (1.0 - damping) * c2 - damping * F.T @ jsc.linalg.solve(Sigma, d)
        theta = (
            (1.0 - damping) * (kappa + rho)
            - 0.5 * damping * logdet(2 * jnp.pi * Sigma)
            - 0.5 * damping * d.T @ jsc.linalg.solve(Sigma, d)
        )

        G11 = symmetrize(G11)
        G22 = symmetrize(G22)

        def _feasible_backward_pass():
            dim = G11.shape[0]
            chol_G11 = jsc.linalg.cho_factor(G11)
            iG11_G12 = jsc.linalg.cho_solve(chol_G11, G12)
            iG11_g1 = jsc.linalg.cho_solve(chol_G11, g1)
            Sigma = jsc.linalg.cho_solve(chol_G11, jnp.eye(dim))
            logdet_G11 = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_G11[0])))

            S = G22 - G12.T @ iG11_G12
            s = g2 + G12.T @ iG11_g1
            xi = theta + 0.5 * (dim * jnp.log(2 * jnp.pi) - logdet_G11) + 0.5 * g1.T @ iG11_g1

            F = iG11_G12
            d = iG11_g1

            R = L + 1.0 / (1.0 - damping) * S
            r = l + 1.0 / (1.0 - damping) * s
            rho = nu + 1.0 / (1.0 - damping) * xi

            value_fn = ValueFn(R, r, rho)
            return value_fn, (value_fn, AffineGaussian(F, d, Sigma), LogMessage(S, s, xi), True)  # feasible

        def _not_feasible_backward_pass():
            S = jnp.zeros_like(G22)
            s = jnp.zeros_like(g2)
            xi = jnp.zeros_like(theta)

            R = jnp.zeros_like(L)
            r = jnp.zeros_like(l)
            rho = jnp.zeros_like(nu)

            value_fn = ValueFn(R, r, rho)
            return value_fn, (value_fn, AffineGaussian(F, d, Sigma), LogMessage(S, s, xi), False)  # Not feasible

        return jax.lax.cond(
            pred=jnp.all(jnp.linalg.eigvalsh(G11) > 1e-8),
            true_fun=_feasible_backward_pass,
            false_fun=_not_feasible_backward_pass,
        )

    last_log_obs = none_or_idx(log_observation, -1)
    last_value_fn = ValueFn(R=last_log_obs.L, r=last_log_obs.l, rho=last_log_obs.nu)

    log_aux_obs = none_or_concat(
        none_or_shift(log_observation, -1),
        LogObservation(log_prior.L, log_prior.l, log_prior.nu),
        1,
    )

    nominal_marginal, nominal_kernels = forward_reference

    first_value_fn, (value_fns, kernels, log_bwd_msgs, feasible_pass) = jax.lax.scan(
        f=_backward_step,
        init=last_value_fn,
        xs=(*log_transition, *log_aux_obs, *nominal_kernels),
        reverse=True,
    )
    value_fns = none_or_concat(value_fns, last_value_fn, -1)

    R, r, rho = first_value_fn

    m, P = nominal_marginal
    inv_P = jsc.linalg.inv(P)

    J11 = (1.0 - damping) * R + damping * inv_P
    J12 = damping * inv_P
    J22 = damping * inv_P
    j1 = (1.0 - damping) * r
    j2 = jnp.zeros_like(j1)
    tau = (1.0 - damping) * rho - 0.5 * damping * logdet(2 * jnp.pi * P)

    J11 = symmetrize(J11)
    J22 = symmetrize(J22)

    def _feasible_marginal():
        dim = J11.shape[0]
        chol_J11 = jsc.linalg.cho_factor(J11)
        iJ11_J12 = jsc.linalg.cho_solve(chol_J11, J12)
        iJ11_j1 = jsc.linalg.cho_solve(chol_J11, j1)
        _P = jsc.linalg.cho_solve(chol_J11, jnp.eye(dim))
        logdet_J11 = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_J11[0])))

        # init marginal
        _m = jsc.linalg.cho_solve(chol_J11, j1 + J12 @ m)

        # log normalizer
        U = J22 - J12.T @ iJ11_J12
        u = j2 - J12.T @ iJ11_j1
        eta = tau + 0.5 * (dim * jnp.log(2 * jnp.pi) - logdet_J11) + 0.5 * j1.T @ iJ11_j1
        return Gaussian(_m, _P), LogMarginalNorm(U, u, eta)

    def _not_feasible_marginal():
        _m = jnp.zeros_like(nominal_marginal.mean)
        _P = jnp.zeros_like(nominal_marginal.cov)

        U = jnp.zeros_like(J22)
        u = jnp.zeros_like(j2)
        eta = jnp.zeros_like(tau)
        return Gaussian(_m, _P), LogMarginalNorm(U, u, eta)

    marginal, log_marg_norm = jax.lax.cond(
        pred=jnp.all(feasible_pass),
        true_fun=_feasible_marginal,
        false_fun=_not_feasible_marginal,
    )
    return (GaussMarkov(marginal, kernels), log_marg_norm, value_fns, log_bwd_msgs, feasible_pass)


(
    forward_markov_smoother,
    dual_objective,
    log_evidence,
    iterated_forward_markov_smoother,
) = make_smoother_suite(
    log_message_fn=log_backward_message,
    std_marginal_fn=std_forward_message,
    kl_fn=kl_between_forward_gauss_markovs,
)
