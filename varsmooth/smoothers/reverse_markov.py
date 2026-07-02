from typing import Tuple

import jax
from jax import Array
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
from varsmooth.smoothers.utils import kl_between_reverse_gauss_markovs
from varsmooth.smoothers.utils import std_backward_message
from varsmooth.utils import none_or_concat
from varsmooth.utils import symmetrize


def log_forward_message(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    reverse_reference: GaussMarkov,
    damping: float,
) -> Tuple[GaussMarkov, LogMarginalNorm, ValueFn, LogMessage, Array]:

    def _forward_step(carry, args):
        R, r, rho = carry
        C11, C12, C21, C22, c1, c2, kappa, L, l, nu, F, d, Sigma = args

        dim = Sigma.shape[0]
        # Factor the kernel covariance once and reuse it for every Sigma-solve.
        chol_Sigma = jsc.linalg.cho_factor(Sigma)
        iSig = jsc.linalg.cho_solve(chol_Sigma, jnp.eye(dim))
        iSig_F = jsc.linalg.cho_solve(chol_Sigma, F)
        iSig_d = jsc.linalg.cho_solve(chol_Sigma, d)
        logdet_Sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_Sigma[0])))

        G11 = (1.0 - damping) * C11 + damping * F.T @ iSig_F
        G22 = (1.0 - damping) * (C22 + R) + damping * iSig
        G21 = (1.0 - damping) * C21 + damping * iSig_F
        g1 = (1.0 - damping) * c1 - damping * F.T @ iSig_d
        g2 = (1.0 - damping) * (c2 + r) + damping * iSig_d
        theta = (
            (1.0 - damping) * (kappa + rho)
            - 0.5 * damping * (dim * jnp.log(2 * jnp.pi) + logdet_Sigma)
            - 0.5 * damping * d.T @ iSig_d
        )

        G11 = symmetrize(G11)
        G22 = symmetrize(G22)

        # Feasibility via a Cholesky attempt: non-PD G22 yields a non-finite factor.
        chol_G22 = jnp.linalg.cholesky(G22)
        pd_G22 = jnp.all(jnp.isfinite(chol_G22))

        def _feasible_forward_pass():
            iG22_G21 = jsc.linalg.cho_solve((chol_G22, True), G21)
            iG22_g2 = jsc.linalg.cho_solve((chol_G22, True), g2)
            Sigma = jsc.linalg.cho_solve((chol_G22, True), jnp.eye(dim))
            logdet_G22 = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_G22)))

            S = G11 - G21.T @ iG22_G21
            s = g1 + G21.T @ iG22_g2
            xi = theta + 0.5 * (dim * jnp.log(2 * jnp.pi) - logdet_G22) + 0.5 * g2.T @ iG22_g2

            F = iG22_G21
            d = iG22_g2

            R = L + 1.0 / (1.0 - damping) * S
            r = l + 1.0 / (1.0 - damping) * s
            rho = nu + 1.0 / (1.0 - damping) * xi

            value_fn = ValueFn(R, r, rho)
            return value_fn, (value_fn, AffineGaussian(F, d, Sigma), LogMessage(S, s, xi), True)  # feasible

        def _not_feasible_forward_pass():
            S = jnp.zeros_like(G11)
            s = jnp.zeros_like(g1)
            xi = jnp.zeros_like(theta)

            R = jnp.zeros_like(L)
            r = jnp.zeros_like(l)
            rho = jnp.zeros_like(nu)

            value_fn = ValueFn(R, r, rho)
            return value_fn, (value_fn, AffineGaussian(F, d, Sigma), LogMessage(S, s, xi), False)  # Not feasible

        return jax.lax.cond(
            pred=pd_G22,
            true_fun=_feasible_forward_pass,
            false_fun=_not_feasible_forward_pass,
        )

    first_value_fn = ValueFn(R=log_prior.L, r=log_prior.l, rho=log_prior.nu)

    nominal_marginal, nominal_kernels = reverse_reference

    last_value_fn, (value_fns, kernels, log_fwd_msgs, feasible_flags) = jax.lax.scan(
        f=_forward_step,
        init=first_value_fn,
        xs=(*log_transition, *log_observation, *nominal_kernels),
    )
    value_fns = none_or_concat(value_fns, first_value_fn, 1)

    R, r, rho = last_value_fn

    m, P = nominal_marginal
    dim = P.shape[0]
    chol_P = jsc.linalg.cho_factor(P)
    inv_P = jsc.linalg.cho_solve(chol_P, jnp.eye(dim))
    logdet_P = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_P[0])))

    J11 = damping * inv_P
    J21 = damping * inv_P
    J22 = (1.0 - damping) * R + damping * inv_P
    j2 = (1.0 - damping) * r
    j1 = jnp.zeros_like(j2)
    tau = (1.0 - damping) * rho - 0.5 * damping * (dim * jnp.log(2 * jnp.pi) + logdet_P)

    J11 = symmetrize(J11)
    J22 = symmetrize(J22)

    def _feasible_marginal():
        chol_J22 = jsc.linalg.cho_factor(J22)
        iJ22_J21 = jsc.linalg.cho_solve(chol_J22, J21)
        iJ22_j2 = jsc.linalg.cho_solve(chol_J22, j2)
        _P = jsc.linalg.cho_solve(chol_J22, jnp.eye(dim))
        logdet_J22 = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_J22[0])))

        # init marginal
        _m = jsc.linalg.cho_solve(chol_J22, j2 + J21 @ m)

        # log normalizer
        U = J11 - J21.T @ iJ22_J21
        u = j1 - J21.T @ iJ22_j2
        eta = tau + 0.5 * (dim * jnp.log(2 * jnp.pi) - logdet_J22) + 0.5 * j2.T @ iJ22_j2
        return Gaussian(_m, _P), LogMarginalNorm(U, u, eta)

    def _not_feasible_marginal():
        _m = jnp.zeros_like(nominal_marginal.mean)
        _P = jnp.zeros_like(nominal_marginal.cov)

        U = jnp.zeros_like(J11)
        u = jnp.zeros_like(j1)
        eta = jnp.zeros_like(tau)
        return Gaussian(_m, _P), LogMarginalNorm(U, u, eta)

    marginal, log_marg_norm = jax.lax.cond(
        pred=jnp.all(feasible_flags),
        true_fun=_feasible_marginal,
        false_fun=_not_feasible_marginal,
    )
    return (GaussMarkov(marginal, kernels), log_marg_norm, value_fns, log_fwd_msgs, feasible_flags)


(
    reverse_markov_smoother,
    dual_objective,
    log_evidence,
    iterated_reverse_markov_smoother,
) = make_smoother_suite(
    log_message_fn=log_forward_message,
    std_marginal_fn=std_backward_message,
    kl_fn=kl_between_reverse_gauss_markovs,
)
