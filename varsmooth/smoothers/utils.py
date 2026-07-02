from functools import partial
from typing import Callable, NamedTuple

import jax
from jax import Array
from jax import numpy as jnp
from jax import scipy as jsc

from varsmooth.objects import AffineGaussian
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import LogMessage
from varsmooth.objects import LogNormalizer
from varsmooth.objects import LogObservation
from varsmooth.objects import LogPrior
from varsmooth.objects import LogTransition
from varsmooth.objects import ValueFn
from varsmooth.utils import bounded_while_loop
from varsmooth.utils import logdet
from varsmooth.utils import none_or_concat
from varsmooth.utils import none_or_idx
from varsmooth.utils import none_or_shift
from varsmooth.utils import symmetrize


def kl_between_marginals(p, q):
    """Return the KL divergence KL(p || q) between two Gaussian marginals.

    Args:
        p: Gaussian
            The first argument of the KL, integrated over.
        q: Gaussian
            The second, reference argument of the KL.

    Returns:
        Array
            The scalar KL divergence KL(p || q).
    """
    dim = p.mean.shape[0]
    diff = q.mean - p.mean
    # Factor q.cov once and reuse it for the trace, quadratic, and logdet terms.
    chol_q = jsc.linalg.cho_factor(q.cov)
    trace_term = jnp.trace(jsc.linalg.cho_solve(chol_q, p.cov))
    quad_term = diff.T @ jsc.linalg.cho_solve(chol_q, diff)
    logdet_q = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_q[0])))
    return 0.5 * (trace_term - dim + quad_term + logdet_q - logdet(p.cov))


@partial(jax.jit, static_argnums=(1, 2, 3))
def statistical_expansion(
    observations: Array,
    log_prior_fn: Callable,
    log_transition_fn: Callable,
    log_observation_fn: Callable,
    kernels: AffineGaussian,
    marginals: Gaussian,
) -> tuple[LogPrior, LogTransition, LogObservation]:
    """Expand the model into quadratic log-potentials around a nominal posterior.

    Evaluates the prior, transition, and observation expansion functions at the
    nominal marginals and kernels to produce the quadratic log-potentials the
    message passes consume. The prior uses the root marginal x_0, each
    transition the preceding marginals x_0..x_{T-1} paired with the kernels, and
    each observation the marginals x_1..x_T it explains.

    Args:
        observations: Array
            Observation sequence of leading shape (T,).
        log_prior_fn: Callable
            Maps the root marginal to a LogPrior.
        log_transition_fn: Callable
            Maps the preceding marginals and kernels to batched LogTransition.
        log_observation_fn: Callable
            Maps the observations and their marginals to batched LogObservation.
        kernels: AffineGaussian
            Nominal Gauss-Markov kernels of leading shape (T,) to expand around.
        marginals: Gaussian
            Nominal marginals of leading shape (T + 1,) to expand around.

    Returns:
        log_prior: LogPrior
            Quadratic log-prior over the root state.
        log_transition: LogTransition
            Batched quadratic log-transitions of leading shape (T,).
        log_observation: LogObservation
            Batched quadratic log-observations of leading shape (T,).
    """

    init_marginal = none_or_idx(marginals, 0)
    prev_marginals = none_or_shift(marginals, -1)
    next_marginals = none_or_shift(marginals, 1)

    log_prior = log_prior_fn(init_marginal)
    log_transition = log_transition_fn(prev_marginals, kernels)
    log_observation = log_observation_fn(observations, next_marginals)
    return log_prior, log_transition, log_observation


def free_energy(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    marginals: Gaussian,
    kernels: AffineGaussian,
):
    """Evaluate the variational free energy (ELBO) of a Gauss-Markov posterior.

    Sums the expected quadratic log-potentials under the posterior marginals --
    the prior E_q[log p(x_0)], the observations sum_k E_q[log p(y_k | x_k)], and
    the transitions sum_k E_q[log p(x_{k+1} | x_k)] over the pairwise marginals
    -- and adds the chain entropy H(x_0) + sum_k H(x_{k+1} | x_k).

    Args:
        log_prior: LogPrior
            Quadratic log-prior over the root state.
        log_transition: LogTransition
            Batched quadratic log-transitions of leading shape (T,).
        log_observation: LogObservation
            Batched quadratic log-observations of leading shape (T,).
        marginals: Gaussian
            Posterior marginals of leading shape (T + 1,).
        kernels: AffineGaussian
            Posterior Gauss-Markov kernels of leading shape (T,); their maps set
            the cross-covariances and their covariances the conditional entropies.

    Returns:
        Array
            The scalar variational free energy.
    """
    m, P = marginals.mean, marginals.cov
    F, _, Sigma = kernels

    def _expected_quadratic(M, v, c, mk, Pk):
        # E_{N(mk, Pk)}[ -0.5 x^T M x + v^T x + c ]
        return -0.5 * (mk @ M @ mk + jnp.trace(M @ Pk)) + v @ mk + c

    def _entropy(cov):
        return 0.5 * (cov.shape[0] * (jnp.log(2.0 * jnp.pi) + 1.0) + jnp.linalg.slogdet(cov)[1])

    # prior:  E_{q(x_0)}[log p(x_0)]
    prior_term = _expected_quadratic(log_prior.L, log_prior.l, log_prior.nu, m[0], P[0])

    # observations:  sum_k E_{q(x_k)}[log p(y_k | x_k)]
    obs_terms = jax.vmap(lambda lo, mk, Pk: _expected_quadratic(lo.L, lo.l, lo.nu, mk, Pk))(
        log_observation, m[1:], P[1:]
    )

    # transitions:  sum_k E_{q(x_k, x_{k+1})}[log p(x_{k+1} | x_k)]  over the twin marginal
    def _transition_term(lt, mk, Pk, Fk, mk1, Pk1):
        cross = Fk @ Pk  # Cov(x_{k+1}, x_k)
        joint_precision = jnp.block([[lt.C11, -lt.C12], [-lt.C21, lt.C22]])
        mean = jnp.concatenate([mk1, mk])  # z = [x_{k+1}, x_k]
        cov = jnp.block([[Pk1, cross], [cross.T, Pk]])
        linear = jnp.concatenate([lt.c1, lt.c2])
        return -0.5 * (mean @ joint_precision @ mean + jnp.trace(joint_precision @ cov)) + linear @ mean + lt.kappa

    trans_terms = jax.vmap(_transition_term)(log_transition, m[:-1], P[:-1], F, m[1:], P[1:])

    # entropy of the chain:  H(x_0) + sum_k H(x_{k+1} | x_k)
    entropy = _entropy(P[0]) + jnp.sum(jax.vmap(_entropy)(Sigma))
    return prior_term + jnp.sum(obs_terms) + jnp.sum(trans_terms) + entropy


def _log_message_pass(
    log_prior: LogPrior,
    log_transition: LogTransition,
    log_observation: LogObservation,
    reference: GaussMarkov,
    damping: float,
    reverse: bool,
) -> tuple[GaussMarkov, LogNormalizer, ValueFn, LogMessage, Array]:
    """Shared quadratic message pass for the forward and reverse smoothers.

    Both directions eliminate one block of every pairwise log-transition and
    accumulate a value function along a Gauss-Markov chain; they are mirror
    images differing only in the scan direction, in which block is eliminated,
    and in how the prior/observation boundary attaches. Parameterized by
    reverse: reverse=True is the backward pass of the forward Gauss-Markov
    smoother (log_backward_message); reverse=False is the forward pass of the
    reverse Gauss-Markov smoother (log_forward_message).

    Args:
        log_prior: LogPrior
            Quadratic log-prior over the boundary state.
        log_transition: LogTransition
            Batched pairwise quadratic log-transitions of leading shape (T,).
        log_observation: LogObservation
            Batched quadratic log-observations of leading shape (T,).
        reference: GaussMarkov
            The Gauss-Markov posterior to expand around; its kernels supply the
            nominal linearization and its marginal the boundary.
        damping: float
            Trust-region damping in [0, 1); damping = t / (1 + t) for
            temperature t.
        reverse: bool
            Direction selector; see above.

    Returns:
        posterior: GaussMarkov
            The updated Gauss-Markov posterior (boundary marginal + kernels).
        log_marg_norm: LogNormalizer
            The quadratic marginal log-normalizer at the boundary.
        value_fns: ValueFn
            The per-marginal value functions of leading shape (T + 1,).
        log_msgs: LogMessage
            The per-step eliminated-variable messages of leading shape (T,).
        feasible_flags: Array
            Boolean array of shape (T,) marking feasible steps.
    """

    def _step(carry, args):
        R, r, rho = carry
        C11, C12, C21, C22, c1, c2, kappa, L, l, nu, F, d, Sigma = args

        if reverse:
            # eliminate block 1 (x_{k+1}); the retained block is 2 (x_k)
            Cee, Coo, Ceo, ce, co = C11, C22, C12, c1, c2
        else:
            # eliminate block 2 (x_k); the retained block is 1 (x_{k+1})
            Cee, Coo, Ceo, ce, co = C22, C11, C21, c2, c1

        dim = Sigma.shape[0]
        # Factor the kernel covariance once and reuse it for every Sigma-solve.
        chol_Sigma = jsc.linalg.cho_factor(Sigma)
        iSig = jsc.linalg.cho_solve(chol_Sigma, jnp.eye(dim))
        iSig_F = jsc.linalg.cho_solve(chol_Sigma, F)
        iSig_d = jsc.linalg.cho_solve(chol_Sigma, d)
        logdet_Sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_Sigma[0])))

        Gee = (1.0 - damping) * (Cee + R) + damping * iSig
        Goo = (1.0 - damping) * Coo + damping * F.T @ iSig_F
        Geo = (1.0 - damping) * Ceo + damping * iSig_F
        ge = (1.0 - damping) * (ce + r) + damping * iSig_d
        go = (1.0 - damping) * co - damping * F.T @ iSig_d
        theta = (
            (1.0 - damping) * (kappa + rho)
            - 0.5 * damping * (dim * jnp.log(2 * jnp.pi) + logdet_Sigma)
            - 0.5 * damping * d.T @ iSig_d
        )

        Gee = symmetrize(Gee)
        Goo = symmetrize(Goo)

        # Feasibility via a Cholesky attempt: non-PD Gee yields a non-finite factor.
        chol_Gee = jnp.linalg.cholesky(Gee)
        pd_Gee = jnp.all(jnp.isfinite(chol_Gee))

        def _feasible():
            iGee_Geo = jsc.linalg.cho_solve((chol_Gee, True), Geo)
            iGee_ge = jsc.linalg.cho_solve((chol_Gee, True), ge)
            post_Sigma = jsc.linalg.cho_solve((chol_Gee, True), jnp.eye(dim))
            logdet_Gee = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_Gee)))

            S = Goo - Geo.T @ iGee_Geo
            s = go + Geo.T @ iGee_ge
            xi = theta + 0.5 * (dim * jnp.log(2 * jnp.pi) - logdet_Gee) + 0.5 * ge.T @ iGee_ge

            new_R = L + 1.0 / (1.0 - damping) * S
            new_r = l + 1.0 / (1.0 - damping) * s
            new_rho = nu + 1.0 / (1.0 - damping) * xi

            value_fn = ValueFn(new_R, new_r, new_rho)
            return value_fn, (value_fn, AffineGaussian(iGee_Geo, iGee_ge, post_Sigma), LogMessage(S, s, xi), True)

        def _not_feasible():
            S = jnp.zeros_like(Goo)
            s = jnp.zeros_like(go)
            xi = jnp.zeros_like(theta)

            new_R = jnp.zeros_like(L)
            new_r = jnp.zeros_like(l)
            new_rho = jnp.zeros_like(nu)

            value_fn = ValueFn(new_R, new_r, new_rho)
            return value_fn, (value_fn, AffineGaussian(F, d, Sigma), LogMessage(S, s, xi), False)

        return jax.lax.cond(pred=pd_Gee, true_fun=_feasible, false_fun=_not_feasible)

    if reverse:
        # backward pass: the last observation seeds the recursion, the prior
        # enters as the root pseudo-observation, and observations shift down.
        last_log_obs = none_or_idx(log_observation, -1)
        boundary_value_fn = ValueFn(R=last_log_obs.L, r=last_log_obs.l, rho=last_log_obs.nu)
        obs_stream = none_or_concat(
            none_or_shift(log_observation, -1),
            LogObservation(log_prior.L, log_prior.l, log_prior.nu),
            1,
        )
        concat_position = -1
    else:
        # forward pass: the prior seeds the recursion and observations feed in order.
        boundary_value_fn = ValueFn(R=log_prior.L, r=log_prior.l, rho=log_prior.nu)
        obs_stream = log_observation
        concat_position = 1

    nominal_marginal, nominal_kernels = reference

    boundary_out, (value_fns, kernels, log_msgs, feasible_flags) = jax.lax.scan(
        f=_step,
        init=boundary_value_fn,
        xs=(*log_transition, *obs_stream, *nominal_kernels),
        reverse=reverse,
    )
    value_fns = none_or_concat(value_fns, boundary_value_fn, concat_position)

    R, r, rho = boundary_out

    m, P = nominal_marginal
    dim = P.shape[0]
    chol_P = jsc.linalg.cho_factor(P)
    inv_P = jsc.linalg.cho_solve(chol_P, jnp.eye(dim))
    logdet_P = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_P[0])))

    Jee = symmetrize((1.0 - damping) * R + damping * inv_P)
    Jeo = damping * inv_P
    Joo = symmetrize(damping * inv_P)
    je = (1.0 - damping) * r
    jo = jnp.zeros_like(je)
    tau = (1.0 - damping) * rho - 0.5 * damping * (dim * jnp.log(2 * jnp.pi) + logdet_P)

    def _feasible_marginal():
        chol_Jee = jsc.linalg.cho_factor(Jee)
        iJee_Jeo = jsc.linalg.cho_solve(chol_Jee, Jeo)
        iJee_je = jsc.linalg.cho_solve(chol_Jee, je)
        post_P = jsc.linalg.cho_solve(chol_Jee, jnp.eye(dim))
        logdet_Jee = 2.0 * jnp.sum(jnp.log(jnp.diag(chol_Jee[0])))

        # boundary marginal
        post_m = jsc.linalg.cho_solve(chol_Jee, je + Jeo @ m)

        # log normalizer
        U = Joo - Jeo.T @ iJee_Jeo
        u = jo - Jeo.T @ iJee_je
        eta = tau + 0.5 * (dim * jnp.log(2 * jnp.pi) - logdet_Jee) + 0.5 * je.T @ iJee_je
        return Gaussian(post_m, post_P), LogNormalizer(U, u, eta)

    def _not_feasible_marginal():
        post_m = jnp.zeros_like(nominal_marginal.mean)
        post_P = jnp.zeros_like(nominal_marginal.cov)

        U = jnp.zeros_like(Joo)
        u = jnp.zeros_like(jo)
        eta = jnp.zeros_like(tau)
        return Gaussian(post_m, post_P), LogNormalizer(U, u, eta)

    marginal, log_marg_norm = jax.lax.cond(
        pred=jnp.all(feasible_flags),
        true_fun=_feasible_marginal,
        false_fun=_not_feasible_marginal,
    )
    return (GaussMarkov(marginal, kernels), log_marg_norm, value_fns, log_msgs, feasible_flags)


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


def initialize_reverse_with_forward(forward_markov: GaussMarkov) -> GaussMarkov:
    """Re-express a forward Gauss-Markov chain as an equivalent reverse chain.

    Propagates the forward marginals and inverts each forward kernel into a
    reverse kernel q(x_k | x_{k+1}), yielding a reverse Gauss-Markov chain rooted
    at the leaf marginal x_T that encodes the same joint distribution.

    Args:
        forward_markov: GaussMarkov
            The forward chain (root marginal x_0 + forward kernels).

    Returns:
        GaussMarkov
            The equivalent reverse chain (leaf marginal x_T + reverse kernels).
    """
    forward_marginals = std_forward_message(forward_markov)

    # reverse kernels q(x_k | x_{k+1}) from forward marginals + forward kernels
    kernels = jax.vmap(get_reverse_kernel)(
        none_or_shift(forward_marginals, -1),  # marginals 0 .. T-1
        forward_markov.kernels,  # forward kernels k+1 | k
        none_or_shift(forward_marginals, 1),  # marginals 1 .. T
    )

    return GaussMarkov(
        marginal=Gaussian(
            mean=forward_marginals.mean[-1],
            cov=forward_marginals.cov[-1],
        ),
        kernels=kernels,
    )


def get_marginal(marginal: Gaussian, kernel: AffineGaussian):
    """Push a Gaussian marginal through an affine-Gaussian kernel.

    Args:
        marginal: Gaussian
            The input marginal N(m, P).
        kernel: AffineGaussian
            The kernel x -> N(F x + d, Sigma) to propagate through.

    Returns:
        Gaussian
            The output marginal N(F m + d, F P F^T + Sigma).
    """
    m, P = marginal
    F, d, Sigma = kernel
    return Gaussian(mean=F @ m + d, cov=F @ P @ F.T + Sigma)


def get_pairwise_marginal(marginal: Gaussian, kernel: AffineGaussian):
    """Form the joint Gaussian over the pair z = (x_{k+1}, x_k).

    Combines a marginal over x_k with a kernel x_k -> N(F x_k + d, Sigma) into
    the joint over the stacked pair, ordered with the next state first.

    Args:
        marginal: Gaussian
            The marginal N(m, P) over x_k.
        kernel: AffineGaussian
            The kernel x_k -> N(F x_k + d, Sigma) to the next state x_{k+1}.

    Returns:
        Gaussian
            The joint over z = (x_{k+1}, x_k) of dimension 2 dx.
    """
    m, P = marginal
    F, d, Sigma = kernel

    q = Gaussian(
        mean=jnp.hstack((F @ m + d, m)),
        cov=jnp.vstack((jnp.hstack((F @ P @ F.T + Sigma, F @ P)), jnp.hstack((P.T @ F.T, P)))),
    )
    return q


def get_conditional(marginal: Gaussian, pairwise: Gaussian):
    """Extract the affine-Gaussian conditional of the second block given the first.

    Splits the pairwise joint into blocks at the dimension of marginal and
    returns the kernel u -> N(F u + d, Sigma) mapping the first (conditioning)
    block u to the second block by Gaussian conditioning.

    Args:
        marginal: Gaussian
            Marginal over the first (conditioning) block; only its dimension is
            used to locate the block split.
        pairwise: Gaussian
            The joint over the stacked pair (first block, second block).

    Returns:
        AffineGaussian
            The conditional kernel of the second block given the first.
    """
    dim = marginal.mean.shape[0]

    a = pairwise.mean[:dim]
    b = pairwise.mean[dim:]

    A = pairwise.cov[:dim, :dim]
    B = pairwise.cov[dim:, dim:]
    C = pairwise.cov[:dim, dim:]

    return AffineGaussian(
        F=jsc.linalg.solve(A, C).T,
        d=b - C.T @ jsc.linalg.solve(A, a),
        Sigma=B - C.T @ jsc.linalg.solve(A, C),
    )


def get_reverse_kernel(
    marginal: Gaussian,
    kernel: AffineGaussian,
    next_marginal: Gaussian,
):
    """Invert a forward kernel into the reverse kernel q(x_k | x_{k+1}).

    Forms the pairwise joint over (x_{k+1}, x_k) from the marginal over x_k and
    the forward kernel, then conditions on x_{k+1} to obtain the reverse kernel.

    Args:
        marginal: Gaussian
            The marginal over x_k.
        kernel: AffineGaussian
            The forward kernel x_k -> N(F x_k + d, Sigma) to x_{k+1}.
        next_marginal: Gaussian
            The marginal over x_{k+1} to condition on.

    Returns:
        AffineGaussian
            The reverse kernel giving q(x_k | x_{k+1}).
    """
    pairwise = get_pairwise_marginal(marginal, kernel)
    return get_conditional(next_marginal, pairwise)


def merge_messages(fwd_message: ValueFn, bwd_message: LogMessage) -> ValueFn:
    """Combine a forward value function and a backward message into one potential.

    Adds the matching quadratic components (R + S, r + s, rho + xi) of the two
    incoming messages.

    Args:
        fwd_message: ValueFn
            The forward value function (R, r, rho).
        bwd_message: LogMessage
            The backward eliminated-variable message (S, s, xi).

    Returns:
        ValueFn
            The merged quadratic potential.
    """
    return ValueFn(
        R=(fwd_message.R + bwd_message.S),
        r=(fwd_message.r + bwd_message.s),
        rho=(fwd_message.rho + bwd_message.xi),
    )


def log_to_std_form(potential: ValueFn) -> Gaussian:
    """Convert a quadratic log-potential to a moment-form Gaussian.

    Maps the information-form potential (R, r, rho) to the Gaussian with
    covariance R^{-1} and mean R^{-1} r; the constant rho is dropped.

    Args:
        potential: ValueFn
            The quadratic log-potential (R, r, rho) with R the precision.

    Returns:
        Gaussian
            The moment-form Gaussian N(R^{-1} r, R^{-1}).
    """
    return Gaussian(mean=jsc.linalg.inv(potential.R) @ potential.r, cov=jsc.linalg.inv(potential.R))


def std_to_log_form(dist: Gaussian) -> ValueFn:
    """Convert a moment-form Gaussian to a normalized quadratic log-potential.

    Maps the Gaussian N(mean, cov) to the information-form potential with
    precision R = cov^{-1}, linear term r = cov^{-1} mean, and constant rho set
    so the quadratic equals the Gaussian log density.

    Args:
        dist: Gaussian
            The moment-form Gaussian to convert.

    Returns:
        ValueFn
            The quadratic log-potential (R, r, rho).
    """
    return ValueFn(
        R=jsc.linalg.inv(dist.cov),
        r=jsc.linalg.solve(dist.cov, dist.mean),
        rho=(-0.5 * logdet(2 * jnp.pi * dist.cov) - 0.5 * dist.mean.T @ jsc.linalg.solve(dist.cov, dist.mean)),
    )


def _kl_between_gauss_markovs(
    marginals: Gaussian,
    gauss_markov: GaussMarkov,
    ref_gauss_markov: GaussMarkov,
    reverse: bool = False,
):
    """Accumulate the KL divergence between two Gauss-Markov chains.

    Returns KL(gauss_markov || ref_gauss_markov): the KL between the boundary
    marginals plus the expected per-kernel KL contributions summed along the
    chain, with the kernel expectations taken under marginals.

    Args:
        marginals: Gaussian
            Marginals of leading shape (T + 1,) supplying the moments the
            per-kernel terms are averaged over.
        gauss_markov: GaussMarkov
            The first chain, integrated over.
        ref_gauss_markov: GaussMarkov
            The reference chain.
        reverse: bool
            Scan direction; True walks the chain backward to match a reverse
            Gauss-Markov chain, False walks it forward.

    Returns:
        Array
            The scalar KL divergence between the two chains.
    """
    dim = gauss_markov.marginal.mean.shape[0]

    def body(carry, args):
        kl_value = carry
        m, P, F, d, Sigma, ref_F, ref_d, ref_Sigma = args

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
            + 0.5 * logdet(ref_Sigma)
            - 0.5 * logdet(Sigma)
        )
        return kl_value, None

    init_kl_value = kl_between_marginals(gauss_markov.marginal, ref_gauss_markov.marginal)

    kl_value, _ = jax.lax.scan(
        f=body,
        init=init_kl_value,
        xs=(*none_or_shift(marginals, 1), *gauss_markov.kernels, *ref_gauss_markov.kernels),
        reverse=reverse,
    )
    return kl_value


def kl_between_reverse_gauss_markovs(marginals, gauss_markov, ref_gauss_markov):
    """Return the KL divergence between two reverse Gauss-Markov chains.

    Thin wrapper over _kl_between_gauss_markovs with a backward scan.

    Args:
        marginals: Gaussian
            Marginals of leading shape (T + 1,) the per-kernel terms average over.
        gauss_markov: GaussMarkov
            The first reverse chain, integrated over.
        ref_gauss_markov: GaussMarkov
            The reference reverse chain.

    Returns:
        Array
            The scalar KL divergence KL(gauss_markov || ref_gauss_markov).
    """
    return _kl_between_gauss_markovs(marginals, gauss_markov, ref_gauss_markov, True)


def kl_between_forward_gauss_markovs(marginals, gauss_markov, ref_gauss_markov):
    """Return the KL divergence between two forward Gauss-Markov chains.

    Thin wrapper over _kl_between_gauss_markovs with a forward scan.

    Args:
        marginals: Gaussian
            Marginals of leading shape (T + 1,) the per-kernel terms average over.
        gauss_markov: GaussMarkov
            The first forward chain, integrated over.
        ref_gauss_markov: GaussMarkov
            The reference forward chain.

    Returns:
        Array
            The scalar KL divergence KL(gauss_markov || ref_gauss_markov).
    """
    return _kl_between_gauss_markovs(marginals, gauss_markov, ref_gauss_markov, False)


def _run_iterations(
    single_iteration_fn: Callable,
    init_state,
    init_temperature: float,
    min_temperature: float,
    max_iterations: int,
    return_history: bool,
):
    """Drive an iterated smoother to convergence, shared across directions.

    Runs single_iteration_fn from init_state until the line-search temperature
    drops to min_temperature or max_iterations is reached. With
    return_history=True the iterations run under a fixed-length scan that stacks
    per-iteration diagnostics; otherwise they run under a temperature-gated
    bounded while loop.

    Args:
        single_iteration_fn: Callable
            Map (state, iteration_idx) -> (next_state, temperature, diagnostics).
        init_state: Any
            Initial loop-carried state (direction-specific pytree).
        init_temperature: float
            Initial line-search temperature.
        min_temperature: float
            Early-stopping threshold on the temperature.
        max_iterations: int
            Maximum number of iterations (static int).
        return_history: bool
            Whether to scan a fixed number of iterations and stack diagnostics.

    Returns:
        final_state: Any
            The final loop-carried state.
        history: Any
            Stacked per-iteration diagnostics if return_history else None.
    """
    if return_history:

        def scan_step(state, iteration_idx):
            next_state, _temperature, diagnostics = single_iteration_fn(state, iteration_idx)
            return next_state, diagnostics

        final_state, history = jax.lax.scan(scan_step, init_state, xs=jnp.arange(max_iterations))
        return final_state, history

    def iteration_body(carry):
        current_state, iteration_count, _ = carry
        next_state, next_temperature, _ = single_iteration_fn(current_state, iteration_count)
        return next_state, iteration_count + 1, next_temperature

    def iteration_condition(carry):
        _, iteration_count, next_temperature = carry
        return jnp.logical_and(iteration_count < max_iterations, next_temperature > min_temperature)

    final_state, _, _ = bounded_while_loop(
        cond_fun=iteration_condition,
        body_fun=iteration_body,
        init_val=(init_state, 0, init_temperature),
        maxiter=max_iterations,
    )
    return final_state, None


class ParamStruct(NamedTuple):
    """Line-search temperature with its current bracketing interval.

    Attributes:
        val: float
            The current temperature.
        min: float
            Lower end of the bracketing interval.
        max: float
            Upper end of the bracketing interval.
    """

    val: float
    min: float
    max: float


class LineSearchState(NamedTuple):
    """Best-so-far state carried through the line search.

    Attributes:
        param: ParamStruct
            The temperature (and its current bracket) achieving the smallest
            slack magnitude seen so far.
        fn_val: float
            Dual objective value at param.
        slack: float
            Constraint slack kl_constraint - realized_KL at param.
        feasible: bool
            Whether any feasible temperature has been accepted.
    """

    param: ParamStruct
    fn_val: float
    slack: float
    feasible: bool


def line_search(
    init_param: float,
    fun: Callable,
    grad: Callable,
    rtol=0.1,
    min_param=1e-14,
    max_param=1e14,
    max_iter=100,
) -> tuple[float, float, float, bool]:
    """Bracket a temperature that drives the constraint slack to (near) zero.

    Each iteration evaluates the dual objective and the constraint slack once
    at the current temperature and bisects the bracket in log-space. The slack
    sign selects the direction: positive slack means the realized KL is below
    the constraint (the step is too conservative), so the temperature is
    reduced; negative slack means the trust region is violated, so the
    temperature is increased. The feasible point with the smallest |slack| seen
    is retained and returned.

    Args:
        init_param: float
            Initial temperature at which the search starts.
        fun: Callable
            Dual objective temperature -> value; one full message pass.
        grad: Callable
            Constraint slack temperature -> (kl_constraint - realized_KL); one
            full message pass. Returns +inf on an infeasible step.
        rtol: float
            Absolute tolerance on |slack| for termination.
        min_param: float
            Lower bound of the temperature bracket.
        max_param: float
            Upper bound of the temperature bracket.
        max_iter: int
            Maximum number of bisection iterations.

    Returns:
        param: float
            The accepted temperature.
        fn_val: float
            The dual objective at param.
        slack: float
            The constraint slack at param.
        feasible: bool
            Whether a feasible temperature was accepted.
    """

    init_param_struct = ParamStruct(val=init_param, min=min_param, max=max_param)
    state = LineSearchState(
        param=init_param_struct,
        fn_val=jnp.inf,
        slack=jnp.inf,
        feasible=False,
    )

    def regularize(args):
        param, state, _fn_val, _slack = args
        return increase_param(param), state

    def update(args):
        param, state, fn_val, slack = args

        state = jax.lax.cond(
            jnp.abs(slack) < jnp.abs(state.slack),
            lambda _: LineSearchState(param, fn_val, slack, True),
            lambda _: state,
            None,
        )

        param = jax.lax.cond(
            pred=slack > 0.0,
            true_fun=reduce_param,
            false_fun=increase_param,
            operand=param,
        )
        return param, state

    def _iteration(carry):
        param, state = carry

        fn_val = fun(param.val)
        slack = grad(param.val)

        nan_condition = jnp.logical_or(jnp.isnan(fn_val), jnp.isnan(slack))
        inf_condition = jnp.logical_or(jnp.isinf(fn_val), jnp.isinf(slack))

        return jax.lax.cond(
            pred=jnp.logical_or(nan_condition, inf_condition),
            true_fun=regularize,
            false_fun=update,
            operand=(param, state, fn_val, slack),
        )

    _, state = bounded_while_loop(
        cond_fun=lambda x: jnp.abs(x[-1].slack) > rtol,
        body_fun=_iteration,
        init_val=(init_param_struct, state),
        maxiter=max_iter,
    )
    return state.param.val, state.fn_val, state.slack, state.feasible


def reduce_param(param) -> ParamStruct:
    """Bisect the temperature bracket downward toward its lower bound.

    Sets the new temperature to the geometric mean of the lower bound and the
    current value, moving the upper bound down to the current value.

    Args:
        param: ParamStruct
            The current temperature and its bracket.

    Returns:
        ParamStruct
            The updated temperature and narrowed bracket.
    """
    # set max to current value
    return ParamStruct(val=jnp.sqrt(param.min * param.val), min=param.min, max=param.val)


def increase_param(param) -> ParamStruct:
    """Bisect the temperature bracket upward toward its upper bound.

    Sets the new temperature to the geometric mean of the current value and the
    upper bound, moving the lower bound up to the current value.

    Args:
        param: ParamStruct
            The current temperature and its bracket.

    Returns:
        ParamStruct
            The updated temperature and narrowed bracket.
    """
    # set min to current value
    return ParamStruct(val=jnp.sqrt(param.val * param.max), min=param.val, max=param.max)


def sample_from_forward_markov(
    rng_key: Array,
    gauss_markov: GaussMarkov,
    num_samples: int,
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
            next_state = jax.random.multivariate_normal(sample_key, conditional_mean, Sigma_t)

            return (next_key, next_state), next_state

        # Sample initial state
        init_key, traj_key = jax.random.split(sample_key)
        initial_state = jax.random.multivariate_normal(init_key, marginal.mean, marginal.cov)

        # Sample trajectory using conditional posteriors
        _, trajectory = jax.lax.scan(sample_step, (traj_key, initial_state), (kernels.F, kernels.d, kernels.Sigma))

        # Prepend initial state to trajectory
        def concat_trees(x, y):
            return jax.tree.map(lambda x, y: jnp.concatenate([x[None, ...], y]), x, y)

        trajectory = concat_trees(initial_state, trajectory)
        return trajectory

    # Generate multiple trajectories
    sample_keys = jax.random.split(rng_key, num_samples)
    trajectories = jax.vmap(sample_single_trajectory)(sample_keys)

    return trajectories
