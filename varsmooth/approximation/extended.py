import jax

from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel
from varsmooth.objects import Gaussian


def quadratize(fun, q):
    """Not implemented: the extended scheme has no analytic quadratization.

    Raises:
        NotImplementedError
            Always; use the cubature or gauss_hermite quadratize instead.
    """
    raise NotImplementedError(
        "analytic quadratization not implemented; use the cubature or "
        "gauss_hermite quadratize instead."
    )


def linearize(model: AdditiveGaussianModel | ConditionalMomentsModel, q: Gaussian):
    """Statistically linearize a model under q via first-order Taylor (Jacobian)."""
    if isinstance(model, AdditiveGaussianModel):
        m_x, _ = q
        fun, noise = model
        return linearize_additive(fun, noise, m_x)
    elif isinstance(model, ConditionalMomentsModel):
        mean_fn, cov_fn = model
        return linearize_conditional(mean_fn, cov_fn, q)
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")


def linearize_additive(fun, noise, x):
    """Linearize y = fun(x) + noise by the Jacobian of fun at x."""
    f0, F = fun(x), jax.jacfwd(fun, 0)(x)
    return F, f0 - F @ x + noise.mean, noise.cov


def linearize_conditional(mean_fn, cov_fn, q):
    """Linearize a conditional-moments model by the Jacobian of mean_fn at the mean of q."""
    m_x, cov_x = q
    F = jax.jacfwd(mean_fn, 0)(m_x)
    b = mean_fn(m_x) - F @ m_x
    Sigma = cov_fn(m_x)
    return F, b, Sigma
