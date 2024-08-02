from typing import Union
import jax

from varsmooth.objects import Gaussian
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import ConditionalMomentsModel


def linearize(
    model: Union[AdditiveGaussianModel, ConditionalMomentsModel],
    q: Gaussian
):
    if isinstance(model, AdditiveGaussianModel):
        m_x, _ = q
        fun, noise = model
        return linearize_additive(fun, noise, m_x)
    elif isinstance(model, ConditionalMomentsModel):
        mean_fn, cov_fn = model
        return linearize_conditional(mean_fn, cov_fn, q)
    else:
        raise NotImplementedError


def linearize_additive(fun, noise, x):
    f0, F = fun(x), jax.jacfwd(fun, 0)(x)
    return F, f0 - F @ x + noise.mean, noise.cov


def linearize_conditional(mean_fn, cov_fn, q):
    m_x, cov_x = q
    F = jax.jacfwd(mean_fn, 0)(m_x)
    b = mean_fn(m_x) - F @ m_x
    Sigma = cov_fn(m_x)
    return F, b, Sigma
