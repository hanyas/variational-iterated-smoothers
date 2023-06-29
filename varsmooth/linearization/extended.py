from typing import Any, Tuple, Union

import jax

from varsmooth.objects import StdGaussian, FunctionalModel, ConditionalMomentsModel


def linearize(model: Union[FunctionalModel, ConditionalMomentsModel], x: StdGaussian):

    if isinstance(model, FunctionalModel):
        f, q = model
        m_x, _ = x
        return _standard_linearize_functional(f, m_x, *q)
    else:
        return _standard_linearize_conditional(model.mean_func, model.cov_func, x)


def _standard_linearize_conditional(c_m, c_cov, x):
    m, p = x
    F = jax.jacfwd(c_m, 0)(m)
    b = c_m(m) - F @ m
    Q = c_cov(m)
    return F, b, Q


def _linearize_callable_common(f, x) -> Tuple[Any, Any]:
    return f(x), jax.jacfwd(f, 0)(x)


def _standard_linearize_functional(f, x, m_q, cov_q):
    res, F_x = _linearize_callable_common(f, x)
    return F_x, res - F_x @ x + m_q, cov_q
