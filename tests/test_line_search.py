import jax
import jax.numpy as jnp
import numpy as np
import pytest

from varsmooth.smoothers.utils import line_search
from varsmooth.utils import bounded_while_loop


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")


def test_bounded_while_loop_hits_maxiter():
    # cond stays true forever: the loop must stop after exactly maxiter steps.
    maxiter = 5
    val = bounded_while_loop(
        cond_fun=lambda v: jnp.array(True),
        body_fun=lambda v: v + 1,
        init_val=0,
        maxiter=maxiter,
    )
    assert int(val) == maxiter


def test_bounded_while_loop_early_termination():
    # cond turns false at v == 3: the loop must return that value, not maxiter.
    val = bounded_while_loop(
        cond_fun=lambda v: v < 3,
        body_fun=lambda v: v + 1,
        init_val=0,
        maxiter=100,
    )
    assert int(val) == 3


def test_line_search_converges_to_root():
    # Monotone scalar problem: slack(t) = c - 1/t has a single root at t = 1/c.
    # slack is increasing in t, matching line_search's reduce-on-positive rule.
    c = 0.5
    root = 1.0 / c
    rtol = 1e-3

    fun = lambda t: 0.0  # dual value is irrelevant to convergence here
    slack_fn = lambda t: c - 1.0 / t

    param, fn_val, slack, feasible = line_search(
        init_param=1.0, fun=fun, grad=slack_fn, rtol=rtol, min_param=1e-14, max_param=1e14
    )

    assert bool(feasible)
    assert abs(float(slack)) <= rtol
    np.testing.assert_allclose(float(param), root, atol=0.05)


def test_line_search_infeasible_on_nan():
    # fun / slack are NaN everywhere: no feasible point is ever accepted.
    param, fn_val, slack, feasible = line_search(
        init_param=1.0,
        fun=lambda t: jnp.nan,
        grad=lambda t: jnp.nan,
        rtol=1e-3,
    )
    assert not bool(feasible)


def test_line_search_respects_bracket_bounds():
    # The root (t = 1/c = 2) lies above max_param, so the search saturates at
    # the upper bound without ever leaving [min_param, max_param].
    c = 0.5
    min_param, max_param = 1e-6, 1.5
    param, fn_val, slack, feasible = line_search(
        init_param=1.0,
        fun=lambda t: 0.0,
        grad=lambda t: c - 1.0 / t,
        rtol=1e-3,
        min_param=min_param,
        max_param=max_param,
    )
    assert min_param <= float(param) <= max_param
