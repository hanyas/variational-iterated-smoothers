import pytest
from functools import partial

import jax
import numpy as np

from varsmooth.objects import StdGaussian, StdLinearGaussian, ForwardGaussMarkov
from varsmooth.linearization import cubature, extended

from tests.kalman import rts_smoother
from varsmooth.smoothers.forward_markov import _var_smoother

from tests.lgssm import get_data
from tests.test_utils import get_system

LIST_LINEARIZATIONS = [cubature, extended]


@pytest.fixture(scope="session", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update('jax_disable_jit', False)
    jax.config.update("jax_debug_nans", False)


@pytest.mark.parametrize("nx", [1, 3])
@pytest.mark.parametrize("ny", [2, 3])
@pytest.mark.parametrize("seed", [0, 42])
def test_rts_smoother_vs_var_smoother(nx, ny, seed):
    np.random.seed(seed)

    T = 10

    x0, chol_x0, F, b, Q, cholQ, _ = get_system(nx, nx)
    _, _, H, c, R, cholR, _ = get_system(nx, ny)

    xs, ys = get_data(x0.mean, F, H, R, Q, b, c, T)

    Fs = np.repeat([F], T, axis=0)
    bs = np.repeat([b], T, axis=0)
    Qs = np.repeat([Q], T, axis=0)
    trans_mdl = StdLinearGaussian(Fs, bs, Qs)

    Hs = np.repeat([H], T, axis=0)
    cs = np.repeat([c], T, axis=0)
    Rs = np.repeat([R], T, axis=0)
    obsrv_mdl = StdLinearGaussian(Hs, cs, Rs)

    rts_smoothed = rts_smoother(ys, x0, trans_mdl, obsrv_mdl)

    init_posterior = ForwardGaussMarkov(
        init=StdGaussian(np.zeros((nx,)), 10.0 * np.eye(nx)),
        kernels=StdLinearGaussian(
            mat=np.zeros((T, nx, nx)),
            bias=np.zeros((T, nx)),
            cov=np.repeat(10.0 * np.eye(nx).reshape(1, nx, nx), T, axis=0),
        )
    )

    var_smoothed = _var_smoother(ys, x0, trans_mdl, obsrv_mdl, init_posterior, 0.0)

    np.testing.assert_allclose(rts_smoothed.mean, var_smoothed.mean, atol=1e-5)
    np.testing.assert_allclose(rts_smoothed.cov, var_smoothed.cov, atol=1e-3)
