import numpy as np

from varsmooth.objects import Gaussian


def generate_system(dim_x, dim_y):
    m = np.random.randn(dim_x)
    chol_P = np.random.rand(dim_x, dim_x)
    chol_P[np.triu_indices(dim_x, 1)] = 0.
    P = chol_P @ chol_P.T

    chol_covar = np.random.rand(dim_y, dim_y)
    chol_covar[np.triu_indices(dim_y, 1)] = 0.
    covar = chol_covar @ chol_covar.T

    mat = np.eye(dim_y, dim_x)
    offset = np.random.randn(dim_y)
    vecs = np.random.randn(dim_y)

    q = Gaussian(m, P)
    return q, mat, offset, covar, vecs
