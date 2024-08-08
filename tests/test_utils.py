import numpy as np

from varsmooth.objects import Gaussian


def generate_system(dim_x, dim_y):
    m = np.random.randn(dim_x)
    chol_P = np.random.rand(dim_x, dim_x)
    chol_P[np.triu_indices(dim_x, 1)] = 0.
    P = chol_P @ chol_P.T

    chol_Sigma = np.random.rand(dim_y, dim_y)
    chol_Sigma[np.triu_indices(dim_y, 1)] = 0.
    Sigma = chol_Sigma @ chol_Sigma.T

    A = 0.9 * np.eye(dim_y, dim_x)
    b = np.random.randn(dim_y)
    xs = np.random.randn(dim_y)

    q = Gaussian(m, P)
    return q, A, b, Sigma, xs
