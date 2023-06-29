import numpy as np

from varsmooth.objects import StdGaussian, SqrtGaussian


def get_system(dim_x, dim_y):
    m = np.random.randn(dim_x)
    cholP = np.random.rand(dim_x, dim_x)
    cholP[np.triu_indices(dim_x, 1)] = 0.
    P = cholP @ cholP.T

    cholR = np.random.rand(dim_y, dim_y)
    cholR[np.triu_indices(dim_y, 1)] = 0.
    R = cholR @ cholR.T

    H = np.eye(dim_y, dim_x)
    c = np.random.randn(dim_y)
    y = np.random.randn(dim_y)

    chol_x = SqrtGaussian(m, cholP)
    x = StdGaussian(m, P)
    return x, chol_x, H, c, R, cholR, y
