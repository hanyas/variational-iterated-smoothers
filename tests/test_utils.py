import numpy as np

from varsmooth.objects import Gaussian


def generate_system(dim_x, dim_y):
    """Draw a random linear-Gaussian system for testing.

    Args:
        dim_x: int
            State dimension.
        dim_y: int
            Observation dimension.

    Returns:
        q: Gaussian
            Random Gaussian prior with mean of shape (dim_x,) and covariance
            of shape (dim_x, dim_x).
        A: np.ndarray
            Linear map of shape (dim_y, dim_x) with 0.9 on its diagonal.
        b: np.ndarray
            Offset vector of shape (dim_y,).
        Sigma: np.ndarray
            Random covariance matrix of shape (dim_y, dim_y).
        xs: np.ndarray
            Random evaluation point of shape (dim_y,).
    """
    m = np.random.randn(dim_x)
    chol_P = np.random.rand(dim_x, dim_x)
    chol_P[np.triu_indices(dim_x, 1)] = 0.0
    P = chol_P @ chol_P.T

    chol_Sigma = np.random.rand(dim_y, dim_y)
    chol_Sigma[np.triu_indices(dim_y, 1)] = 0.0
    Sigma = chol_Sigma @ chol_Sigma.T

    A = 0.9 * np.eye(dim_y, dim_x)
    b = np.random.randn(dim_y)
    xs = np.random.randn(dim_y)

    q = Gaussian(m, P)
    return q, A, b, Sigma, xs
