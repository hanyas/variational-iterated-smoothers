"""Reference environments for the variational smoothers.

Each environment module exposes the same small surface as examples/bearing_model.py:
a make_parameters(...) that returns the model functions together with their noise
covariances and Jacobians, and a get_data(...) that simulates a trajectory and its
observations. They give experiments and examples a single, uniform way to instantiate a
state-space model.
"""

from varsmooth.environments import bearing_only
from varsmooth.environments import cubic_sensor
from varsmooth.environments import linear_gaussian
from varsmooth.environments import stoch_volatility

__all__ = ["bearing_only", "cubic_sensor", "linear_gaussian", "stoch_volatility"]
