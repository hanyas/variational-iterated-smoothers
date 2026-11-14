"""Reference environments for the variational smoothers."""

from varsmooth.environments import bearing_only
from varsmooth.environments import cubic_sensor
from varsmooth.environments import differential_drive
from varsmooth.environments import linear_gaussian
from varsmooth.environments import stoch_volatility

__all__ = ["bearing_only", "cubic_sensor", "differential_drive", "linear_gaussian", "stoch_volatility"]
