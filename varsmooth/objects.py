import itertools
from typing import NamedTuple, Callable

import jax.numpy as jnp


class StdGaussian(NamedTuple):
    mean: jnp.ndarray
    cov: jnp.ndarray


class SqrtGaussian(NamedTuple):
    mean: jnp.ndarray
    cov_chol: jnp.ndarray


class FunctionalModel(NamedTuple):
    func: Callable
    mvn: StdGaussian


class ConditionalMomentsModel(NamedTuple):
    mean_func: Callable
    cov_func: Callable


class StdLinearGaussian(NamedTuple):
    mat: jnp.ndarray
    bias: jnp.ndarray
    cov: jnp.ndarray


class ForwardGaussMarkov(NamedTuple):
    init: StdGaussian
    kernels: StdLinearGaussian


class QuadraticFunction(NamedTuple):
    Vxx: jnp.ndarray
    vx: jnp.ndarray
    v0: jnp.ndarray
