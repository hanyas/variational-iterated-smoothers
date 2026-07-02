# Recursive Entropic Variational Inference

Implements the approximate inference algorithms from the paper [Recursive Entropic Variational Inference for Nonlinear State-Space Models](https://arxiv.org/abs/2511.15409).

`varsmooth` performs iterated Gaussian smoothing in nonlinear, non-Gaussian state-space models. Inference is cast as a sequence of entropic trust-region (KL-constrained) updates over a Gauss–Markov posterior, with the model expanded through generalized statistical linear regression or Fourier–Hermite moment matching. See the scripts in `examples/` for demonstrations.

## Installation

Install [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) for the available hardware. Then run

```bash
$ pip install -e .
```

for an editable install. The library core depends only on NumPy, SciPy, and JAX; the plotting used by `examples/` and `experiments/` is an optional extra:

```bash
$ pip install -e ".[examples]"
```

## Quickstart

Run an iterated forward smoother on a small linear-Gaussian model and read off the posterior marginals:

```python
import jax
import numpy as np

from varsmooth.approximation import gauss_hermite_linearization as linearize
from varsmooth.approximation.linearization import get_log_likelihood, get_log_prior, get_log_transition
from varsmooth.objects import AdditiveGaussianModel, AffineGaussian, Gaussian, GaussMarkov
from varsmooth.smoothers.forward_markov import iterated_forward_markov_smoother
from varsmooth.smoothers.utils import std_forward_message

jax.config.update("jax_enable_x64", True)

dim, num_steps = 2, 25
A, H = 0.9 * np.eye(dim), np.eye(dim)
prior = Gaussian(np.zeros(dim), np.eye(dim))
transition = AdditiveGaussianModel(lambda x: A @ x, Gaussian(np.zeros(dim), 0.1 * np.eye(dim)))
likelihood = AdditiveGaussianModel(lambda x: H @ x, Gaussian(np.zeros(dim), 0.1 * np.eye(dim)))
observations = np.zeros((num_steps, dim))  # replace with your data

# initial forward Gauss-Markov posterior: a root marginal plus num_steps kernels
init = GaussMarkov(
    marginal=prior,
    kernels=AffineGaussian(
        np.repeat([0.1 * np.eye(dim)], num_steps, axis=0),
        np.zeros((num_steps, dim)),
        np.repeat([np.eye(dim)], num_steps, axis=0),
    ),
)

# expand the model into quadratic log-potentials via posterior linearization
log_prior_fn = lambda q: get_log_prior(prior, q, linearize)
log_transition_fn = lambda q, _: get_log_transition(transition, q, linearize)
log_likelihood_fn = lambda y, q: get_log_likelihood(y, likelihood, q, linearize)

posterior = iterated_forward_markov_smoother(
    observations, log_prior_fn, log_transition_fn, log_likelihood_fn,
    init, kl_constraint=1.0, verbose=False,
)
marginals = std_forward_message(posterior)  # Gaussian marginals over x_0 .. x_T
```

## Smoothers

The library provides three variational smoothers (`varsmooth/smoothers/`):

- `forward_markov` — forward Gauss–Markov parameterization.
- `reverse_markov` — reverse Gauss–Markov parameterization.
- `hybrid_markov` — forward/reverse (hybrid-Markov) smoother.

Each is available as a single-pass smoother, an undamped iterated smoother, and a KL-constrained iterated smoother with a temperature-based line search.

The quadratic log-potentials are built from the model with either posterior linearization (`extended`, `cubature`, `unscented`, `gauss_hermite`) or Fourier–Hermite moment matching (`cubature`, `gauss_hermite`), under `varsmooth/approximation/`.

## Examples

Single-run demonstrations live under `examples/`, one directory per model family:

- `linear_gaussian` — linear-Gaussian system, validated against the exact RTS smoother.
- `bearing_only` — nonlinear coordinated-turn (bearings-only) tracking.
- `cubic_sensor` — scalar AR(1) state observed through a cubic sensor (Katayama, 2013).
- `stoch_volatility` — univariate stochastic-volatility model.

Each script selects a smoother and an approximation: the prefix is `pl_` (posterior linearization) or `fh_` (Fourier–Hermite), and the suffix is `_forward`, `_reverse`, or `_hybrid`. Available combinations vary by model. For example:

```bash
python examples/linear_gaussian/fh_hybrid.py
python examples/bearing_only/pl_reverse.py
```

## Citation

If you find this code useful, please cite the paper

```bib
@article{abdulsamad2025recursive,
  title = {Recursive Entropic Variational Inference for Nonlinear State-Space Models},
  author = {Hany Abdulsamad and {\'A}ngel F. Garc{\'i}a-Fern{\'a}ndez and Simo S{\"a}rkk{\"a}},
  journal = {arXiv preprint arXiv:2511.15409},
  year = {2025},
}
```

## Credit

This project uses code snippets from [sqrt-parallel-smoothers](https://github.com/EEA-sensors/sqrt-parallel-smoothers).
