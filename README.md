# Variational Iterated Gaussian Smoothing

Implements the approximate inference algorithms from the paper [Proximal Approximate Inference in State-Space Models](https://arxiv.org/abs/2511.15409). This code was written by [Hany Abdulsamad](https://github.com/hanyas).

`varsmooth` performs iterated Gaussian smoothing in nonlinear, non-Gaussian state-space models. Inference is cast as a sequence of entropic trust-region (KL-constrained) updates over a Gauss–Markov posterior, with the model expanded through generalized statistical linear regression or Fourier–Hermite moment matching. See the scripts in `examples/` for demonstrations.

## Installation

Install [JAX](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) for the available hardware. Then run

```bash
$ pip install -e .
```

for an editable install.

## Smoothers

The library provides three variational smoothers (`varsmooth/smoothers/`):

- `forward_markov` — forward Gauss–Markov parameterization.
- `reverse_markov` — reverse Gauss–Markov parameterization.
- `two_filter` — combined forward/reverse (two-filter) smoother.

Each is available as a single-pass smoother, an undamped iterated smoother, and a KL-constrained iterated smoother with a temperature-based line search.

The quadratic log-potentials are built from the model with either posterior linearization (`extended`, `cubature`, `unscented`, `gauss_hermite`) or Fourier–Hermite moment matching (`cubature`, `gauss_hermite`), under `varsmooth/approximation/`.

## Examples

Two model families are provided under `examples/`:

- `linear_model` — a linear-Gaussian system, validated against the exact RTS smoother.
- `bearing_model` — a nonlinear coordinated-turn (bearings-only) tracking task.

Each script selects a smoother (forward / reverse / two-filter) and an approximation, named `pl_` for posterior linearization and `fh_` for Fourier–Hermite. For example:

```bash
python examples/bearing_model/pl_reverse.py
```

## Citation

If you find this code useful, please cite the paper

```bib
@article{abdulsamad2025proximal,
  title = {Proximal Approximate Inference in State-Space Models},
  author = {Hany Abdulsamad and {\'A}ngel F. Garc{\'i}a-Fern{\'a}ndez and Simo S{\"a}rkk{\"a}},
  journal = {arXiv preprint arXiv:2511.15409},
  year = {2025},
}
```

## Credit

This project uses code snippets from [sqrt-parallel-smoothers](https://github.com/EEA-sensors/sqrt-parallel-smoothers).
