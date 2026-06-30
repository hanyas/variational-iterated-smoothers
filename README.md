# Recursive Entropic Variational Smoothing

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

Single-run demonstrations live under `examples/`, one directory per model family:

- `linear_gaussian` — linear-Gaussian system, validated against the exact RTS smoother.
- `bearing_only` — nonlinear coordinated-turn (bearings-only) tracking.
- `cubic_sensor` — scalar AR(1) state observed through a cubic sensor (Katayama, 2013).
- `stoch_volatility` — univariate stochastic-volatility model.

Each script selects a smoother and an approximation: the prefix is `pl_` (posterior linearization) or `fh_` (Fourier–Hermite), and the suffix is `_forward`, `_reverse`, or `_two_filt`. Available combinations vary by model. For example:

```bash
python examples/linear_gaussian/fh_two_filt.py
python examples/bearing_only/pl_reverse.py
```

## Experiments

The quantitative experiments from the paper live under `experiments/`. Shared utilities (metrics, smoother runners, plotting and I/O) are collected in `experiments/common.py`, with a thin per-model harness (`_<model>_common.py`):

- `linear_gaussian/` — exact recovery across approximations, trust-region convergence to the exact posterior, and entropic β-interpolation (`exp1_recovery.py`, `exp2_proximal.py`, `exp3_interpolation.py`; `run_all.py` runs all three).
- `cubic_sensor/` — ELBO and calibration NLPD per iteration, the damped smoother vs. undamped IPLS (`exp1_elbo_nlpd.py`; the IPLS baseline uses [`parsmooth`](https://github.com/EEA-sensors/sqrt-parallel-smoothers)).
- `stoch_volatility/` — smoothed log-volatility vs. ground truth, and an accuracy / NLPD sweep over the vol-of-vol (`exp1_comparison.py`, `exp2_sweep.py`).

Each script writes its figures and CSVs to the model's `outputs/` directory; run it from the model directory, e.g.:

```bash
cd experiments/linear_gaussian && python run_all.py
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
