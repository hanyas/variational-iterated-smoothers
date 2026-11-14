"""Shared helpers for the differential-drive experiments."""

import csv
from functools import partial
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from experiments.common import save_fig
from experiments.common import set_style
from experiments.common import write_csv
from varsmooth.approximation import cubature_quadratization
from varsmooth.approximation.fourier_hermite import get_log_likelihood
from varsmooth.approximation.fourier_hermite import get_log_prior
from varsmooth.baselines.convergence import make_convergence_checker
from varsmooth.baselines.convergence import symmetric_chain_kl
from varsmooth.baselines.gaussian import expected_gaussian_logpdf
from varsmooth.baselines.gaussian import make_elbo as make_elbo_evaluator
from varsmooth.environments import differential_drive as robot
from varsmooth.environments.differential_drive import make_parameters
from varsmooth.objects import AffineGaussian
from varsmooth.objects import AdditiveGaussianModel
from varsmooth.objects import Gaussian
from varsmooth.objects import GaussMarkov
from varsmooth.objects import LogTransition
from varsmooth.smoothers.reverse_markov import iterated_reverse_markov_smoother
from varsmooth.smoothers.rts_kalman import filtering
from varsmooth.smoothers.rts_kalman import smoothing
from varsmooth.smoothers.utils import initialize_reverse_with_forward
from varsmooth.smoothers.utils import get_pairwise_marginal

PHYSICAL = dict(dt=0.1, m=5.0, J=2.0, l=0.15, r1=1.0, r2=1.0, a=0.5)
PRIOR_MEAN = np.zeros(5)
PRIOR_COV = np.diag(np.array([0.1, 0.1, 0.1, 0.5, 0.2]) ** 2)
STATE_SCALE = np.array([0.1, 0.1, 0.05, 0.3, 0.1])
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

PANELS = (
    ("linearized", "results_drive_elbo_obs.csv", "(a) Observation-based initialization"),
    ("prior", "results_drive_elbo_prior.csv", "(b) Observation-independent initialization"),
)
SETTINGS = (
    ("REVS_eps5", "REVS", 5.0, "0.05", "-o", r"REVS ($\varepsilon=5$)"),
    ("REVS_eps25", "REVS", 25.0, "0.5", "-o", r"REVS ($\varepsilon=25$)"),
    ("Barfoot_alpha0.25", "Barfoot", 0.25, "#238b45", "--s", r"Barfoot ($\alpha=0.25$)"),
    ("Barfoot_alpha0.5", "Barfoot", 0.5, "#005a32", "--s", r"Barfoot ($\alpha=0.5$)"),
    ("Courts", "Courts", None, "tab:blue", ":^", "Courts"),
)


def make_prior():
    """Build the initial-state Gaussian prior.

    Returns:
        prior: Gaussian
            Five-dimensional initial-state distribution.
    """
    return Gaussian(jnp.asarray(PRIOR_MEAN), jnp.asarray(PRIOR_COV))


def simulate_data(prior, Q, R, num_steps, rng):
    """Simulate the controlled robot and its pose observations.

    Args:
        prior: Gaussian
            Initial-state distribution.
        Q: array_like
            Process covariance of shape (5, 5).
        R: array_like
            Observation covariance of shape (3, 3).
        num_steps: int
            Number of transitions.
        rng: numpy.random.RandomState
            Random state driving the simulation.

    Returns:
        true_states: array_like
            True states of shape (T + 1, 5).
        observations: array_like
            Noisy poses of shape (T, 3).
        controls: array_like
            Wheel-force inputs of shape (T, 2).
    """
    t = PHYSICAL["dt"] * np.arange(num_steps)
    forward = 0.25 + 0.05 * np.sin(0.1 * t)
    turn = 0.15 * np.sin(0.25 * t)
    controls = np.column_stack([forward + turn, forward - turn])
    x0 = rng.multivariate_normal(prior.mean, prior.cov)
    _, truth, observations = robot.get_data(x0, controls, Q=Q, R=R, random_state=rng, **PHYSICAL)
    return jnp.asarray(truth), observations, controls


def make_model_fns(prior, Q, R):
    """Build the three analytic Gaussian log-factor expectations.

    Args:
        prior: Gaussian
            Initial-state distribution.
        Q: array_like
            Diagonal process covariance of shape (5, 5).
        R: array_like
            Observation covariance of shape (3, 3).

    Returns:
        log_prior_fn: callable
            Expected log prior taking (mean, covariance).
        log_transition_fn: callable
            Expected log transition taking (pair_mean, pair_covariance, control).
        log_likelihood_fn: callable
            Expected log likelihood taking (mean, covariance, observation).
    """
    if not np.array_equal(Q, np.diag(np.diag(Q))):
        raise ValueError("The exact transition expectation requires diagonal Q")
    log_prior_fn = partial(expected_gaussian_logpdf, target_mean=prior.mean, target_cov=prior.cov)
    log_transition_fn = partial(robot.expected_transition, Q=Q, **PHYSICAL)

    def log_likelihood_fn(mean, cov, y):
        """Evaluate the Gaussian expected pose log likelihood.

        Args:
            mean: array_like
                State mean of shape (5,).
            cov: array_like
                State covariance of shape (5, 5).
            y: array_like
                Pose observation; all-NaN denotes a missing observation.

        Returns:
            value: float
                Expected log likelihood.
        """
        observed = jnp.all(jnp.isfinite(y))
        safe_y = jnp.where(observed, y, 0.0)
        return jnp.where(observed, expected_gaussian_logpdf(mean[:3], cov[:3, :3], safe_y, R), 0.0)

    return log_prior_fn, log_transition_fn, log_likelihood_fn


def make_revs_fns(prior, Q, R, controls):
    """Build numerical FH factors using third-degree cubature.

    Args:
        prior: Gaussian
            Initial-state distribution.
        Q: array_like
            Process covariance of shape (5, 5).
        R: array_like
            Observation covariance of shape (3, 3).
        controls: array_like
            Wheel-force inputs of shape (T, 2).

    Returns:
        log_prior_fn: callable
            Quadratic prior taking a Gaussian marginal.
        log_transition_fn: callable
            Quadratic transitions taking preceding marginals and forward kernels.
        log_likelihood_fn: callable
            Quadratic likelihoods taking observations and state marginals.
    """
    _, _, f, h, _, _ = robot.make_parameters(Q=Q, R=R, **PHYSICAL)
    observation_model = AdditiveGaussianModel(h, Gaussian(jnp.zeros(3), R))
    log_prior_fn = partial(get_log_prior, prior, method=cubature_quadratization)

    def transition(marginal, kernel, control):
        """Quadratize one controlled transition under its pairwise marginal.

        Args:
            marginal: Gaussian
                Preceding state marginal.
            kernel: AffineGaussian
                Forward conditional kernel.
            control: array_like
                Wheel forces of shape (2,).

        Returns:
            factor: LogTransition
                FH coefficients in (next, previous) state order.
        """
        model = AdditiveGaussianModel(lambda x: f(x, control), Gaussian(jnp.zeros(5), Q))
        pair = get_pairwise_marginal(marginal, kernel)
        C, c, kappa = cubature_quadratization(lambda z: model.log_prob(z[:5], z[5:]), pair)
        return LogTransition(C[:5, :5], -C[:5, 5:], -C[5:, :5], C[5:, 5:], c[:5], c[5:], kappa)

    def log_transition_fn(marginals, kernels):
        """Build the controlled transition factors.

        Args:
            marginals: Gaussian
                Batched preceding state marginals.
            kernels: AffineGaussian
                Batched forward conditional kernels.

        Returns:
            factors: LogTransition
                Batched quadratic transition factors.
        """
        return jax.vmap(transition)(marginals, kernels, jnp.asarray(controls))

    def log_likelihood_fn(observations, marginals):
        """Build pose likelihood factors, omitting missing observations.

        Args:
            observations: array_like
                Pose observations of shape (T, 3).
            marginals: Gaussian
                Batched observed-state marginals.

        Returns:
            factors: LogLikelihood
                Batched quadratic likelihood factors.
        """
        observed = jnp.all(jnp.isfinite(observations), axis=-1)
        safe_y = jnp.where(observed[:, None], observations, 0.0)
        factors = get_log_likelihood(safe_y, observation_model, marginals, cubature_quadratization)
        return jax.tree.map(
            lambda x: jnp.where(observed.reshape((-1,) + (1,) * (x.ndim - 1)), x, 0.0), factors
        )

    return log_prior_fn, log_transition_fn, log_likelihood_fn


def run_revs(
    observations,
    log_prior_fn,
    log_transition_fn,
    log_likelihood_fn,
    initial,
    *,
    kl_constraint,
    max_iterations,
    evaluator,
    stationarity_tolerance,
    feasibility_tolerance,
    callback=None,
):
    """Run library REVS updates with the experiment's shared stopping rule.

    Args:
        observations: array_like
            Pose observations of shape (T, 3).
        log_prior_fn: callable
            Numerical FH prior expansion.
        log_transition_fn: callable
            Numerical FH transition expansion.
        log_likelihood_fn: callable
            Numerical FH likelihood expansion.
        initial: GaussMarkov
            Initial reverse Gaussian chain.
        kl_constraint: float
            Total-chain KL budget per update.
        max_iterations: int
            Maximum number of updates.
        evaluator: callable
            Shared analytic ELBO and stationarity evaluator.
        stationarity_tolerance: float
            Common gradient tolerance.
        feasibility_tolerance: float
            Common consistency tolerance.
        callback: callable, optional
            Recorder taking (diagnostics, posterior).

    Returns:
        posterior: GaussMarkov
            Final reverse Gaussian posterior.
        history: list
            Per-iteration convergence diagnostics.
    """
    update = partial(
        iterated_reverse_markov_smoother,
        observations,
        log_prior_fn,
        log_transition_fn,
        log_likelihood_fn,
        kl_constraint=kl_constraint,
        max_iterations=1,
        return_history=True,
        verbose=False,
    )
    check_convergence = make_convergence_checker(
        evaluator, len(observations), stationarity_tolerance, feasibility_tolerance
    )
    compile_start = time.perf_counter()
    jax.block_until_ready(update(initial))
    check = check_convergence(initial)
    jax.block_until_ready(symmetric_chain_kl(initial, initial))
    compile_seconds = time.perf_counter() - compile_start
    posterior = initial
    record = dict(iteration=0, seconds=0.0, **check)
    history = [record]
    if callback is not None:
        callback(record, posterior)
    started = time.perf_counter()
    message = "iteration limit"
    for iteration in range(1, max_iterations + 1):
        if record["converged"] or not record["valid"]:
            break
        posterior, diagnostics = jax.block_until_ready(update(posterior))
        check = check_convergence(posterior)
        record = dict(
            iteration=iteration,
            seconds=time.perf_counter() - started,
            realized_kl=float(diagnostics["realized_kl"][0]),
            damping=float(diagnostics["damping"][0]),
            **check,
        )
        history.append(record)
        if callback is not None:
            callback(record, posterior)
        if not bool(diagnostics["feasible"][0]):
            message = "no feasible library REVS update"
            break
    if record["converged"]:
        message = "shared stationarity and feasibility tolerances satisfied"
    elif not record["valid"]:
        message = "invalid posterior"
    history[-1].update(
        message=message,
        total_seconds=time.perf_counter() - started,
        compile_seconds=compile_seconds,
    )
    return posterior, history


def make_prior_init(prior, Q, controls):
    """Build a reverse Gaussian chain from the noiseless prior rollout.

    Args:
        prior: Gaussian
            Initial-state distribution.
        Q: array_like
            Process covariance of shape (5, 5).
        controls: array_like
            Wheel-force inputs of shape (T, 2).

    Returns:
        posterior: GaussMarkov
            Observation-independent reverse Gaussian chain.
    """
    _, _, f, _, df, _ = robot.make_parameters(Q=Q, **PHYSICAL)

    def predict(mean, control):
        """Linearize one transition along the noiseless rollout.

        Args:
            mean: array_like
                State mean of shape (5,).
            control: array_like
                Wheel forces of shape (2,).

        Returns:
            next_mean: array_like
                Predicted state mean.
            affine: tuple
                Transition Jacobian and offset.
        """
        next_mean = f(mean, control)
        F = df(mean, control)
        return next_mean, (F, next_mean - F @ mean)

    _, (F, d) = jax.lax.scan(predict, prior.mean, jnp.asarray(controls))
    forward = GaussMarkov(prior, AffineGaussian(F, d, jnp.broadcast_to(Q, (len(controls), 5, 5))))
    return initialize_reverse_with_forward(forward)


def make_observation_init(prior, Q, R, controls, observations):
    """Build a reverse Gaussian chain using an observation-based Kalman/RTS pass.

    Args:
        prior: Gaussian
            Initial-state distribution.
        Q: array_like
            Process covariance of shape (5, 5).
        R: array_like
            Observation covariance of shape (3, 3).
        controls: array_like
            Wheel-force inputs of shape (T, 2).
        observations: array_like
            Pose observations of shape (T, 3); all-NaN rows are omitted.

    Returns:
        posterior: GaussMarkov
            Reverse Gaussian chain from the shared Kalman/RTS initialization.
    """
    T = len(controls)
    observed = np.all(np.isfinite(observations), axis=1)
    poses = np.asarray(observations)
    if not np.all(observed):
        retained = np.concatenate([[0], np.flatnonzero(observed) + 1])
        values = np.concatenate([np.asarray(prior.mean)[None, :3], poses[observed]])
        poses = np.column_stack([np.interp(np.arange(1, T + 1), retained, values[:, i]) for i in range(3)])
    means = jnp.concatenate([prior.mean[None], jnp.pad(jnp.asarray(poses), ((0, 0), (0, 2)))])
    _, _, f, _, df, _ = robot.make_parameters(Q=Q, R=R, **PHYSICAL)
    F = jax.vmap(df)(means[:-1], controls)
    d = jax.vmap(f)(means[:-1], controls) - jnp.einsum("nij,nj->ni", F, means[:-1])
    transition = AffineGaussian(F, d, jnp.broadcast_to(Q, (T, 5, 5)))
    observation = AffineGaussian(
        observed[:, None, None] * jnp.broadcast_to(jnp.eye(3, 5), (T, 3, 5)),
        jnp.zeros((T, 3)),
        jnp.broadcast_to(R, (T, 3, 3)),
    )
    filtered = filtering(np.where(observed[:, None], observations, 0.0), prior, transition, observation)
    return initialize_reverse_with_forward(smoothing(transition, filtered))


def save_results(
    results, initialization, seed, num_steps, max_iterations, stationarity_tolerance, feasibility_tolerance
):
    """Write per-iteration scores and convergence diagnostics to CSV.

    Args:
        results: dict
            Mapping from curve names to (posterior, history).
        initialization: str
            Initialization identifier for the output panel.
        seed: int
            Data-generation seed.
        num_steps: int
            Number of transitions.
        max_iterations: int
            Common iteration limit.
        stationarity_tolerance: float
            Common ELBO-gradient tolerance.
        feasibility_tolerance: float
            Common consistency tolerance.
    """
    rows = []
    initial_elbo = None
    for curve, method, setting, *_ in SETTINGS:
        _, history = results[curve]
        final_record = history[-1]
        if initial_elbo is None:
            initial_elbo = history[0]["elbo"]
        np.testing.assert_allclose(history[0]["elbo"], initial_elbo, rtol=1e-12, atol=1e-9)
        assert not any(record["converged"] for record in history[:-1])
        assert history[-1]["converged"] == final_record["converged"]
        for index, record in enumerate(history):
            final = index == len(history) - 1
            certified = (
                record["valid"]
                and record["stationarity"] <= stationarity_tolerance
                and record["constraint_violation"] <= feasibility_tolerance
            )
            assert record["converged"] == certified
            rows.append(
                dict(
                    initialization=initialization,
                    seed=seed,
                    num_steps=num_steps,
                    curve=curve,
                    method=method,
                    kl_constraint=setting if method == "REVS" else "",
                    learning_rate=setting if method == "Barfoot" else "",
                    iteration=record["iteration"],
                    elbo=record["elbo"],
                    stationarity=record["stationarity"],
                    constraint_violation=record["constraint_violation"],
                    valid=record["valid"],
                    converged=record["converged"],
                    seconds=record["seconds"],
                    solver_objective=record.get("solver_elbo", np.nan),
                    final=final,
                    termination=final_record["message"] if final else "",
                    compile_seconds=final_record["compile_seconds"] if final else "",
                    max_iterations=max_iterations,
                    stationarity_tolerance=stationarity_tolerance,
                    feasibility_tolerance=feasibility_tolerance,
                    state_scale=" ".join(map(str, STATE_SCALE)),
                )
            )
        print(
            f"  {curve}: {final_record['iteration']} iterations, converged={final_record['converged']}, "
            f"ELBO={final_record['elbo']:.9f}, stationarity={final_record['stationarity']:.3g}",
            flush=True,
        )
    OUTPUT_DIR.mkdir(exist_ok=True)
    filename = next(filename for name, filename, _ in PANELS if name == initialization)
    write_csv(
        OUTPUT_DIR / filename,
        [
            {
                key: "" if isinstance(value, float) and not np.isfinite(value) else value
                for key, value in row.items()
            }
            for row in rows
        ],
    )


def plot_comparison(output=OUTPUT_DIR):
    """Plot saved objectives and Courts consistency for both initializations.

    Args:
        output: str or Path
            Directory containing the two CSVs and the combined PDF.
    """
    import matplotlib.pyplot as plt

    output = Path(output)
    if not all((output / filename).exists() for _, filename, _ in PANELS):
        print("CSV saved; run the other initialization script to create the combined PDF.", flush=True)
        return
    set_style()
    fig, axes = plt.subplots(
        2, 2, figsize=(13.0, 7.2), sharex="col", gridspec_kw=dict(height_ratios=(2.0, 1.0))
    )
    metadata = []
    for column, (initialization, filename, title) in enumerate(PANELS):
        ax, consistency_ax = axes[:, column]
        if initialization == "prior":
            ax.set_yscale("symlog", linthresh=1e4)
        with (output / filename).open() as stream:
            rows = list(csv.DictReader(stream))
        assert all(row["initialization"] == initialization for row in rows)
        metadata.append(
            {
                (
                    row["seed"],
                    row["num_steps"],
                    row["max_iterations"],
                    row["stationarity_tolerance"],
                    row["feasibility_tolerance"],
                    row["state_scale"],
                )
                for row in rows
            }
        )
        notes = []
        tolerance = float(rows[0]["feasibility_tolerance"])
        for curve, method, _, color, style, label in SETTINGS:
            history = [row for row in rows if row["curve"] == curve]
            assert history
            iterations = np.array([int(row["iteration"]) for row in history])
            # Away from consistency, Courts' native objective need not be a
            # posterior ELBO. Show it throughout, with the residual below.
            score = "solver_objective" if method == "Courts" else "elbo"
            values = np.array([float(row[score]) if row[score] else np.nan for row in history])
            ax.plot(iterations, values, style, color=color, ms=3.0, label=label)
            if method == "Courts":
                residuals = np.array([float(row["constraint_violation"]) for row in history])
                consistency_ax.semilogy(iterations, residuals, style, color=color, ms=3.0)
            final = history[-1]
            assert final["final"] == "True"
            if final["converged"] != "True":
                reason = (
                    "consistency unmet"
                    if float(final["constraint_violation"]) > tolerance
                    else "not converged"
                )
                notes.append(f"{label}: {reason} at iteration {final['iteration']}")
        ax.set(ylabel="ELBO (Courts: objective)", title=title)
        consistency_ax.set(xlabel="Iteration", ylabel="Courts consistency\nresidual (log scale)")
        consistency_ax.axhline(tolerance, color="0.3", ls="--", lw=1.2)
        consistency_ax.annotate(
            f"Tolerance: {tolerance:g}",
            xy=(0.98, tolerance),
            xycoords=consistency_ax.get_yaxis_transform(),
            xytext=(0, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
        )
        if initialization == "linearized":
            ax.set_xlim(0, 15)
            ax.set_xticks(np.arange(0, 16, 3))
        if initialization == "prior":
            ax.set_xlim(0, 75)
            ax.set_xticks(np.arange(0, 76, 15))
            ax.set_ylabel("ELBO (Courts: objective)\n(symmetric log scale)")
        if notes:
            ax.text(
                0.98,
                0.04,
                "\n".join(notes),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.9),
            )
    assert len(metadata[0]) == 1 and metadata[0] == metadata[1]
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.035), ncol=5, frameon=False, fontsize=9
    )
    fig.text(
        0.5,
        0.015,
        "Courts: the objective equals a posterior ELBO only when adjacent marginals are consistent.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.11, 1.0, 1.0))
    save_fig(fig, output / "fig_elbo_initializations.pdf")
