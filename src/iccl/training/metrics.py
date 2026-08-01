"""Evaluation metrics for ICCL, computed on the frozen eval suites.

The universal unit is **normalized MSE**: per-position squared error divided by
that task's output variance (``base_mse``), so 1.0 means predicting the mean
and 0 means perfect. Everything derives from the per-(sequence, task, demo)
normalized-MSE array: in-context learning curves (error vs demo index),
task-position curves (forward transfer within a context), the composite-vs-
control benefit of history, and retention on revisit suites (the revisit block
paired demo-by-demo against position-matched controls that hold everything but
the task's identity fixed — a savings measure).

The generic learning and task-position curves belong to the pure-curriculum
suites (in_dist, structural_*). Suites built around a special final task report
that task directly: composite's ``learning_curve`` is the novel composition's
few-shot curve (with history) against composite_control's (without), and
retention reports its revisit against its controls — their curriculum tasks
duplicate in_dist, so no generic curve is emitted for them.
"""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from iccl.data.export import RETENTION_CONTROL_SUITES, load_suite
from iccl.models.model import GDNModel

Suite = dict[str, np.ndarray]

BASE_MSE_FLOOR = 1e-12  # guards the normalization against degenerate tasks
THRESHOLD_NMSE = 0.5  # reporting level for the demos-to-threshold retention scalars


def load_eval_suites(out_dir: Path) -> dict[str, Suite]:
    """Load every frozen suite in ``out_dir``, keyed by suite name."""
    paths = sorted(out_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no frozen eval suites found in {out_dir}; "
            "generate them once with `uv run python scripts/make_eval_sets.py`"
        )
    suites = {path.stem: load_suite(path.with_suffix("")) for path in paths}
    # Retention is read against position-matched controls, so a set carrying the
    # revisit suite without them is stale. Caught here rather than at the first
    # eval, which can be thousands of steps into a job.
    novel = RETENTION_CONTROL_SUITES["novel"]
    if "retention" in suites and novel not in suites:
        raise FileNotFoundError(
            f"the frozen eval sets in {out_dir} have no '{novel}' suite, which the retention "
            "metrics are measured against; delete the directory and regenerate it with "
            "`uv run python scripts/make_eval_sets.py`"
        )
    return suites


@torch.no_grad()
def predict_suite(
    model: GDNModel,
    suite: Suite,
    device: torch.device,
    *,
    batch_size: int = 32,
    autocast_dtype: torch.dtype | None = None,
) -> Float[np.ndarray, "seqs seq_len d_out"]:
    """Run the model over a suite, returning fp32 predictions on CPU."""
    tokens = torch.from_numpy(suite["tokens"])
    token_type = torch.from_numpy(suite["token_type"])
    preds = []
    for i in range(0, tokens.shape[0], batch_size):
        tokens_b = tokens[i : i + batch_size].to(device)
        types_b = token_type[i : i + batch_size].to(device)
        ctx = (
            torch.autocast(device.type, autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with ctx:
            out = model(tokens_b, types_b)
        preds.append(out.preds.float().cpu())
    return torch.cat(preds).numpy()


def demo_nmse(
    preds: Float[np.ndarray, "seqs seq_len d_out"], suite: Suite
) -> Float[np.ndarray, "seqs tasks max_demos"]:
    """Normalized MSE per (sequence, task, demo index), NaN-padded where a
    task has fewer demos than the suite maximum."""
    sq_err = ((preds - suite["targets"]) ** 2).mean(axis=-1)
    spans, counts = suite["task_spans"], suite["demo_counts"]
    denom = np.maximum(suite["base_mse"].mean(axis=-1), BASE_MSE_FLOOR)
    num_seqs, num_tasks = counts.shape
    nmse = np.full((num_seqs, num_tasks, int(counts.max())), np.nan)
    for n in range(num_seqs):
        for k in range(num_tasks):
            count = int(counts[n, k])
            # Demos alternate x, y from the span start; loss sits on the x positions.
            positions = spans[n, k, 0] + 2 * np.arange(count)
            nmse[n, k, :count] = sq_err[n, positions] / denom[n, k]
    return nmse


def suite_scalars(
    nmse: Float[np.ndarray, "seqs tasks max_demos"],
    counts: Float[np.ndarray, "seqs tasks"],
) -> dict[str, float]:
    num_seqs, num_tasks = counts.shape
    last = nmse[np.arange(num_seqs)[:, None], np.arange(num_tasks)[None, :], counts - 1]
    return {
        "nmse_first_demo": float(np.nanmean(nmse[:, :, 0])),
        "nmse_last_demo": float(np.mean(last)),
        "nmse_mean": float(np.nanmean(nmse)),
    }


def _last_task_curve(
    nmse: Float[np.ndarray, "seqs tasks max_demos"],
    counts: Float[np.ndarray, "seqs tasks"],
) -> Float[np.ndarray, " demos"]:
    """Mean curve of the last span, trimmed to that span's own demo count
    (suites are NaN-padded to the longest span, which may be a different one)."""
    return np.nanmean(nmse[:, -1, : int(counts[:, -1].max())], axis=0)


def _paired_difference(
    left: Float[np.ndarray, " demos"], right: Float[np.ndarray, " demos"]
) -> Float[np.ndarray, " demos"]:
    """``left - right`` over the demo indices both curves cover."""
    paired = min(left.shape[0], right.shape[0])
    return left[:paired] - right[:paired]


def _curve_scalars(curve: Float[np.ndarray, " demos"]) -> dict[str, float]:
    """First-demo, last-demo, and mean normalized MSE of a per-demo curve."""
    return {
        "nmse_first_demo": float(curve[0]),
        "nmse_last_demo": float(curve[-1]),
        "nmse_mean": float(np.nanmean(curve)),
    }


def _demos_to_threshold(curve: Float[np.ndarray, " demos"]) -> float:
    """First demo index at or below ``THRESHOLD_NMSE``, censored at the curve
    length when the threshold is never reached."""
    reached = np.flatnonzero(curve <= THRESHOLD_NMSE)
    return float(reached[0]) if reached.size else float(curve.shape[0])


def retention_metrics(
    nmse: Float[np.ndarray, "seqs tasks max_demos"],
    counts: Float[np.ndarray, "seqs tasks"],
    controls: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Retention metrics for the revisited task, read against position-matched
    controls rather than against its own first visit.

    The last span re-demonstrates the first curriculum task. Its own first visit
    sat at task position 0, where the sequence's freshly instantiated modules
    were still unknown, so comparing the two measures serial position as much as
    memory. The controls repeat the sequence with a different task in that same
    final block: ``novel`` demonstrates an undemonstrated task, ``shared`` one
    built from the revisited task's own modules. Writing R for the revisit
    curve, N for novel and N_s for shared,

        savings          = N   - R    total task-specific savings
        episodic_savings = N_s - R    savings not explained by A's modules being familiar
        module_savings   = N   - N_s  savings attributable to A's module set

    which sum by construction: N_s is a partition point, so the total is
    unaffected by where it falls and only the allocation between the two terms
    moves. That allocation is biased, in a direction that is itself empirical. A
    trace of the revisited task warm-starts inference on ``shared`` (it narrows
    which modules to look for, though the model still has to identify them from
    demonstrations), and under a discrete weighting the redrawn weights can land
    close to the original, both of which understate ``episodic_savings``; but
    memory can equally interfere, the model committing to the remembered mixture
    on same-module evidence, which overstates it. The shape of the shared
    control's early curve tells the two apart. Under a binary weighting the
    split does not exist at all — a task is its support, so N_s would be R — and
    only the ``savings`` family is reported.

    ``original_curve`` and ``original_last_demo`` describe the first visit (how
    well the task was learned before interference began) and are not baselines:
    they sit at a different task position, and ``original_last_demo`` also at a
    different demo index, so they must not be differenced against R or N.
    """
    if "novel" not in controls:
        raise KeyError(
            "retention metrics need the position-matched 'novel' control suite; "
            "regenerate the frozen sets with `uv run python scripts/make_eval_sets.py`"
        )
    num_seqs = counts.shape[0]
    original = nmse[:, 0, :]
    original_curve = np.nanmean(original[:, : int(counts[:, 0].max())], axis=0)
    relearning_curve = _last_task_curve(nmse, counts)
    if relearning_curve.shape[0] < 2:
        raise ValueError("retention metrics need a revisit block of at least two demonstrations")
    control_curves = {
        name: _last_task_curve(control_nmse, control_counts)
        for name, (control_nmse, control_counts) in controls.items()
    }

    novel_curve = control_curves["novel"]
    savings = _paired_difference(novel_curve, relearning_curve)
    curves = {
        "original_curve": original_curve,
        "relearning_curve": relearning_curve,
        "control_curve": novel_curve,
        "savings_curve": savings,
    }
    scalars = {
        "original_last_demo": float(np.mean(original[np.arange(num_seqs), counts[:, 0] - 1])),
        "relearning_last_demo": float(relearning_curve[-1]),
        "control_last_demo": float(novel_curve[-1]),
        "savings_demo0": float(savings[0]),
        "savings_one_demo": float(savings[1]),
        "savings_mean": float(np.nanmean(savings)),
        "demos_to_threshold_revisit": _demos_to_threshold(relearning_curve),
        "demos_to_threshold_control": _demos_to_threshold(novel_curve),
    }
    scalars["demos_to_threshold_delta"] = (
        scalars["demos_to_threshold_control"] - scalars["demos_to_threshold_revisit"]
    )

    if "shared" in control_curves:
        shared_curve = control_curves["shared"]
        episodic = _paired_difference(shared_curve, relearning_curve)
        module = _paired_difference(novel_curve, shared_curve)
        curves["control_shared_curve"] = shared_curve
        curves["episodic_savings_curve"] = episodic
        curves["module_savings_curve"] = module
        scalars["episodic_savings_one_demo"] = float(episodic[1])
        scalars["episodic_savings_mean"] = float(np.nanmean(episodic))
        scalars["module_savings_mean"] = float(np.nanmean(module))
    return scalars, curves


def evaluate_suites(
    model: GDNModel,
    suites: dict[str, Suite],
    device: torch.device,
    *,
    batch_size: int = 32,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Evaluate every suite, returning flat ``{suite}/{metric}`` scalar and
    curve dicts (scalars go to the logger; curves to the run dir)."""
    scalars: dict[str, float] = {}
    curves: dict[str, np.ndarray] = {}
    nmses: dict[str, np.ndarray] = {}
    for name, suite in suites.items():
        preds = predict_suite(
            model, suite, device, batch_size=batch_size, autocast_dtype=autocast_dtype
        )
        nmse = demo_nmse(preds, suite)
        nmses[name] = nmse

        # The generic in-context-learning curves belong to the pure-curriculum
        # suites (in_dist, structural_*). Suites built around a special final
        # task (composite, retention) draw their curriculum from the same
        # distribution as in_dist, so a curriculum curve here would only
        # duplicate it; those suites report their final task through the
        # dedicated curves below instead.
        num_tasks = suite["demo_counts"].shape[1]
        curriculum = (
            int(suite["num_curriculum_tasks"][0]) if "num_curriculum_tasks" in suite else num_tasks
        )
        if curriculum == num_tasks:
            for key, value in suite_scalars(nmse, suite["demo_counts"]).items():
                scalars[f"{name}/{key}"] = value
            curves[f"{name}/learning_curve"] = np.nanmean(nmse, axis=(0, 1))
            curves[f"{name}/task_position_curve"] = np.nanmean(nmse, axis=(0, 2))

    if "composite" in nmses and "composite_control" in nmses:
        # The composite suite's point is the novel few-shot composition (its
        # last task); the curriculum before it only supplies in-context history.
        # Report that task's demo curve with history (composite) and without
        # (control), and the benefit of history as their difference.
        composite_curve = _last_task_curve(nmses["composite"], suites["composite"]["demo_counts"])
        control_curve = _last_task_curve(
            nmses["composite_control"], suites["composite_control"]["demo_counts"]
        )
        curves["composite/learning_curve"] = composite_curve
        curves["composite_control/learning_curve"] = control_curve
        for key, value in _curve_scalars(composite_curve).items():
            scalars[f"composite/{key}"] = value
        for key, value in _curve_scalars(control_curve).items():
            scalars[f"composite_control/{key}"] = value
        benefit_curve = _paired_difference(control_curve, composite_curve)
        curves["composite/benefit_curve"] = benefit_curve
        scalars["composite/benefit_first_demo"] = float(benefit_curve[0])
        scalars["composite/benefit_last_demo"] = float(benefit_curve[-1])
    if "retention" in nmses:
        controls = {
            mode: (nmses[name], suites[name]["demo_counts"])
            for mode, name in RETENTION_CONTROL_SUITES.items()
            if name in nmses
        }
        retention_s, retention_c = retention_metrics(
            nmses["retention"], suites["retention"]["demo_counts"], controls
        )
        for key, value in retention_s.items():
            scalars[f"retention/{key}"] = value
        for key, curve in retention_c.items():
            curves[f"retention/{key}"] = curve
    return scalars, curves
