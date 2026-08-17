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

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float

from iccl.data.export import (
    RETENTION_CONTROL_SUITES,
    load_suite,
    load_suite_metadata,
)
from iccl.models.model import GDNModel

Suite = dict[str, Any]

BASE_MSE_FLOOR = 1e-12  # guards the normalization against degenerate tasks
THRESHOLD_NMSE = 0.5  # reporting level for the demos-to-threshold retention scalars


@dataclass
class EvaluationReport:
    """Evaluation payload shared by training, standalone evaluation, and W&B."""

    scalars: dict[str, float]
    curves: dict[str, np.ndarray]
    summary_rows: list[dict[str, Any]]
    curve_rows: list[dict[str, Any]]

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Preserve two-value unpacking for callers that only need legacy outputs."""
        yield self.scalars
        yield self.curves


def load_eval_suites(out_dir: Path) -> dict[str, Suite]:
    """Load every frozen suite in ``out_dir``, keyed by suite name."""
    paths = sorted(out_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no frozen eval suites found in {out_dir}; "
            "generate them once with `uv run python scripts/make_eval_sets.py`"
        )
    suites = {}
    for path in paths:
        suite: Suite = load_suite(path.with_suffix(""))
        metadata_path = path.with_suffix(".meta.json")
        if metadata_path.exists():
            suite["__meta__"] = load_suite_metadata(metadata_path)
        suites[path.stem] = suite
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
    variable_groups: dict[tuple[str, tuple[str, ...]], set[str]] = {}
    for name in suites:
        parts = name.split("__")
        if len(parts) >= 7 and parts[0] in {"composition", "retention"}:
            variable_groups.setdefault((parts[0], tuple(parts[2:])), set()).add(parts[1])
    for (capability, cell), conditions in variable_groups.items():
        required = {"constituent", "matched_prefix"} if capability == "composition" else {
            "repeat",
            "novel",
        }
        missing = required - conditions
        if missing:
            raise FileNotFoundError(
                f"frozen {capability} cell {'__'.join(cell)} is missing paired conditions "
                f"{sorted(missing)}; regenerate with `uv run python scripts/make_eval_sets.py`"
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
    mse = demo_mse(preds, suite)
    denom = np.maximum(suite["base_mse"].mean(axis=-1), BASE_MSE_FLOOR)
    return mse / denom[:, :, None]


def demo_mse(
    preds: Float[np.ndarray, "seqs seq_len d_out"], suite: Suite
) -> Float[np.ndarray, "seqs tasks max_demos"]:
    """Raw MSE per (sequence, task, demo index), with the same causal x-token
    extraction and NaN padding as :func:`demo_nmse`."""
    sq_err = ((preds - suite["targets"]) ** 2).mean(axis=-1)
    spans, counts = suite["task_spans"], suite["demo_counts"]
    num_seqs, num_tasks = counts.shape
    mse = np.full((num_seqs, num_tasks, int(counts.max())), np.nan)
    for n in range(num_seqs):
        for k in range(num_tasks):
            count = int(counts[n, k])
            # Demos alternate x, y from the span start; loss sits on the x positions.
            positions = spans[n, k, 0] + 2 * np.arange(count)
            mse[n, k, :count] = sq_err[n, positions]
    return mse


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


def _last_task_values(
    errors: Float[np.ndarray, "seqs tasks max_demos"],
    counts: Float[np.ndarray, "seqs tasks"],
) -> Float[np.ndarray, "seqs demos"]:
    """Per-sequence final-task errors trimmed to the final block's width."""
    width = int(counts[:, -1].max())
    return errors[:, -1, :width]


def _sequence_last_values(
    errors: Float[np.ndarray, "seqs tasks max_demos"],
    counts: Float[np.ndarray, "seqs tasks"],
) -> Float[np.ndarray, "seqs tasks"]:
    sequences, tasks = counts.shape
    return errors[
        np.arange(sequences)[:, None],
        np.arange(tasks)[None, :],
        counts - 1,
    ]


def _mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float, float]:
    """Sequence-clustered bootstrap interval for a scalar sequence statistic."""
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(clean.mean())
    if clean.size == 1 or replicates < 2:
        return mean, mean, mean
    rng = np.random.Generator(np.random.Philox(key=np.array([seed, clean.size], dtype=np.uint64)))
    indices = rng.integers(0, clean.size, size=(replicates, clean.size))
    bootstrap = clean[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return mean, float(low), float(high)


def _curve_intervals(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise sequence-clustered bootstrap intervals for a curve."""
    width = values.shape[1]
    mean = np.full(width, np.nan, dtype=np.float64)
    low = np.full(width, np.nan, dtype=np.float64)
    high = np.full(width, np.nan, dtype=np.float64)
    for position in range(width):
        mean[position], low[position], high[position] = _mean_interval(
            values[:, position], seed=seed + position, replicates=replicates
        )
    return mean, low, high


def _suite_descriptor(name: str, suite: Suite) -> dict[str, Any]:
    """Structural coordinates encoded by a variable-world suite name/sidecar."""
    parts = name.split("__")
    if len(parts) < 7:
        return {
            "suite": name,
            "capability": "legacy",
            "condition": name,
            "slice": "legacy",
            "status": "legacy",
            "M": None,
            "T": int(suite["num_curriculum_tasks"][0])
            if "num_curriculum_tasks" in suite
            else None,
            "S": None,
            "B_history": None,
            "L_history": None,
            "variant": "",
            "sampler": "legacy",
            "weighting": "unknown",
            "demo_counts": None,
        }
    capability, condition, slice_name, status = parts[:4]
    modules = int(parts[4][1:])
    tasks = int(parts[5][1:])
    prediction_tokens = int(parts[6][1:])
    metadata = suite.get("__meta__", {})
    return {
        "suite": name,
        "capability": capability,
        "condition": condition,
        "slice": slice_name,
        "status": status,
        "M": modules,
        "T": tasks,
        "S": tasks - (modules - 1),
        "B_history": prediction_tokens,
        "L_history": 2 * prediction_tokens + tasks,
        "variant": parts[7] if len(parts) > 7 else "",
        "sampler": metadata.get("config", {})
        .get("sequence", {})
        .get("curriculum_sampler", "unknown"),
        "weighting": metadata.get("config", {}).get("weighting", "unknown"),
        "demo_counts": metadata.get("demo_counts"),
    }


def _cell_slug(descriptor: dict[str, Any]) -> str:
    variant = f"__{descriptor['variant']}" if descriptor["variant"] else ""
    return (
        f"{descriptor['slice']}__{descriptor['status']}__m{descriptor['M']:02d}__"
        f"t{descriptor['T']:02d}__b{descriptor['B_history']:04d}{variant}"
    )


def _row_base(descriptor: dict[str, Any], condition: str) -> dict[str, Any]:
    demo_counts = descriptor.get("demo_counts")
    if demo_counts:
        demos = np.asarray(demo_counts, dtype=np.float64)
        d_min, d_max, d_mean = float(demos.min()), float(demos.max()), float(demos.mean())
        d_cv = float(demos.std() / d_mean) if d_mean else 0.0
    else:
        d_min = d_max = d_mean = d_cv = None
    return {
        "capability": descriptor["capability"],
        "condition": condition,
        "slice": descriptor["slice"],
        "variant": descriptor["variant"],
        "suite": descriptor["suite"],
        "status": descriptor["status"],
        "sampler": descriptor["sampler"],
        "weighting": descriptor["weighting"],
        "M": descriptor["M"],
        "T": descriptor["T"],
        "S": descriptor["S"],
        "B_history": descriptor["B_history"],
        "L_history": descriptor["L_history"],
        "D_target": None,
        "D_min": d_min,
        "D_max": d_max,
        "D_mean": d_mean,
        "D_cv": d_cv,
        "constituent_exposure_min": None,
        "constituent_exposure_max": None,
        "constituent_demo_exposure_min": None,
        "constituent_demo_exposure_max": None,
        "intervening_tasks": None,
        "prediction_token_delay": None,
        "serialized_token_delay": None,
    }


def _summary_row(
    descriptor: dict[str, Any],
    condition: str,
    metric: str,
    sequence_values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    suite: Suite,
) -> dict[str, Any]:
    mean, low, high = _mean_interval(sequence_values, seed=seed, replicates=replicates)
    row = _row_base(descriptor, condition)
    row.update(
        metric=metric,
        value=mean,
        ci_low=low,
        ci_high=high,
        n_sequences=int(len(sequence_values)),
    )
    if "demo_counts" in suite:
        row["D_target"] = int(suite["demo_counts"][0, -1])
    if "constituent_task_exposures" in suite:
        exposure = suite["constituent_task_exposures"]
        row["constituent_exposure_min"] = int(exposure.min())
        row["constituent_exposure_max"] = int(exposure.max())
    if "constituent_demo_exposures" in suite:
        exposure = suite["constituent_demo_exposures"]
        row["constituent_demo_exposure_min"] = int(exposure.min())
        row["constituent_demo_exposure_max"] = int(exposure.max())
    for key in ("intervening_tasks", "prediction_token_delay", "serialized_token_delay"):
        if key in suite:
            row[key] = float(np.mean(suite[key]))
    return row


def _curve_table_rows(
    descriptor: dict[str, Any],
    condition: str,
    curve_type: str,
    mse_values: np.ndarray,
    nmse_values: np.ndarray,
    *,
    x_name: str,
    seed: int,
    replicates: int,
) -> list[dict[str, Any]]:
    mse_mean, _, _ = _curve_intervals(mse_values, seed=seed, replicates=replicates)
    nmse_mean, low, high = _curve_intervals(
        nmse_values, seed=seed + 10_000, replicates=replicates
    )
    base = _row_base(descriptor, condition)
    return [
        dict(
            base,
            curve_type=curve_type,
            x_name=x_name,
            x_value=position,
            mse=float(mse_mean[position]),
            nmse=float(nmse_mean[position]),
            ci_low=float(low[position]),
            ci_high=float(high[position]),
            n_sequences=int(nmse_values.shape[0]),
        )
        for position in range(nmse_values.shape[1])
    ]


def _binned_task_rows(
    descriptor: dict[str, Any],
    task_mse: np.ndarray,
    task_nmse: np.ndarray,
    x_values: np.ndarray,
    *,
    curve_type: str,
    x_name: str,
    seed: int,
    replicates: int,
) -> list[dict[str, Any]]:
    """Aggregate task-level errors by a structural prefix coordinate."""
    rows: list[dict[str, Any]] = []
    for offset, x_value in enumerate(sorted(int(value) for value in np.unique(x_values))):
        sequence_mse = np.full(task_mse.shape[0], np.nan, dtype=np.float64)
        sequence_nmse = np.full(task_nmse.shape[0], np.nan, dtype=np.float64)
        for sequence in range(task_mse.shape[0]):
            mask = x_values[sequence] == x_value
            if mask.any():
                sequence_mse[sequence] = float(np.nanmean(task_mse[sequence, mask]))
                sequence_nmse[sequence] = float(np.nanmean(task_nmse[sequence, mask]))
        mse_mean, _, _ = _mean_interval(
            sequence_mse, seed=seed + offset, replicates=replicates
        )
        nmse_mean, low, high = _mean_interval(
            sequence_nmse, seed=seed + 1000 + offset, replicates=replicates
        )
        rows.append(
            dict(
                _row_base(descriptor, "ordinary"),
                curve_type=curve_type,
                x_name=x_name,
                x_value=x_value,
                mse=mse_mean,
                nmse=nmse_mean,
                ci_low=low,
                ci_high=high,
                n_sequences=int(np.isfinite(sequence_nmse).sum()),
            )
        )
    return rows


def _append_generalization_gaps(
    summary_rows: list[dict[str, Any]], scalars: dict[str, float]
) -> None:
    """Add interpolation residuals and above-range OOD gaps within matched slices."""
    primary = {
        ("icl", "ordinary", "nmse_aulc"),
        ("composition", "benefit", "benefit_mean"),
        ("retention", "savings", "savings_mean"),
    }
    source = [
        row
        for row in summary_rows
        if (row["capability"], row["condition"], row["metric"]) in primary
        and row["M"] is not None
    ]
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in source:
        key = (row["capability"], row["slice"], row["condition"], row["metric"])
        groups.setdefault(key, []).append(row)
    additions: list[dict[str, Any]] = []
    for (capability, slice_name, _, metric), rows in groups.items():
        seen = sorted((row for row in rows if row["status"] == "seen"), key=lambda row: row["M"])
        if not seen:
            continue
        for row in rows:
            value: float | None = None
            gap_kind = ""
            if row["status"] == "heldout":
                left = [candidate for candidate in seen if candidate["M"] < row["M"]]
                right = [candidate for candidate in seen if candidate["M"] > row["M"]]
                if left and right:
                    low, high = left[-1], right[0]
                    fraction = (row["M"] - low["M"]) / (high["M"] - low["M"])
                    interpolated = low["value"] + fraction * (high["value"] - low["value"])
                    value = float(row["value"] - interpolated)
                    gap_kind = "interpolation_gap"
            elif row["status"] == "ood":
                boundary = seen[-1]
                value = float(row["value"] - boundary["value"])
                gap_kind = "ood_gap"
            if value is None:
                continue
            gap = dict(row)
            gap["condition"] = gap_kind
            gap["metric"] = f"{metric}_{gap_kind}"
            gap["value"] = value
            gap["ci_low"] = value
            gap["ci_high"] = value
            additions.append(gap)
            scalars[
                f"generalization/{capability}/{slice_name}/m{row['M']:02d}/{gap['metric']}"
            ] = value
    summary_rows.extend(additions)


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


def _evaluate_legacy(
    suites: dict[str, Suite], nmses: dict[str, np.ndarray]
) -> EvaluationReport:
    """Legacy fixed-world metric surface."""
    scalars: dict[str, float] = {}
    curves: dict[str, np.ndarray] = {}
    for name, nmse in nmses.items():
        suite = suites[name]
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
    return EvaluationReport(scalars, curves, [], [])


def _sequence_thresholds(values: np.ndarray) -> np.ndarray:
    thresholds = np.empty(values.shape[0], dtype=np.float64)
    for i, curve in enumerate(values):
        valid = curve[np.isfinite(curve)]
        thresholds[i] = _demos_to_threshold(valid)
    return thresholds


def _absolute_capability_metrics(
    name: str,
    descriptor: dict[str, Any],
    suite: Suite,
    mse_values: np.ndarray,
    nmse_values: np.ndarray,
    scalars: dict[str, float],
    curves: dict[str, np.ndarray],
    summary_rows: list[dict[str, Any]],
    curve_rows: list[dict[str, Any]],
    *,
    condition: str,
    seed: int,
    replicates: int,
    curve_type: str,
    mse_last_values: np.ndarray | None = None,
    nmse_last_values: np.ndarray | None = None,
) -> None:
    """Absolute first/last/mean errors for one condition."""
    for prefix, values, last_values in (
        ("mse", mse_values, mse_last_values),
        ("nmse", nmse_values, nmse_last_values),
    ):
        metrics = {
            f"{prefix}_first_demo": values[:, 0],
            f"{prefix}_last_demo": values[:, -1] if last_values is None else last_values,
            f"{prefix}_mean": np.nanmean(values, axis=1),
        }
        if prefix == "nmse":
            metrics["nmse_aulc"] = np.nanmean(values, axis=1)
            metrics["demos_to_threshold"] = _sequence_thresholds(values)
        for offset, (metric, sequence_values) in enumerate(metrics.items()):
            row = _summary_row(
                descriptor,
                condition,
                metric,
                sequence_values,
                seed=seed + offset,
                replicates=replicates,
                suite=suite,
            )
            summary_rows.append(row)
            scalars[f"{name}/{metric}"] = float(row["value"])
    mean_curve = np.nanmean(nmse_values, axis=0)
    curves[f"{name}/{curve_type}"] = mean_curve
    curve_rows.extend(
        _curve_table_rows(
            descriptor,
            condition,
            curve_type,
            mse_values,
            nmse_values,
            x_name="demo_index",
            seed=seed + 100,
            replicates=replicates,
        )
    )


def _evaluate_variable(
    suites: dict[str, Suite],
    mses: dict[str, np.ndarray],
    nmses: dict[str, np.ndarray],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> EvaluationReport:
    scalars: dict[str, float] = {}
    curves: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    descriptors = {name: _suite_descriptor(name, suite) for name, suite in suites.items()}

    icl_names = sorted(name for name, desc in descriptors.items() if desc["capability"] == "icl")
    for suite_index, name in enumerate(icl_names):
        suite, descriptor = suites[name], descriptors[name]
        counts = suite["demo_counts"]
        mse, nmse = mses[name], nmses[name]
        mse_curve_values = np.nanmean(mse, axis=1)
        nmse_curve_values = np.nanmean(nmse, axis=1)
        seed = bootstrap_seed + suite_index * 1000
        _absolute_capability_metrics(
            name,
            descriptor,
            suite,
            mse_curve_values,
            nmse_curve_values,
            scalars,
            curves,
            summary_rows,
            curve_rows,
            condition="ordinary",
            seed=seed,
            replicates=bootstrap_replicates,
            curve_type="learning_curve",
            mse_last_values=_sequence_last_values(mse, counts).mean(axis=1),
            nmse_last_values=_sequence_last_values(nmse, counts).mean(axis=1),
        )
        task_mse = np.nanmean(mse, axis=2)
        task_nmse = np.nanmean(nmse, axis=2)
        curves[f"{name}/task_position_curve"] = np.nanmean(task_nmse, axis=0)
        curve_rows.extend(
            _curve_table_rows(
                descriptor,
                "ordinary",
                "task_position_curve",
                task_mse,
                task_nmse,
                x_name="task_position",
                seed=seed + 500,
                replicates=bootstrap_replicates,
            )
        )
        prefix_coordinates = {
            "history_prediction_tokens": (
                "nmse_by_prediction_tokens_observed",
                "prediction_tokens_observed",
            ),
            "num_unique_supports_seen": (
                "nmse_by_unique_supports_observed",
                "unique_supports_observed",
            ),
            "num_modules_covered": ("nmse_by_modules_covered", "modules_covered"),
        }
        for field, (curve_type, x_name) in prefix_coordinates.items():
            if field in suite:
                curve_rows.extend(
                    _binned_task_rows(
                        descriptor,
                        task_mse,
                        task_nmse,
                        suite[field],
                        curve_type=curve_type,
                        x_name=x_name,
                        seed=seed + 600,
                        replicates=bootstrap_replicates,
                    )
                )
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    for name, descriptor in descriptors.items():
        capability = descriptor["capability"]
        if capability in {"composition", "retention"}:
            grouped.setdefault((capability, _cell_slug(descriptor)), {})[
                descriptor["condition"]
            ] = name

    for group_index, ((capability, cell_slug), conditions) in enumerate(sorted(grouped.items())):
        seed = bootstrap_seed + 100_000 + group_index * 1000
        if capability == "composition":
            missing = {"constituent", "matched_prefix"} - conditions.keys()
            if missing:
                raise ValueError(
                    f"composition cell {cell_slug} is missing conditions {sorted(missing)}"
                )
            constituent_name = conditions["constituent"]
            matched_name = conditions["matched_prefix"]
            descriptor = descriptors[constituent_name]
            constituent_suite = suites[constituent_name]
            constituent_mse = _last_task_values(
                mses[constituent_name], constituent_suite["demo_counts"]
            )
            constituent_nmse = _last_task_values(
                nmses[constituent_name], constituent_suite["demo_counts"]
            )
            matched_suite = suites[matched_name]
            matched_mse = _last_task_values(mses[matched_name], matched_suite["demo_counts"])
            matched_nmse = _last_task_values(nmses[matched_name], matched_suite["demo_counts"])
            _absolute_capability_metrics(
                constituent_name,
                descriptor,
                constituent_suite,
                constituent_mse,
                constituent_nmse,
                scalars,
                curves,
                summary_rows,
                curve_rows,
                condition="constituent",
                seed=seed,
                replicates=bootstrap_replicates,
                curve_type="constituent_curve",
            )
            matched_descriptor = descriptors[matched_name]
            _absolute_capability_metrics(
                matched_name,
                matched_descriptor,
                matched_suite,
                matched_mse,
                matched_nmse,
                scalars,
                curves,
                summary_rows,
                curve_rows,
                condition="matched_prefix",
                seed=seed + 100,
                replicates=bootstrap_replicates,
                curve_type="matched_prefix_curve",
            )

            benefit_mse = matched_mse - constituent_mse
            benefit_nmse = matched_nmse - constituent_nmse
            benefit_name = f"composition/{cell_slug}"
            curves[f"{benefit_name}/benefit_curve"] = np.nanmean(benefit_nmse, axis=0)
            curve_rows.extend(
                _curve_table_rows(
                    descriptor,
                    "benefit",
                    "benefit_curve",
                    benefit_mse,
                    benefit_nmse,
                    x_name="demo_index",
                    seed=seed + 200,
                    replicates=bootstrap_replicates,
                )
            )
            benefit_metrics = {
                "benefit_first_demo": benefit_nmse[:, 0],
                "benefit_last_demo": benefit_nmse[:, -1],
                "benefit_mean": np.nanmean(benefit_nmse, axis=1),
            }
            for offset, (metric, values) in enumerate(benefit_metrics.items()):
                row = _summary_row(
                    descriptor,
                    "benefit",
                    metric,
                    values,
                    seed=seed + 300 + offset,
                    replicates=bootstrap_replicates,
                    suite=constituent_suite,
                )
                summary_rows.append(row)
                scalars[f"{benefit_name}/{metric}"] = float(row["value"])

            if "no_history" in conditions:
                no_history_name = conditions["no_history"]
                no_history_suite = suites[no_history_name]
                no_history_mse = _last_task_values(
                    mses[no_history_name], no_history_suite["demo_counts"]
                )
                no_history_nmse = _last_task_values(
                    nmses[no_history_name], no_history_suite["demo_counts"]
                )
                _absolute_capability_metrics(
                    no_history_name,
                    descriptors[no_history_name],
                    no_history_suite,
                    no_history_mse,
                    no_history_nmse,
                    scalars,
                    curves,
                    summary_rows,
                    curve_rows,
                    condition="no_history",
                    seed=seed + 400,
                    replicates=bootstrap_replicates,
                    curve_type="no_history_curve",
                )
                no_history_benefit = no_history_nmse - constituent_nmse
                row = _summary_row(
                    descriptor,
                    "no_history_benefit",
                    "no_history_benefit_mean",
                    np.nanmean(no_history_benefit, axis=1),
                    seed=seed + 500,
                    replicates=bootstrap_replicates,
                    suite=constituent_suite,
                )
                summary_rows.append(row)
                scalars[f"{benefit_name}/no_history_benefit_mean"] = float(row["value"])

        else:
            missing = {"repeat", "novel"} - conditions.keys()
            if missing:
                raise ValueError(f"retention cell {cell_slug} is missing {sorted(missing)}")
            repeat_name, novel_name = conditions["repeat"], conditions["novel"]
            descriptor = descriptors[repeat_name]
            repeat_suite, novel_suite = suites[repeat_name], suites[novel_name]
            repeat_mse = _last_task_values(mses[repeat_name], repeat_suite["demo_counts"])
            repeat_nmse = _last_task_values(nmses[repeat_name], repeat_suite["demo_counts"])
            novel_mse = _last_task_values(mses[novel_name], novel_suite["demo_counts"])
            novel_nmse = _last_task_values(nmses[novel_name], novel_suite["demo_counts"])
            for offset, (condition, name, suite, mse_values, nmse_values) in enumerate(
                [
                    ("repeat", repeat_name, repeat_suite, repeat_mse, repeat_nmse),
                    ("novel", novel_name, novel_suite, novel_mse, novel_nmse),
                ]
            ):
                _absolute_capability_metrics(
                    name,
                    descriptors[name],
                    suite,
                    mse_values,
                    nmse_values,
                    scalars,
                    curves,
                    summary_rows,
                    curve_rows,
                    condition=condition,
                    seed=seed + offset * 100,
                    replicates=bootstrap_replicates,
                    curve_type=f"{condition}_curve",
                )

            retention_name = f"retention/{cell_slug}"
            savings_mse = novel_mse - repeat_mse
            savings_nmse = novel_nmse - repeat_nmse
            difference_curves = {
                "savings": (savings_mse, savings_nmse),
            }
            if "shared" in conditions:
                shared_name = conditions["shared"]
                shared_suite = suites[shared_name]
                shared_mse = _last_task_values(mses[shared_name], shared_suite["demo_counts"])
                shared_nmse = _last_task_values(
                    nmses[shared_name], shared_suite["demo_counts"]
                )
                _absolute_capability_metrics(
                    shared_name,
                    descriptors[shared_name],
                    shared_suite,
                    shared_mse,
                    shared_nmse,
                    scalars,
                    curves,
                    summary_rows,
                    curve_rows,
                    condition="shared",
                    seed=seed + 200,
                    replicates=bootstrap_replicates,
                    curve_type="shared_curve",
                )
                difference_curves["episodic_savings"] = (
                    shared_mse - repeat_mse,
                    shared_nmse - repeat_nmse,
                )
                difference_curves["module_savings"] = (
                    novel_mse - shared_mse,
                    novel_nmse - shared_nmse,
                )

            for offset, (metric_root, (mse_values, nmse_values)) in enumerate(
                difference_curves.items()
            ):
                curve_key = f"{metric_root}_curve"
                curves[f"{retention_name}/{curve_key}"] = np.nanmean(nmse_values, axis=0)
                curve_rows.extend(
                    _curve_table_rows(
                        descriptor,
                        metric_root,
                        curve_key,
                        mse_values,
                        nmse_values,
                        x_name="demo_index",
                        seed=seed + 300 + offset * 100,
                        replicates=bootstrap_replicates,
                    )
                )
                mean_metric = f"{metric_root}_mean"
                row = _summary_row(
                    descriptor,
                    metric_root,
                    mean_metric,
                    np.nanmean(nmse_values, axis=1),
                    seed=seed + 350 + offset,
                    replicates=bootstrap_replicates,
                    suite=repeat_suite,
                )
                summary_rows.append(row)
                scalars[f"{retention_name}/{mean_metric}"] = float(row["value"])

            scalars[f"{retention_name}/savings_demo0"] = float(np.nanmean(savings_nmse[:, 0]))
            if savings_nmse.shape[1] < 2:
                raise ValueError("retention metrics need at least two revisit demonstrations")
            scalars[f"{retention_name}/savings_one_demo"] = float(
                np.nanmean(savings_nmse[:, 1])
            )
            threshold_delta = _sequence_thresholds(novel_nmse) - _sequence_thresholds(
                repeat_nmse
            )
            row = _summary_row(
                descriptor,
                "savings",
                "demos_to_threshold_delta",
                threshold_delta,
                seed=seed + 900,
                replicates=bootstrap_replicates,
                suite=repeat_suite,
            )
            summary_rows.append(row)
            scalars[f"{retention_name}/demos_to_threshold_delta"] = float(row["value"])

            original_mse = mses[repeat_name][:, 0, : int(repeat_suite["demo_counts"][:, 0].max())]
            original_nmse = nmses[repeat_name][
                :, 0, : int(repeat_suite["demo_counts"][:, 0].max())
            ]
            curves[f"{retention_name}/original_curve"] = np.nanmean(original_nmse, axis=0)
            curves[f"{retention_name}/relearning_curve"] = np.nanmean(repeat_nmse, axis=0)
            curve_rows.extend(
                _curve_table_rows(
                    descriptor,
                    "original",
                    "original_curve",
                    original_mse,
                    original_nmse,
                    x_name="demo_index",
                    seed=seed + 950,
                    replicates=bootstrap_replicates,
                )
            )

    _append_generalization_gaps(summary_rows, scalars)
    return EvaluationReport(scalars, curves, summary_rows, curve_rows)


def evaluate_suites(
    model: GDNModel,
    suites: dict[str, Suite],
    device: torch.device,
    *,
    batch_size: int = 32,
    autocast_dtype: torch.dtype | None = None,
    bootstrap_seed: int = 0,
    bootstrap_replicates: int = 1000,
) -> EvaluationReport:
    """Evaluate frozen suites into scalars, curves, and filterable structural rows."""
    mses: dict[str, np.ndarray] = {}
    nmses: dict[str, np.ndarray] = {}
    for name, suite in suites.items():
        preds = predict_suite(
            model, suite, device, batch_size=batch_size, autocast_dtype=autocast_dtype
        )
        mses[name] = demo_mse(preds, suite)
        nmses[name] = demo_nmse(preds, suite)
    if any("__" in name for name in suites):
        return _evaluate_variable(
            suites,
            mses,
            nmses,
            bootstrap_seed=bootstrap_seed,
            bootstrap_replicates=bootstrap_replicates,
        )
    return _evaluate_legacy(suites, nmses)
