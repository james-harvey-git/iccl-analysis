"""Evaluation metrics for ICCL, computed on the frozen eval suites.

The universal unit is **normalized MSE**: per-position squared error divided by
that task's output variance (``base_mse``), so 1.0 means predicting the mean
and 0 means perfect. Everything derives from the per-(sequence, task, demo)
normalized-MSE array: in-context learning curves (error vs demo index),
task-position curves (forward transfer within a context), the composite-vs-
control benefit of history, and retention on revisit suites (the revisit block
paired demo-by-demo against the same task's original block — a savings
measure, plus the forgetting gap to the original block's final error).

The generic learning and task-position curves belong to the pure-curriculum
suites (in_dist, structural_*). Suites built around a special final task report
that task directly: composite's ``learning_curve`` is the novel composition's
few-shot curve (with history) against composite_control's (without), and
retention reports its revisit through the paired original/relearning/savings
curves — their curriculum tasks duplicate in_dist, so no generic curve is
emitted for them.
"""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from iccl.data.export import load_suite
from iccl.models.model import GDNModel

Suite = dict[str, np.ndarray]

BASE_MSE_FLOOR = 1e-12  # guards the normalization against degenerate tasks


def load_eval_suites(out_dir: Path) -> dict[str, Suite]:
    """Load every frozen suite in ``out_dir``, keyed by suite name."""
    paths = sorted(out_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no frozen eval suites found in {out_dir}; "
            "generate them once with `uv run python scripts/make_eval_sets.py`"
        )
    return {path.stem: load_suite(path.with_suffix("")) for path in paths}


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


def _curve_scalars(curve: Float[np.ndarray, " demos"]) -> dict[str, float]:
    """First-demo, last-demo, and mean normalized MSE of a per-demo curve."""
    return {
        "nmse_first_demo": float(curve[0]),
        "nmse_last_demo": float(curve[-1]),
        "nmse_mean": float(np.nanmean(curve)),
    }


def retention_metrics(
    nmse: Float[np.ndarray, "seqs tasks max_demos"],
    counts: Float[np.ndarray, "seqs tasks"],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Retention metrics from the paired blocks of the revisited task: the last
    span re-demonstrates the first curriculum task, so demo j of the revisit
    compares directly against demo j of the original block (same task, same
    world, same sequence — a savings measure). ``savings_curve`` is original
    minus revisit error per demo: zero everywhere means complete forgetting,
    while retention shows as large savings at demo 0 (the revisit starting near
    ``original_last_demo``, which is what ``forgetting`` measures)."""
    num_seqs = counts.shape[0]
    original, revisit = nmse[:, 0, :], nmse[:, -1, :]
    original_curve = np.nanmean(original[:, : int(counts[:, 0].max())], axis=0)
    revisit_curve = _last_task_curve(nmse, counts)
    paired = min(original_curve.shape[0], revisit_curve.shape[0])
    revisit_first = float(np.nanmean(revisit[:, 0]))
    original_last = float(np.mean(original[np.arange(num_seqs), counts[:, 0] - 1]))
    scalars = {
        "revisit_first_demo": revisit_first,
        "original_last_demo": original_last,
        "forgetting": revisit_first - original_last,
        "savings_first_demo": float(np.nanmean(original[:, 0] - revisit[:, 0])),
    }
    curves = {
        "original_curve": original_curve,
        "relearning_curve": revisit_curve,
        "savings_curve": original_curve[:paired] - revisit_curve[:paired],
    }
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
        paired = min(composite_curve.shape[0], control_curve.shape[0])
        benefit_curve = control_curve[:paired] - composite_curve[:paired]
        curves["composite/benefit_curve"] = benefit_curve
        scalars["composite/benefit_first_demo"] = float(benefit_curve[0])
        scalars["composite/benefit_last_demo"] = float(benefit_curve[-1])
    if "retention" in nmses:
        retention_s, retention_c = retention_metrics(
            nmses["retention"], suites["retention"]["demo_counts"]
        )
        for key, value in retention_s.items():
            scalars[f"retention/{key}"] = value
        for key, curve in retention_c.items():
            curves[f"retention/{key}"] = curve
    return scalars, curves
