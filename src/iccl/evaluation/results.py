"""Portable numerical records for frozen capability evaluations."""

import csv
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from iccl.evaluation.metrics import METRIC_DEFINITIONS, METRIC_VERSION, EvaluationReport

STRUCTURAL_COLUMNS = [
    "family_memberships",
    "cell_id",
    "suite",
    "module_count_status",
    "sampler",
    "weighting",
    "M",
    "T",
    "S",
    "D",
]
RETENTION_COLUMNS = ["retention_component", "original_task_position", "intervening_tasks"]
DIAGNOSTIC_COLUMNS = ["diagnostic_family", "rehearsal_mode", "support_status"]
SUMMARY_COLUMNS = [
    "step",
    "checkpoint_reference",
    *STRUCTURAL_COLUMNS,
    "capability",
    "condition",
    "metric",
    *RETENTION_COLUMNS,
    *DIAGNOSTIC_COLUMNS,
    "value",
    "ci_low",
    "ci_high",
    "n_sequences",
]
CURVE_COLUMNS = [
    "step",
    "checkpoint_reference",
    *STRUCTURAL_COLUMNS,
    "capability",
    "condition",
    "curve_type",
    *RETENTION_COLUMNS,
    *DIAGNOSTIC_COLUMNS,
    "x_name",
    "x_value",
    "mse",
    "nmse",
    "ci_low",
    "ci_high",
    "n_sequences",
]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_rows(
    path: Path,
    columns: list[str],
    rows: list[dict[str, Any]],
    step: int,
    checkpoint_reference: str,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(
            dict(row, step=step, checkpoint_reference=checkpoint_reference) for row in rows
        )


def write_evaluation_results(
    report: EvaluationReport,
    out_dir: Path,
    step: int,
    manifest: dict[str, Any],
) -> Path:
    """Write aggregate rows, raw arrays and interpretation metadata."""
    path = out_dir / f"step_{step:07d}"
    path.mkdir(parents=True, exist_ok=True)
    reference = str(manifest.get("checkpoint_reference", ""))
    _write_rows(path / "summary.csv", SUMMARY_COLUMNS, report.summary_rows, step, reference)
    _write_rows(path / "curves.csv", CURVE_COLUMNS, report.curve_rows, step, reference)
    (path / "scalars.json").write_text(json.dumps(report.scalars, indent=2, sort_keys=True))

    stored_names = {key.replace("/", "."): key for key in report.raw_errors}
    np.savez_compressed(
        path / "raw_errors.npz",
        **{  # pyright: ignore[reportArgumentType]
            stored: report.raw_errors[original] for stored, original in stored_names.items()
        },
    )
    raw_arrays = {
        stored: {
            "logical_name": original,
            "shape": list(report.raw_errors[original].shape),
            "dtype": str(report.raw_errors[original].dtype),
        }
        for stored, original in stored_names.items()
    }
    payload = dict(
        manifest,
        step=step,
        git_commit=_git_commit(),
        metric_version=METRIC_VERSION,
        metric_definitions=METRIC_DEFINITIONS,
        raw_arrays=raw_arrays,
        runtime={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    )
    (path / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read one artifact CSV and restore numeric columns used by plotting."""
    integer_fields = {
        "step",
        "M",
        "T",
        "S",
        "D",
        "x_value",
        "n_sequences",
        "original_task_position",
        "intervening_tasks",
    }
    float_fields = {"value", "mse", "nmse", "ci_low", "ci_high"}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in integer_fields & row.keys():
            row[field] = int(row[field]) if row[field] else None
        for field in float_fields & row.keys():
            row[field] = float(row[field]) if row[field] else None
    return rows
