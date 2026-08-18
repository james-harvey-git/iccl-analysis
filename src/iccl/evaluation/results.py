"""Portable numerical records for full frozen-suite evaluations."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from iccl.evaluation.metrics import EvaluationReport

SUMMARY_COLUMNS = [
    "step",
    "capability",
    "condition",
    "slice",
    "variant",
    "suite",
    "status",
    "sampler",
    "weighting",
    "M",
    "T",
    "S",
    "B_history",
    "L_history",
    "D_target",
    "D_min",
    "D_max",
    "D_mean",
    "D_cv",
    "constituent_exposure_min",
    "constituent_exposure_max",
    "constituent_demo_exposure_min",
    "constituent_demo_exposure_max",
    "intervening_tasks",
    "prediction_token_delay",
    "serialized_token_delay",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n_sequences",
]

CURVE_COLUMNS = [
    "step",
    "capability",
    "condition",
    "slice",
    "variant",
    "status",
    "M",
    "T",
    "S",
    "B_history",
    "L_history",
    "curve_type",
    "x_name",
    "x_value",
    "mse",
    "nmse",
    "ci_low",
    "ci_high",
    "n_sequences",
]


def _write_rows(path: Path, columns: list[str], rows: list[dict[str, Any]], step: int) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(dict(row, step=step) for row in rows)


def write_evaluation_results(
    report: EvaluationReport,
    out_dir: Path,
    step: int,
    manifest: dict[str, Any],
) -> Path:
    """Write aggregate rows and per-demo errors for one checkpoint."""
    path = out_dir / f"step_{step:07d}"
    path.mkdir(parents=True, exist_ok=True)
    _write_rows(path / "summary.csv", SUMMARY_COLUMNS, report.summary_rows, step)
    _write_rows(path / "curves.csv", CURVE_COLUMNS, report.curve_rows, step)
    (path / "scalars.json").write_text(json.dumps(report.scalars, indent=2, sort_keys=True))

    raw_names = {key.replace("/", "."): key for key in report.raw_errors}
    np.savez_compressed(
        path / "raw_errors.npz",
        **{  # pyright: ignore[reportArgumentType]
            stored: report.raw_errors[original] for stored, original in raw_names.items()
        },
    )
    payload = dict(manifest, step=step, raw_arrays=raw_names)
    (path / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read one artifact CSV and restore numeric columns used by plotting."""
    integer_fields = {"step", "M", "T", "S", "B_history", "L_history", "x_value", "n_sequences"}
    float_fields = {
        "D_target",
        "D_min",
        "D_max",
        "D_mean",
        "D_cv",
        "constituent_exposure_min",
        "constituent_exposure_max",
        "constituent_demo_exposure_min",
        "constituent_demo_exposure_max",
        "intervening_tasks",
        "prediction_token_delay",
        "serialized_token_delay",
        "value",
        "mse",
        "nmse",
        "ci_low",
        "ci_high",
    }
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in integer_fields & row.keys():
            if row[field]:
                row[field] = int(row[field])
        for field in float_fields & row.keys():
            if row[field]:
                row[field] = float(row[field])
            else:
                row[field] = None
    return rows
