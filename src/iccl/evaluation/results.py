"""Portable numerical records for frozen capability evaluations."""

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from iccl.checkpoints import (
    checkpoint_model_config,
    checkpoint_model_digest,
    source_from_checkpoint,
)
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
    "protocol",
    "sample_scope",
    "exposure_scope",
    "n_episodes",
]
RETENTION_COLUMNS = ["retention_component", "original_task_position", "intervening_tasks"]
REHEARSAL_COLUMNS = ["rehearsal_mode", "support_status"]
SUMMARY_COLUMNS = [
    "step",
    "checkpoint_reference",
    *STRUCTURAL_COLUMNS,
    "capability",
    "condition",
    "metric",
    *RETENTION_COLUMNS,
    *REHEARSAL_COLUMNS,
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
    *REHEARSAL_COLUMNS,
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
        result_files={name: _digest(path / name) for name in RESULT_FILES},
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
        "n_episodes",
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


RESULT_FILES = ("curves.csv", "summary.csv", "raw_errors.npz", "scalars.json")


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def evaluation_identity(
    checkpoint: dict[str, Any],
    suites: dict[str, Any],
    bundle: dict[str, Any],
    *,
    batch_size: int,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    backend: str,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, Any]:
    """Numerical cache identity independent of the suite-selection entry point."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(
        p for folder in ("models", "evaluation") for p in (root / folder).glob("*.py")
    ):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    source = source_from_checkpoint(checkpoint)
    return {
        "schema": 2,
        "sample_scope": "full",
        "metric_version": METRIC_VERSION,
        "source_run": None
        if source is None
        else f"{source.entity}/{source.project}/{source.run_id}",
        "step": int(checkpoint["step"]),
        "model_sha256": checkpoint_model_digest(checkpoint),
        "implementation_sha256": digest.hexdigest(),
        "architecture": checkpoint_model_config(checkpoint),
        "suite_files": {
            f"{name}{ext}": bundle["files"][f"{name}{ext}"]
            for name in suites
            for ext in (".npz", ".meta.json")
        },
        "selections": {
            name: suite["__meta__"].get("selected_indices") for name, suite in suites.items()
        },
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "batch_size": batch_size,
        "backend": backend,
        "device_type": device.type,
        "precision": str(dtype),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
    }


def validate_cached_results(
    path: Path, expected: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Validate complete numerical artifacts and optionally a requested suite subset."""
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.get("evaluation_identity")
    if (
        manifest.get("metric_version") != METRIC_VERSION
        or not identity
        or identity.get("schema") != 2
    ):
        raise ValueError(
            f"Incompatible cached evaluation protocol in {path}; use a fresh output directory"
        )
    if identity.get("sample_scope") != "full" or any(
        v is not None for v in identity.get("selections", {}).values()
    ):
        raise ValueError(f"Incomplete full evaluation: monitor-only cache in {path}")
    if expected is not None:
        for key, value in expected.items():
            if key in {"suite_files", "selections"}:
                if not value.keys() <= identity.get(key, {}).keys():
                    raise ValueError(f"Incomplete results in {path}: missing requested suites")
                matches = all(identity[key][name] == item for name, item in value.items())
            else:
                matches = identity.get(key) == value
            if not matches:
                raise ValueError(
                    f"Incompatible cached evaluation ({key}) in {path}; "
                    "use a fresh output directory"
                )
    checksums = manifest.get("result_files", {})
    if set(checksums) != set(RESULT_FILES):
        raise ValueError(f"Incomplete cached evaluation in {path}")
    for name, checksum in checksums.items():
        if not (path / name).is_file() or _digest(path / name) != checksum:
            raise ValueError(f"Cached result checksum mismatch: {path / name}")
    return manifest


def read_evaluation_results(
    path: Path, expected: dict[str, Any], suites: dict[str, Any]
) -> EvaluationReport | None:
    """Reuse a verified full report or a compatible scientific-suite subset."""
    manifest = validate_cached_results(path, expected)
    if manifest is None:
        return None
    summaries = [r for r in read_rows(path / "summary.csv") if r["suite"] in suites]
    curves = [r for r in read_rows(path / "curves.csv") if r["suite"] in suites]
    prefixes = tuple(
        f"{s['__meta__']['capability']}/{s['__meta__'].get('cell_id', '')}" for s in suites.values()
    )
    scalar_values = json.loads((path / "scalars.json").read_text())
    scalars = {k: v for k, v in scalar_values.items() if k.startswith(prefixes)}
    with np.load(path / "raw_errors.npz", allow_pickle=False) as arrays:
        raw = {
            meta["logical_name"]: arrays[name]
            for name, meta in manifest["raw_arrays"].items()
            if meta["logical_name"].startswith((*prefixes, *(name + "/" for name in suites)))
        }
    return EvaluationReport(scalars, {}, summaries, curves, raw)
