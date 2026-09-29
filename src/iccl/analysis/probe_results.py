"""Portable, versioned probe checkpoints and numerical results."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from iccl.analysis.probe_dataset import CapturedDataset, file_digest, write_json
from iccl.analysis.probe_targets import PROTOCOL, TARGET_LAYOUT
from iccl.checkpoints import SourceRun

CHECKPOINT_VERSION = f"{PROTOCOL}/checkpoint-v1"
RESULT_VERSION = f"{PROTOCOL}/results-v1"


def probe_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten saved episode-level estimates without recomputing their confidence intervals."""
    rows = []
    for prediction, populations in summary.items():
        for population, report in populations.items():
            for metric, estimate in report["metrics"].items():
                by_task = isinstance(estimate["mean"], list)
                means = estimate["mean"] if by_task else [estimate["mean"]]
                for index, mean in enumerate(means):
                    row = {
                        "prediction": prediction,
                        "population": population,
                        "metric": metric,
                        "task_position": index + 1 if by_task else None,
                        "exposure_count": None,
                        "n_modules": None,
                        "mean": float(mean),
                        "n_episodes": int(estimate["n_episodes"]),
                        "variance_floored_tasks": int(report["variance_floored_tasks"]),
                    }
                    for key in ("ci_low", "ci_high"):
                        value = estimate[key]
                        row[key] = (
                            None if value is None else float(value[index] if by_task else value)
                        )
                    rows.append(row)
            for exposure, estimate in report["exposure"].items():
                rows.append(
                    {
                        "prediction": prediction,
                        "population": population,
                        "metric": "module_mse",
                        "task_position": None,
                        "exposure_count": int(exposure),
                        "mean": estimate["mean"],
                        "ci_low": estimate["ci_low"],
                        "ci_high": estimate["ci_high"],
                        "n_episodes": estimate["n_episodes"],
                        "n_modules": estimate["n_modules"],
                        "variance_floored_tasks": None,
                    }
                )
    return rows


def save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    """Atomically replace one local checkpoint, including its optimizer when supplied."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(checkpoint, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path: Path | str, dataset: CapturedDataset) -> dict[str, Any]:
    checkpoint = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=False)
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(
            "expected a module-set-decoder checkpoint; capture fresh M=4 episodes and retrain "
            "for old task-occurrence checkpoints, or supply a probe rather than GDN checkpoint"
        )
    identity = dataset.manifest["identity"]
    if (
        checkpoint.get("dataset_id") != dataset.manifest["dataset_id"]
        or checkpoint.get("state_layout") != identity["state_layout"]
        or checkpoint.get("target_layout") != TARGET_LAYOUT
        or checkpoint.get("source_model_digest") != identity["source_model_digest"]
    ):
        raise ValueError("probe checkpoint and captured dataset/layout are incompatible")
    if checkpoint.get("control") not in {"none", "constant", "shuffled_targets"}:
        raise ValueError("invalid decoder kind in checkpoint")
    weight = checkpoint.get("readout_weight")
    if not isinstance(weight, (float, int)) or not np.isfinite(weight) or weight <= 0:
        raise ValueError("invalid readout loss weight in checkpoint")
    if dataset.split in checkpoint["split_signatures"] and (
        dataset.signature != checkpoint["split_signatures"][dataset.split]
    ):
        raise ValueError("captured split changed since this probe checkpoint was trained")
    return checkpoint


def source_run(manifest: dict[str, Any]) -> SourceRun | None:
    reference = manifest.get("source_provenance", {}).get("source_run")
    return None if reference is None else SourceRun(**reference)


def write_results(
    directory: Path | str,
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> Path:
    """Publish a complete report in a fresh directory; never overwrite another evaluation."""
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"probe results directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".episodes-", dir=directory)
    path = directory / "episodes.npz"
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # pyright: ignore[reportArgumentType]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    write_json(directory / "summary.json", summary)
    write_json(
        directory / "manifest.json",
        {
            "version": RESULT_VERSION,
            "metadata": metadata,
            "files": {
                name: file_digest(directory / name) for name in ("episodes.npz", "summary.json")
            },
        },
    )
    return directory


def read_results(
    directory: Path | str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("version") != RESULT_VERSION:
        raise ValueError(
            "incompatible module-set-decoder result protocol; recapture/retrain and evaluate "
            "the M=4 set decoder before using this plotting pipeline"
        )
    if set(manifest["files"]) != {"episodes.npz", "summary.json"}:
        raise ValueError("incomplete probe result manifest")
    for name, checksum in manifest["files"].items():
        if file_digest(directory / name) != checksum:
            raise ValueError(f"probe result checksum mismatch: {name}")
    with np.load(directory / "episodes.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return manifest["metadata"], arrays, json.loads((directory / "summary.json").read_text())


def require_comparable(reports: list[dict[str, Any]]) -> None:
    """Refuse comparison plots when their held-out populations or scoring rules differ."""
    keys = (
        "protocol",
        "dataset_id",
        "split_signature",
        "readout_weight",
        "target_layout",
        "evaluation_precision",
        "functional_inputs_per_task",
        "functional_seed",
        "functional_stream_seed",
        "variance_floor",
    )
    if any(any(report[key] != reports[0][key] for key in keys) for report in reports[1:]):
        raise ValueError("comparison requires identical held-out worlds and evaluation settings")
