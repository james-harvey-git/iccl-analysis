"""Independent within-world affine ridge fits, selected exclusively on validation trials."""

from __future__ import annotations

import time
from dataclasses import asdict
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from iccl.analysis.probe_config import configure_runtime, resolved_config, validate_probe_config
from iccl.analysis.probe_execution import run_stage
from iccl.analysis.probe_features import FeatureSelection, feature_selections
from iccl.analysis.probe_results import score_predictions
from iccl.analysis.probe_storage import (
    CapturedSplit,
    complete_artifact,
    digest,
    file_digest,
    initialize_artifact,
    lock,
    read_json,
    read_manifest,
    save_torch,
    source_hashes,
    write_json,
)
from iccl.analysis.probe_tracking import WorkerEvents
from iccl.analysis.probe_worlds import CAPTURES, stream_seed
from iccl.analysis.probes import Matrix, fit_ridge, predict_ridge


def check_dataset_config(cfg: DictConfig, manifest: dict[str, Any]) -> None:
    identity = manifest["identity"]
    current = resolved_config(cfg)
    if identity["phase"] != cfg.experiment.phase or identity["seed"] != cfg.experiment.seed:
        raise ValueError("dataset phase/seed differs from the study configuration")
    for key, value in identity["dataset"].items():
        if current["probe"]["dataset"][key] != value:
            raise ValueError(f"dataset.{key} differs from the captured manifest")


def selected_features(cfg: DictConfig, manifest: dict[str, Any]) -> list[FeatureSelection]:
    captured = manifest["identity"]["capture"]
    requested = resolved_config(cfg)["probe"]["training"]["representations"]
    specification = captured["representations"] if requested == "all_captured" else requested
    return feature_selections(captured["shapes"], specification, captured["head_groups"])


def condition_path(root: Path, condition: dict[str, Any]) -> Path:
    return (
        root
        / f"world-{condition['world']:04d}"
        / f"position-{condition['position']:03d}"
        / condition["capture"]
        / condition["family"]
        / condition["selection"]
        / condition["control"]
        / f"n-{condition['train_count']:08d}"
    )


def feature_matrix(
    split: CapturedSplit,
    condition: dict[str, Any],
    selection: FeatureSelection | None,
    indices: np.ndarray | None = None,
) -> Matrix:
    point, control = condition["capture"], condition["control"]
    if control in {"observations", "training_mean"}:
        # The mean predictor ignores this lightweight matrix.
        key = f"observations__{point}" if control == "observations" else "target"
        return split.matrix(key, rows=indices)
    model = "untrained" if control == "untrained" else "trained"
    return split.matrix(f"{model}__{point}__{condition['family']}", selection, indices)


def cpu_state(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.detach().cpu().clone() if isinstance(item, torch.Tensor) else item
        for key, item in value.items()
    }


def checked_bundle(path: Path, checksum: str) -> dict[str, Any]:
    if file_digest(path) != checksum:
        raise ValueError(f"probe checksum mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def fit_condition(
    cfg: DictConfig,
    root: Path,
    condition: dict[str, Any],
    selection: FeatureSelection | None,
    train: Matrix,
    validation: Matrix,
    targets: np.ndarray,
    validation_targets: np.ndarray,
    variance: np.ndarray,
    artifact_id: str,
    device: torch.device,
    events: WorkerEvents,
) -> dict[str, Any]:
    t = resolved_config(cfg)["probe"]["training"]
    folder = condition_path(root, condition)
    folder.mkdir(parents=True, exist_ok=True)
    contract = digest({"artifact_id": artifact_id, "condition": condition})
    final_path = folder / "selected.json"
    if final_path.exists():
        if not t["resume"]:
            raise ValueError("selected probe exists and training.resume is false")
        saved = read_json(final_path)
        if saved["contract"] != contract:
            raise ValueError("selected probe belongs to a different condition")
        checked_bundle(folder / "selected.pt", saved["sha256"])
        return saved
    common = {
        "condition": condition,
        "selection": None if selection is None else asdict(selection),
        "training_variance": variance.tolist(),
        "contract": contract,
    }
    batch_size = t["solver"]["row_batch_size"]
    if condition["control"] == "training_mean":
        bundle = {"kind": "training_mean", "target_mean": targets.mean(axis=0), **common}
        predictions = predict_ridge(bundle, validation, device, batch_size)
        scores = score_predictions(validation_targets, predictions, variance)
        save_torch(folder / "selected.pt", bundle)
        saved = common | {
            "sha256": file_digest(folder / "selected.pt"),
            "validation": scores,
            "regularizer": None,
            "candidates": [],
            "features": 0,
        }
        write_json(final_path, saved)
        return saved

    monitoring = t["monitoring"]
    rng = np.random.default_rng(
        stream_seed(cfg.experiment.seed, f"monitor/{condition['world']}/{condition['position']}")
    )
    train_ids = np.sort(rng.permutation(len(targets))[: monitoring["train_subset_size"]])
    val_ids = np.sort(
        rng.permutation(len(validation_targets))[: monitoring["validation_subset_size"]]
    )
    candidates = []
    callback_every = reduce(
        gcd,
        [
            monitoring[k]
            for k in (
                "log_every_iterations",
                "validation_every_iterations",
                "checkpoint_every_iterations",
            )
        ],
    )
    for regularizer in t["l2_grid"]:
        candidate_id = digest({"contract": contract, "regularizer": regularizer})
        candidate_dir = folder / f"lambda-{regularizer}"
        candidate_dir.mkdir(exist_ok=True)
        result_path, checkpoint_path = candidate_dir / "result.json", candidate_dir / "iterate.pt"
        if result_path.exists():
            saved = read_json(result_path)
            if saved["candidate_id"] != candidate_id or not t["resume"]:
                raise ValueError("incompatible completed candidate or resume disabled")
            checked_bundle(candidate_dir / "probe.pt", saved["sha256"])
            candidates.append(saved)
            continue
        fit_id = (
            f"w{condition['world']}/p{condition['position']}/{condition['capture']}/"
            f"{condition['family']}/{condition['selection']}/{condition['control']}/"
            f"n{condition['train_count']}/lambda{regularizer:g}"
        )
        started = time.perf_counter()
        resume = None
        if checkpoint_path.exists():
            if not t["resume"]:
                raise ValueError("partial ridge iterate exists and resume is disabled")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if state["candidate_id"] != candidate_id:
                raise ValueError("partial ridge iterate belongs to a different fit")
            resume = state["iterate"]

        def monitor(
            diagnostics: dict[str, Any],
            state: dict[str, Any],
            *,
            fit_id: str = fit_id,
            started: float = started,
            checkpoint_path: Path = checkpoint_path,
            candidate_id: str = candidate_id,
        ) -> None:
            iteration = diagnostics["iteration"]
            final = diagnostics["final"]
            record = {
                "phase": cfg.experiment.phase,
                "fit_id": fit_id,
                **condition,
                **diagnostics,
                "seconds": time.perf_counter() - started,
                "train_monitor_trials": len(train_ids),
                "validation_monitor_trials": len(val_ids),
            }
            if final or iteration % monitoring["log_every_iterations"] == 0:
                prediction = predict_ridge(state, train, device, batch_size, train_ids)
                scores = score_predictions(targets[train_ids], prediction, variance)
                record.update(train_mse=scores["mse"], train_r2=scores["r2_macro"])
            if final or iteration % monitoring["validation_every_iterations"] == 0:
                prediction = predict_ridge(state, validation, device, batch_size, val_ids)
                scores = score_predictions(validation_targets[val_ids], prediction, variance)
                record.update(validation_mse=scores["mse"], validation_r2=scores["r2_macro"])
            events.emit("fit_trace", **record)
            if (
                state["method"] == "cg"
                and iteration > 0
                and not final
                and iteration % monitoring["checkpoint_every_iterations"] == 0
            ):
                save_torch(
                    checkpoint_path, {"candidate_id": candidate_id, "iterate": cpu_state(state)}
                )

        bundle, diagnostics = fit_ridge(
            train,
            targets,
            regularizer,
            device=device,
            options=t["solver"],
            log_every=callback_every,
            callback=monitor,
            resume=resume,
        )
        if not diagnostics["converged"] and t["solver"]["require_convergence"]:
            raise RuntimeError(f"ridge did not converge: {fit_id}; {diagnostics}")
        prediction = predict_ridge(bundle, validation, device, batch_size)
        scores = score_predictions(validation_targets, prediction, variance)
        bundle.update(common)
        save_torch(candidate_dir / "probe.pt", bundle)
        saved = {
            "candidate_id": candidate_id,
            "regularizer": regularizer,
            "diagnostics": diagnostics,
            "validation": scores,
            "sha256": file_digest(candidate_dir / "probe.pt"),
            "seconds": time.perf_counter() - started,
        }
        write_json(result_path, saved)
        checkpoint_path.unlink(missing_ok=True)
        events.emit(
            "fit_complete",
            fit_id=fit_id,
            **condition,
            **diagnostics,
            validation_full_r2=scores["r2_macro"],
            seconds=saved["seconds"],
            features=train.shape[1],
        )
        candidates.append(saved)
    eligible = [
        item
        for item in candidates
        if item["diagnostics"]["converged"] and item["validation"]["r2_macro"] is not None
    ]
    if not eligible:
        raise ValueError("no converged candidate has defined full-validation macro R2")
    # A validation tie selects the larger regularizer deterministically.
    best = max(eligible, key=lambda item: (item["validation"]["r2_macro"], item["regularizer"]))
    bundle = checked_bundle(folder / f"lambda-{best['regularizer']}" / "probe.pt", best["sha256"])
    save_torch(folder / "selected.pt", bundle)
    saved = common | {
        "sha256": file_digest(folder / "selected.pt"),
        "regularizer": best["regularizer"],
        "validation": best["validation"],
        "candidates": candidates,
        "features": train.shape[1],
    }
    write_json(final_path, saved)
    return saved


def train_probe(cfg: DictConfig, out_dir: Path | str) -> dict[str, Any]:
    validate_probe_config(cfg, "train")
    device = configure_runtime(cfg)
    t = cfg.probe.training
    dataset = read_manifest(t.dataset)
    check_dataset_config(cfg, dataset)
    features = selected_features(cfg, dataset)
    settings = resolved_config(cfg)["probe"]["training"]
    settings = {k: v for k, v in settings.items() if k not in {"dataset", "output", "resume"}}
    root = Path(t.output)
    artifact = initialize_artifact(
        root,
        "probes",
        {
            "dataset_id": dataset["dataset_id"],
            "settings": settings,
            "phase": cfg.experiment.phase,
            "seed": cfg.experiment.seed,
            "solver_device_type": device.type,
            "implementation": source_hashes(
                "probes.py", "probe_training.py", "probe_results.py", "probe_features.py"
            ),
        },
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def worker(world: int, events: WorkerEvents) -> None:
        folder = root / f"world-{world:04d}"
        with lock(folder / ".training.lock", blocking=False):
            selected = []
            for position in cfg.probe.dataset.target_positions:
                train = CapturedSplit(t.dataset, world, position, "train")
                validation = CapturedSplit(t.dataset, world, position, "validation")
                try:
                    variance = train.targets.astype(np.float64).var(axis=0)
                    nested = np.random.default_rng(
                        stream_seed(cfg.experiment.seed, f"budget/{world}/{position}")
                    ).permutation(train.count)
                    budgets = sorted({train.count, *list(t.sample_sizes)})
                    for count in budgets:
                        ids = np.sort(nested[:count])
                        targets = train.targets[ids]
                        if np.linalg.matrix_rank(targets - targets.mean(axis=0)) < targets.shape[1]:
                            raise ValueError(
                                "nested training subset lacks full affine target variation"
                            )
                        permutation = np.random.default_rng(
                            stream_seed(t.shuffled_seed, f"shuffle/{world}/{position}/{count}")
                        ).permutation(count)
                        for capture in CAPTURES:
                            conditions: list[tuple[str, FeatureSelection | None]] = []
                            for selection in features:
                                conditions.append(("trained", selection))
                                conditions.extend(
                                    (control, selection)
                                    for control in ("shuffled_targets", "untrained")
                                    if control in t.controls
                                )
                            conditions.extend(
                                (control, None)
                                for control in ("training_mean", "observations")
                                if control in t.controls
                            )
                            for control, selection in conditions:
                                condition = {
                                    "phase": cfg.experiment.phase,
                                    "world": world,
                                    "position": position,
                                    "capture": capture,
                                    "family": selection.family if selection else "baseline",
                                    "selection": selection.name if selection else control,
                                    "control": control,
                                    "train_count": count,
                                }
                                x = feature_matrix(train, condition, selection, ids)
                                val_x = feature_matrix(validation, condition, selection)
                                z = (
                                    targets[permutation]
                                    if control == "shuffled_targets"
                                    else targets
                                )
                                item = fit_condition(
                                    cfg,
                                    root,
                                    condition,
                                    selection,
                                    x,
                                    val_x,
                                    z,
                                    validation.targets,
                                    variance,
                                    artifact["artifact_id"],
                                    device,
                                    events,
                                )
                                relative = condition_path(root, condition).relative_to(root)
                                selected.append({"path": str(relative), **item})
                finally:
                    train.close()
                    validation.close()
            write_json(
                folder / "complete.json",
                {"artifact_id": artifact["artifact_id"], "selected": selected},
            )

    def finalize() -> dict[str, Any]:
        completed = complete_artifact(root, "probes", cfg.probe.dataset.num_worlds)
        return {"probes_path": str(root.resolve()), "probes_id": completed["artifact_id"]}

    return run_stage(cfg, "train", worker, finalize)
