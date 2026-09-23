"""Held-out reconstruction with one alignment per episode and episode-level uncertainty."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from iccl.analysis.probe_config import (
    configure_runtime,
    resolved_config,
    stream_seed,
    validate_probe_config,
)
from iccl.analysis.probe_dataset import CapturedDataset, file_digest
from iccl.analysis.probe_loss import aligned_targets, parameter_errors
from iccl.analysis.probe_matching import Assignment, AssignmentSolver
from iccl.analysis.probe_results import (
    TARGET_LAYOUT,
    load_checkpoint,
    probe_summary_rows,
    source_run,
    write_results,
)
from iccl.analysis.probe_targets import MODULE_FEATURES, MODULE_SHAPE, PROTOCOL
from iccl.analysis.probe_training import probe_loader
from iccl.analysis.probes import make_decoder
from iccl.data.dataset import sequence_rng
from iccl.data.teacher import ModulePool, teacher_forward
from iccl.evaluation.metrics import BASE_MSE_FLOOR
from iccl.reporting.logger import RunLogger
from iccl.training.trainer import resolve_autocast_dtype


def assigned_module_ids(module_ids: np.ndarray, assignment: Assignment) -> np.ndarray:
    """True module IDs associated with each predicted task/slot after the chosen swaps."""
    bits = (assignment.masks[:, None] >> np.arange(8)) & 1
    pairs = bits[:, :, None] ^ np.arange(2)
    return np.take_along_axis(module_ids, pairs, axis=2)


def repeated_module_error(
    predictions: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean within-module occurrence variance, equally weighted over repeated module IDs.

    All occurrences already share the decoder's hidden basis. This diagnostic is
    zero for a constant prediction and therefore does not establish reconstruction.
    """
    modules = predictions[:, :MODULE_FEATURES].reshape(-1, *MODULE_SHAPE).astype(np.float64)
    error = np.zeros(len(predictions), np.float64)
    counts = np.zeros(len(predictions), np.int64)
    for episode in range(len(predictions)):
        values = []
        for module in np.unique(ids[episode]):
            occurrences = modules[episode][ids[episode] == module]
            if len(occurrences) >= 2:
                values.append(float(np.mean((occurrences - occurrences.mean(axis=0)) ** 2)))
        if not values:
            raise ValueError("the fixed eight-module/eight-task protocol requires repeated modules")
        error[episode], counts[episode] = np.mean(values), len(values)
    return error, counts


def functional_reconstruction(
    predictions: np.ndarray,
    batch: dict[str, np.ndarray],
    assignment: Assignment,
    *,
    inputs_per_task: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Oracle-coefficient reconstruction; parameter matching alone chooses the alignment."""
    ids = assigned_module_ids(batch["module_ids"], assignment)
    coefficients = np.take_along_axis(batch["latents"], ids, axis=2) / np.float32(math.sqrt(2))
    modules = predictions[:, :MODULE_FEATURES].reshape(-1, *MODULE_SHAPE)
    composed = np.einsum("btsih,bts->btih", modules, coefficients)
    readout = predictions[:, MODULE_FEATURES:].reshape(-1, 16, 16)
    mse = np.empty((len(predictions), 8), np.float64)
    variance = np.empty_like(mse)
    for episode, index in enumerate(batch["episode_index"]):
        inputs = (
            sequence_rng(seed, int(index))
            .uniform(-1, 1, size=(8, inputs_per_task, 16))
            .astype(np.float32)
        )
        pool = ModulePool(
            [batch["world_modules"][episode]],
            [batch["world_biases"][episode]],
            batch["world_readout"][episode],
        )
        for task in range(8):
            original = teacher_forward(pool, batch["latents"][episode, task], inputs[task]).astype(
                np.float64
            )
            hidden = np.maximum(
                inputs[task] @ composed[episode, task, :16] + composed[episode, task, 16], 0
            )
            decoded = (hidden @ readout[episode]).astype(np.float64)
            if not np.isfinite(decoded).all():
                raise FloatingPointError("nonfinite oracle-coefficient functional reconstruction")
            mse[episode, task] = ((decoded - original) ** 2).mean()
            variance[episode, task] = ((original - original.mean(axis=0)) ** 2).mean()
    return {
        "functional_mse_by_task": mse,
        "functional_nmse_by_task": mse / np.maximum(variance, BASE_MSE_FLOOR),
        "functional_output_variance": variance,
        "functional_variance_floored": variance < BASE_MSE_FLOOR,
    }


def score_predictions(
    predictions: np.ndarray,
    batch: dict[str, np.ndarray],
    solver: AssignmentSolver,
    *,
    readout_weight: float,
    inputs_per_task: int,
    functional_seed: int,
) -> dict[str, np.ndarray]:
    assignment = solver.match(predictions, batch["target"], readout_weight)
    target = aligned_targets(torch.from_numpy(batch["target"]), assignment)
    errors = parameter_errors(torch.from_numpy(predictions), target, readout_weight)
    ids = assigned_module_ids(batch["module_ids"], assignment)
    consistency, counts = repeated_module_error(predictions, ids)
    return {
        "joint_mse": errors.joint.numpy(),
        "weight_mse": errors.weights.numpy(),
        "bias_mse": errors.biases.numpy(),
        "readout_mse": errors.readout.numpy(),
        "module_mse_by_task": errors.by_task.numpy(),
        "repeated_module_mse": consistency,
        "repeated_module_count": counts,
        "swap_mask": assignment.masks,
        "hidden_permutation": assignment.permutations,
        "assigned_module_ids": ids,
        "assignment_score": assignment.scores,
        "matching_counters": assignment.counters,
        **functional_reconstruction(
            predictions, batch, assignment, inputs_per_task=inputs_per_task, seed=functional_seed
        ),
    }


def episode_interval(values: np.ndarray, *, seed: int, replicates: int) -> dict[str, Any]:
    """Resample rows, preserving each episode's complete vector of task-position errors."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("confidence intervals require nonempty, finite per-episode measurements")
    result = {
        "mean": values.mean(axis=0).tolist(),
        "n_episodes": len(values),
        "ci_low": None,
        "ci_high": None,
    }
    if replicates >= 2 and len(values) >= 2:
        rng = sequence_rng(seed, len(values))
        draws = np.empty((replicates, *values.shape[1:]), np.float64)
        # Bound temporary memory when the held-out population is large.
        for start in range(0, replicates, 32):
            stop = min(start + 32, replicates)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            draws[start:stop] = values[indices].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975], axis=0)
        result.update(ci_low=low.tolist(), ci_high=high.tolist())
    return result


def summarize_scores(
    arrays: dict[str, np.ndarray], *, seed: int, replicates: int
) -> dict[str, Any]:
    metrics = (
        "joint_mse",
        "weight_mse",
        "bias_mse",
        "readout_mse",
        "module_mse_by_task",
        "repeated_module_mse",
        "functional_mse_by_task",
        "functional_nmse_by_task",
    )
    ranks = arrays["latent_rank"]
    groups = {
        "all": np.ones(len(ranks), dtype=bool),
        **{f"rank_{rank}": ranks == rank for rank in np.unique(ranks)},
    }
    result: dict[str, Any] = {}
    for label in ("decoder", "zero"):
        report = {}
        for group, selected in groups.items():
            measurements = {name: arrays[f"{label}_{name}"][selected] for name in metrics}
            for name in ("functional_mse", "functional_nmse"):
                measurements[name] = measurements[f"{name}_by_task"].mean(axis=1)
            report[group] = {
                "n_episodes": int(selected.sum()),
                "metrics": {
                    name: episode_interval(value, seed=seed, replicates=replicates)
                    for name, value in measurements.items()
                },
                "variance_floored_tasks": int(
                    arrays[f"{label}_functional_variance_floored"][selected].sum()
                ),
            }
        result[label] = report
    return result


@torch.inference_mode()
def evaluate_probe(cfg: DictConfig, out_dir: Path | str) -> Path:
    validate_probe_config(cfg, "eval")
    started = time.perf_counter()
    device = configure_runtime(cfg)
    p, e = cfg.probe, cfg.probe.evaluation
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = Path(e.results_dir) if e.results_dir else out_dir / "probe-results"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"choose a fresh probe.evaluation.results_dir: {destination}")
    dataset = CapturedDataset(p.dataset.path, e.split)
    checkpoint_path = Path(e.checkpoint).expanduser()
    checkpoint = load_checkpoint(checkpoint_path, dataset)
    model = make_decoder(dataset.manifest["identity"]["input_features"], checkpoint["control"]).to(
        device
    )
    model.load_state_dict(checkpoint["model"])
    # Standalone scoring does not retain multi-gigabyte training optimizer buffers.
    checkpoint.pop("optimizer", None)
    checkpoint.pop("model", None)
    model.eval().requires_grad_(False)
    dtype = resolve_autocast_dtype(e.precision, device)
    weight = float(checkpoint["readout_weight"])
    functional_seed = stream_seed(
        e.functional_seed, f"functional/{dataset.manifest['dataset_id']}/{e.split}"
    )
    bootstrap_seed = stream_seed(e.bootstrap_seed, "episode-bootstrap")
    logger = RunLogger(
        cfg, out_dir, job_type="probe-eval", source=source_run(dataset.manifest), protocol=PROTOCOL
    )
    buffers: dict[str, list[np.ndarray]] = {}
    try:
        with AssignmentSolver(p.solver.num_threads, p.solver.cache_dir) as solver:
            logger.start()
            loader = probe_loader(
                dataset, e.batch_size, p.training.num_workers, device, seed=cfg.seed
            )
            for batch in loader:
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    predictions = model(batch["states"].to(device, non_blocking=True))
                predictions = predictions.float().cpu().numpy()
                cpu = {key: value.numpy() for key, value in batch.items() if key != "states"}
                measured = {key: cpu[key] for key in ("episode_index", "latent_rank")}
                for label, values in (
                    ("decoder", predictions),
                    ("zero", np.zeros_like(predictions)),
                ):
                    scores = score_predictions(
                        values,
                        cpu,
                        solver,
                        readout_weight=weight,
                        inputs_per_task=e.functional_inputs_per_task,
                        functional_seed=functional_seed,
                    )
                    measured.update({f"{label}_{key}": value for key, value in scores.items()})
                for key, value in measured.items():
                    buffers.setdefault(key, []).append(value)
            arrays = {key: np.concatenate(values) for key, values in buffers.items()}
            summary = summarize_scores(
                arrays, seed=bootstrap_seed, replicates=e.bootstrap_replicates
            )
            metadata = {
                "protocol": PROTOCOL,
                "dataset_id": dataset.manifest["dataset_id"],
                "dataset_path": str(dataset.root),
                "split": str(e.split),
                "split_signature": dataset.signature,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": file_digest(checkpoint_path),
                "step": int(checkpoint["step"]),
                "best_step": checkpoint["best_step"],
                "control": checkpoint["control"],
                "source_model_digest": checkpoint["source_model_digest"],
                "source_provenance": checkpoint["source_provenance"],
                "state_layout": checkpoint["state_layout"],
                "target_layout": TARGET_LAYOUT,
                "readout_weight": weight,
                "evaluation_precision": str(dtype or torch.float32),
                "device": str(device),
                "solver": solver.identity,
                "training_solver": checkpoint["solver"],
                "functional_diagnostic": "oracle-coefficient functional reconstruction",
                "functional_inputs_per_task": int(e.functional_inputs_per_task),
                "functional_seed": int(e.functional_seed),
                "functional_stream_seed": functional_seed,
                "functional_input_distribution": "uniform[-1,1], FP32",
                "variance_floor": BASE_MSE_FLOOR,
                "variance_definition": (
                    "fresh-input population variance per output, averaged over outputs"
                ),
                "bootstrap_seed": int(e.bootstrap_seed),
                "bootstrap_stream_seed": bootstrap_seed,
                "bootstrap_replicates": int(e.bootstrap_replicates),
                "bootstrap_unit": "whole episode",
                "confidence": 0.95,
                "config": resolved_config(cfg),
                "training_config": checkpoint["config"],
                "evaluation_seconds": time.perf_counter() - started,
            }
            write_results(destination, metadata, arrays, summary)
            # Figures depend only on the portable artifacts, never on the model or data loader.
            from iccl.analysis.plotting import plot_probe_results, probe_evaluation_figures

            plot_probe_results([destination], destination / "plots")
            rows = probe_summary_rows(summary)
            scalars = {
                f"probe/{e.split}/{row['prediction']}/{row['metric']}": row["mean"]
                for row in rows
                if row["population"] == "all" and row["task_position"] is None
            }
            logger.log_probe_evaluation(
                scalars,
                rows,
                probe_evaluation_figures(metadata, arrays, summary),
                checkpoint["step"],
                namespace=f"probe/{e.split}",
            )
            logger.upload_probe_artifact(destination, kind="results")
            return destination
    finally:
        dataset.close()
        logger.finish()
