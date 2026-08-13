"""Causal sequence driver and frozen-suite evaluation for structured observers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from iccl.analysis.structured_observer.events import (
    BoundaryEvent,
    InputEvent,
    OutputEvent,
    iter_observation_events,
)
from iccl.analysis.structured_observer.kernel import FeatureBank, sample_feature_bank
from iccl.analysis.structured_observer.observer import (
    CurrentTaskObserver,
    ObserverPrediction,
    ObserverUpdate,
    TaskEndDiagnostics,
)
from iccl.analysis.structured_observer.schedule import ScheduleConfig, StructuredSuiteSpec
from iccl.analysis.structured_observer.smc import FullHistoryObserver, SMCConfig
from iccl.training.metrics import BASE_MSE_FLOOR


class SequenceObserver(Protocol):
    """The information-limited interface shared by both reference algorithms."""

    def start_task(self) -> None: ...

    def predict(self, x: np.ndarray) -> ObserverPrediction: ...

    def observe(self, y: np.ndarray) -> ObserverUpdate: ...

    def end_task(self) -> TaskEndDiagnostics: ...


@dataclass
class _DemoRecord:
    prediction: ObserverPrediction
    update: ObserverUpdate | None = None


def _record_arrays(
    tasks: list[list[_DemoRecord]],
    task_ends: list[TaskEndDiagnostics],
    output_dim: int,
) -> dict[str, np.ndarray]:
    num_tasks = len(tasks)
    max_demos = max(len(task) for task in tasks)
    shape = (num_tasks, max_demos)

    def floats() -> np.ndarray:
        return np.full(shape, np.nan, dtype=np.float32)

    arrays: dict[str, np.ndarray] = {
        "predictions": np.full((*shape, output_dim), np.nan, dtype=np.float32),
        "predictive_covariance_trace": floats(),
        "ess_before": floats(),
        "ess_after": floats(),
        "max_weight_before": floats(),
        "max_weight_after": floats(),
        "unique_prefixes_before": np.full(shape, -1, dtype=np.int32),
        "unique_prefixes_after": np.full(shape, -1, dtype=np.int32),
        "log_evidence_before": floats(),
        "log_evidence_after": floats(),
        "relative_jitter": floats(),
        "resampled": np.zeros(shape, dtype=bool),
        "rejuvenation_acceptance": floats(),
        "completion_attempts": np.full(shape, -1, dtype=np.int64),
        "demo_counts": np.asarray([len(task) for task in tasks], dtype=np.int64),
        "task_end_rejuvenation_acceptance": np.asarray(
            [diagnostic.rejuvenation_acceptance for diagnostic in task_ends],
            dtype=np.float32,
        ),
        "task_end_completion_attempts": np.asarray(
            [diagnostic.completion_attempts for diagnostic in task_ends],
            dtype=np.int64,
        ),
        "task_end_resampling_events": np.asarray(
            [diagnostic.resampling_events for diagnostic in task_ends],
            dtype=np.int64,
        ),
        "task_end_unique_prefixes": np.asarray(
            [diagnostic.unique_prefixes for diagnostic in task_ends],
            dtype=np.int32,
        ),
    }
    for task_index, task in enumerate(tasks):
        for demo_index, record in enumerate(task):
            if record.update is None:
                raise RuntimeError("demonstration record has no posterior update")
            prediction = record.prediction
            update = record.update
            arrays["predictions"][task_index, demo_index] = prediction.mean
            arrays["predictive_covariance_trace"][task_index, demo_index] = (
                prediction.covariance_trace
            )
            arrays["ess_before"][task_index, demo_index] = prediction.effective_sample_size
            arrays["ess_after"][task_index, demo_index] = update.effective_sample_size
            arrays["max_weight_before"][task_index, demo_index] = prediction.max_weight
            arrays["max_weight_after"][task_index, demo_index] = update.max_weight
            arrays["unique_prefixes_before"][task_index, demo_index] = (
                prediction.unique_prefixes
            )
            arrays["unique_prefixes_after"][task_index, demo_index] = update.unique_prefixes
            arrays["log_evidence_before"][task_index, demo_index] = prediction.log_evidence
            arrays["log_evidence_after"][task_index, demo_index] = update.log_evidence
            arrays["relative_jitter"][task_index, demo_index] = update.relative_jitter
            arrays["resampled"][task_index, demo_index] = update.resampled
            arrays["rejuvenation_acceptance"][task_index, demo_index] = (
                update.rejuvenation_acceptance
            )
            arrays["completion_attempts"][task_index, demo_index] = (
                update.completion_attempts
            )
    return arrays


def run_observation_sequence(
    observer: SequenceObserver,
    tokens: np.ndarray,
    token_types: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
) -> dict[str, np.ndarray]:
    """Drive an observer one event at a time without exposing future data."""
    tasks: list[list[_DemoRecord]] = []
    task_ends: list[TaskEndDiagnostics] = []
    active_task = -1
    for event in iter_observation_events(
        tokens,
        token_types,
        input_dim=input_dim,
        output_dim=output_dim,
    ):
        if isinstance(event, BoundaryEvent):
            if active_task >= 0:
                task_ends.append(observer.end_task())
            observer.start_task()
            tasks.append([])
            active_task += 1
        elif isinstance(event, InputEvent):
            if active_task < 0 or event.task_position != active_task + 1:
                raise RuntimeError("input event does not match the active task")
            if event.demo_index != len(tasks[active_task]):
                raise RuntimeError("input event demo index is not sequential")
            tasks[active_task].append(_DemoRecord(prediction=observer.predict(event.value)))
        elif isinstance(event, OutputEvent):
            if active_task < 0 or not tasks[active_task]:
                raise RuntimeError("output event has no active prediction")
            record = tasks[active_task][-1]
            if record.update is not None or event.demo_index != len(tasks[active_task]) - 1:
                raise RuntimeError("output event does not match the pending prediction")
            record.update = observer.observe(event.value)
    if active_task < 0:
        raise ValueError("observation stream contains no task boundaries")
    task_ends.append(observer.end_task())
    return _record_arrays(tasks, task_ends, output_dim)


def _allocate_stacked(
    sequence_results: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Stack equal-shape sequence records after causal processing is complete."""
    keys = sequence_results[0].keys()
    return {key: np.stack([result[key] for result in sequence_results]) for key in keys}


def _metric_arrays(
    predictions: np.ndarray,
    targets: np.ndarray,
    base_mse: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.mean((predictions - targets) ** 2, axis=-1)
    denominator = np.maximum(base_mse.mean(axis=-1), BASE_MSE_FLOOR)
    nmse = raw / denominator[:, :, None]
    valid = np.arange(raw.shape[-1])[None, None, :] < counts[:, :, None]
    return np.where(valid, raw, np.nan), np.where(valid, nmse, np.nan)


def _targets_from_suite(
    suite: dict[str, np.ndarray],
    num_sequences: int,
) -> np.ndarray:
    counts = suite["demo_counts"][:num_sequences].astype(np.int64)
    max_demos = int(counts.max())
    output_dim = int(suite["targets"].shape[-1])
    targets = np.full((*counts.shape, max_demos, output_dim), np.nan, dtype=np.float32)
    for sequence_index in range(num_sequences):
        for task_index, count in enumerate(counts[sequence_index]):
            start = int(suite["task_spans"][sequence_index, task_index, 0])
            positions = start + 2 * np.arange(count)
            targets[sequence_index, task_index, :count] = suite["targets"][
                sequence_index, positions
            ]
    return targets


def _coverage_arrays(latents: np.ndarray) -> dict[str, np.ndarray]:
    num_sequences, num_tasks, _ = latents.shape
    seen = np.zeros(latents.shape[::2], dtype=bool)
    all_seen = np.zeros((num_sequences, num_tasks), dtype=bool)
    seen_count = np.zeros((num_sequences, num_tasks), dtype=np.int16)
    current_seen_count = np.zeros((num_sequences, num_tasks), dtype=np.int16)
    for sequence in range(num_sequences):
        for task in range(num_tasks):
            support = latents[sequence, task] != 0.0
            seen_count[sequence, task] = int(seen[sequence].sum())
            current_seen = support & seen[sequence]
            current_seen_count[sequence, task] = int(current_seen.sum())
            all_seen[sequence, task] = bool(np.all(seen[sequence, support]))
            seen[sequence] |= support
    return {
        "num_modules_seen_before": seen_count,
        "num_current_modules_seen": current_seen_count,
        "all_current_modules_seen": all_seen,
    }


def compute_structured_observers(
    suite: dict[str, np.ndarray],
    *,
    schedule_config: ScheduleConfig,
    feature_bank: FeatureBank,
    modes: tuple[str, ...],
    smc_config: SMCConfig,
    smc_seeds: tuple[int, ...],
    relative_jitter: float,
    max_relative_jitter: float,
    sequence_limit: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate requested observers and compute metrics only after prediction."""
    allowed_modes = {"full_history", "current_task"}
    if not modes or not set(modes) <= allowed_modes:
        raise ValueError(f"modes must be a non-empty subset of {sorted(allowed_modes)}")
    total_sequences = int(suite["tokens"].shape[0])
    num_sequences = (
        total_sequences
        if sequence_limit is None
        else min(sequence_limit, total_sequences)
    )
    if num_sequences <= 0:
        raise ValueError("sequence_limit selects no sequences")
    output_dim = int(suite["targets"].shape[-1])
    input_dim = feature_bank.input_dim
    arrays: dict[str, np.ndarray] = {
        "sequence_indices": np.arange(num_sequences, dtype=np.int32),
        "task_positions": np.arange(1, schedule_config.num_tasks + 1, dtype=np.int16),
        "demo_counts": suite["demo_counts"][:num_sequences].astype(np.int64, copy=True),
        "smc_seeds": np.asarray(smc_seeds, dtype=np.int64),
    }

    if "current_task" in modes:
        results = []
        for sequence_index in range(num_sequences):
            observer = CurrentTaskObserver(
                feature_bank=feature_bank,
                schedule_config=schedule_config,
                output_dim=output_dim,
                relative_jitter=relative_jitter,
                max_relative_jitter=max_relative_jitter,
            )
            results.append(
                run_observation_sequence(
                    observer,
                    suite["tokens"][sequence_index],
                    suite["token_type"][sequence_index],
                    input_dim=input_dim,
                    output_dim=output_dim,
                )
            )
            if progress is not None:
                progress(f"current_task sequence {sequence_index + 1}/{num_sequences}")
        for key, value in _allocate_stacked(results).items():
            arrays[f"current_task_{key}"] = value

    if "full_history" in modes:
        results_by_seed: list[dict[str, np.ndarray]] = []
        for seed in smc_seeds:
            results = []
            for sequence_index in range(num_sequences):
                keyed_seed = int(
                    np.random.SeedSequence([seed, sequence_index]).generate_state(1)[0]
                )
                observer = FullHistoryObserver(
                    feature_bank=feature_bank,
                    schedule_config=schedule_config,
                    output_dim=output_dim,
                    relative_jitter=relative_jitter,
                    max_relative_jitter=max_relative_jitter,
                    smc_config=smc_config,
                    seed=keyed_seed,
                )
                results.append(
                    run_observation_sequence(
                        observer,
                        suite["tokens"][sequence_index],
                        suite["token_type"][sequence_index],
                        input_dim=input_dim,
                        output_dim=output_dim,
                    )
                )
                if progress is not None:
                    progress(
                        f"full_history seed {seed} sequence "
                        f"{sequence_index + 1}/{num_sequences}"
                    )
            results_by_seed.append(_allocate_stacked(results))
        for key in results_by_seed[0]:
            stacked = np.stack([result[key] for result in results_by_seed])
            arrays[f"full_{key}_by_seed"] = stacked
        full_predictions = arrays["full_predictions_by_seed"]
        arrays["full_predictions_mean"] = np.mean(full_predictions, axis=0)
        arrays["full_algorithmic_prediction_std"] = np.std(
            full_predictions,
            axis=0,
            ddof=1 if len(smc_seeds) > 1 else 0,
        )

    targets = _targets_from_suite(suite, num_sequences)
    counts = arrays["demo_counts"]
    base_mse = suite["base_mse"][:num_sequences]
    arrays["targets"] = targets
    if "current_task" in modes:
        raw, nmse = _metric_arrays(
            arrays["current_task_predictions"], targets, base_mse, counts
        )
        arrays["current_task_raw_mse"] = raw.astype(np.float32)
        arrays["current_task_nmse"] = nmse.astype(np.float32)
    if "full_history" in modes:
        raw_by_seed = []
        nmse_by_seed = []
        for predictions in arrays["full_predictions_by_seed"]:
            raw, nmse = _metric_arrays(predictions, targets, base_mse, counts)
            raw_by_seed.append(raw)
            nmse_by_seed.append(nmse)
        arrays["full_raw_mse_by_seed"] = np.stack(raw_by_seed).astype(np.float32)
        arrays["full_nmse_by_seed"] = np.stack(nmse_by_seed).astype(np.float32)
        raw, nmse = _metric_arrays(arrays["full_predictions_mean"], targets, base_mse, counts)
        arrays["full_raw_mse"] = raw.astype(np.float32)
        arrays["full_nmse"] = nmse.astype(np.float32)
    arrays.update(_coverage_arrays(suite["latents"][:num_sequences]))
    return arrays


def make_feature_bank_from_spec(
    spec: StructuredSuiteSpec,
    *,
    num_features: int,
    seed: int,
    device: str,
    dtype: str,
) -> FeatureBank:
    """Construct the fixed kernel draw recorded in a cache identity."""
    return sample_feature_bank(
        input_dim=int(spec["input_dim"]),
        num_modules=int(spec["num_modules"]),
        scale=float(spec["scale"]),
        num_features=num_features,
        seed=seed,
        device=device,
        dtype=dtype,
        use_bias=bool(spec["use_bias"]),
    )
