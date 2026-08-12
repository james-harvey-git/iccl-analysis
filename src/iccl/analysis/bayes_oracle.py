"""Noise-free known-world, current-task-only Bayesian reference predictor.

The oracle knows each frozen sequence's module pool and the finite task-family
prior. It predicts from demonstrations in the current task, then discards its
posterior at the next explicit boundary. Ground-truth latents are read only for
post-hoc validation and diagnostics, never to form a prediction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from iccl.data.export import load_suite
from iccl.data.sequences import TOKEN_BOUNDARY, TOKEN_X, TOKEN_Y
from iccl.data.teacher import DISCRETE_WEIGHT_VALUES, ModulePool
from iccl.training.metrics import BASE_MSE_FLOOR

ORACLE_SCHEMA_VERSION = 1
NOT_IDENTIFIED = -1
_WORLD_MODULE = re.compile(r"world_modules_(\d+)$")


@dataclass(frozen=True)
class OracleConfig:
    """Numerical settings that are part of the deterministic cache identity."""

    atol: float = 1e-5
    rtol: float = 1e-5
    candidate_chunk_size: int = 128
    store_full_covariance: bool = False
    sensitivity_tolerances: tuple[tuple[float, float], ...] = (
        (1e-6, 1e-6),
        (1e-4, 1e-4),
    )


def oracle_config_from(cfg: DictConfig) -> OracleConfig:
    sensitivity = tuple(
        (float(pair[0]), float(pair[1])) for pair in cfg.get("sensitivity_tolerances", [])
    )
    return OracleConfig(
        atol=float(cfg.atol),
        rtol=float(cfg.rtol),
        candidate_chunk_size=int(cfg.candidate_chunk_size),
        store_full_covariance=bool(cfg.store_full_covariance),
        sensitivity_tolerances=sensitivity,
    )


def suite_paths(
    eval_dir: Path, suite_name: str, explicit_path: str | None = None
) -> tuple[Path, Path]:
    """The frozen array and metadata paths for one suite."""
    suite_path = (
        Path(explicit_path) if explicit_path is not None else eval_dir / f"{suite_name}.npz"
    )
    if suite_path.suffix != ".npz":
        suite_path = suite_path.with_suffix(".npz")
    meta_path = suite_path.with_suffix(".meta.json")
    if not suite_path.exists():
        raise FileNotFoundError(f"frozen suite not found: {suite_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"frozen suite metadata not found: {meta_path}")
    return suite_path, meta_path


def load_suite_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"suite metadata must be a JSON object: {path}")
    return metadata


def task_family_spec(metadata: dict[str, Any]) -> dict[str, Any]:
    """Resolve the finite prior from suite metadata without reading true latents."""
    try:
        data = metadata["config"]
        sequence = data["sequence"]
        phases = sequence["phases"]
    except (KeyError, TypeError) as error:
        raise ValueError("suite metadata has no resolved data/sequence configuration") from error

    if data.get("weighting") != "discrete":
        raise ValueError(
            f"known-world oracle v1 supports weighting=discrete only; got {data.get('weighting')!r}"
        )
    if not sequence.get("signal_boundaries", False):
        raise ValueError("known-world oracle v1 requires signalled task boundaries")

    fixed_hotness: set[int] = set()
    for phase in phases:
        lo, hi = (int(value) for value in phase["hotness"])
        if lo != hi:
            raise ValueError("known-world oracle v1 requires fixed task hotness")
        fixed_hotness.add(lo)
    if len(fixed_hotness) != 1:
        raise ValueError("known-world oracle v1 requires one hotness across all phases")
    hotness = fixed_hotness.pop()

    suite_name = str(metadata.get("suite", ""))
    if suite_name in {"composite", "composite_control"}:
        final_hotness = int(data["eval_sets"]["composite"]["hotness"])
        if final_hotness != hotness:
            raise ValueError(
                "known-world oracle v1 cannot mix curriculum and final-task hotness values"
            )

    num_modules = int(data["num_modules"])
    if not 0 < hotness <= num_modules:
        raise ValueError(f"invalid fixed hotness {hotness} for {num_modules} modules")
    return {
        "weighting": "discrete",
        "num_modules": num_modules,
        "hotness": hotness,
        "active_weight_values": [float(value) for value in DISCRETE_WEIGHT_VALUES],
    }


def enumerate_candidates(
    num_modules: int,
    hotness: int,
    active_weight_values: np.ndarray = DISCRETE_WEIGHT_VALUES,
) -> np.ndarray:
    """All weighted k-hot task latents in deterministic lexicographic order."""
    if not 0 < hotness <= num_modules:
        raise ValueError(f"hotness must be in [1, {num_modules}], got {hotness}")
    values = tuple(float(value) for value in active_weight_values)
    candidates: list[np.ndarray] = []
    for support in combinations(range(num_modules), hotness):
        for weights in product(values, repeat=hotness):
            latent = np.zeros(num_modules, dtype=np.float32)
            latent[list(support)] = weights
            candidates.append(latent)
    return np.stack(candidates)


def module_pool_from_suite(suite: dict[str, np.ndarray], sequence_index: int) -> ModulePool:
    """Reconstruct one frozen sequence's world from exported arrays."""
    layer_indices = sorted(
        int(match.group(1)) for key in suite if (match := _WORLD_MODULE.fullmatch(key)) is not None
    )
    if not layer_indices or layer_indices != list(range(len(layer_indices))):
        raise ValueError("suite has missing or non-contiguous world_modules_<layer> arrays")
    if "world_readout" not in suite:
        raise ValueError("suite has no world_readout array")
    modules = [suite[f"world_modules_{layer}"][sequence_index] for layer in layer_indices]
    try:
        biases = [suite[f"world_biases_{layer}"][sequence_index] for layer in layer_indices]
    except KeyError as error:
        raise ValueError(f"suite has no {error.args[0]} array") from error
    return ModulePool(
        modules=modules, biases=biases, readout=suite["world_readout"][sequence_index]
    )


def candidate_forward(
    pool: ModulePool,
    candidates: np.ndarray,
    x: np.ndarray,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Vectorized teacher forward for every candidate and input.

    Returns ``[candidates, inputs, output_dim]`` in candidate order. Chunking
    bounds temporary composed-weight and activation storage without changing
    the reduction over candidates used by the posterior predictive mean.
    """
    if chunk_size <= 0:
        raise ValueError(f"candidate_chunk_size must be positive, got {chunk_size}")
    hotness = np.count_nonzero(candidates, axis=1)
    if not np.all(hotness == hotness[0]):
        raise ValueError("candidate_forward requires a fixed-hotness candidate set")
    outputs = np.empty((len(candidates), len(x), pool.readout.shape[-1]), dtype=np.float32)
    x32 = x.astype(np.float32, copy=False)
    for start in range(0, len(candidates), chunk_size):
        stop = min(start + chunk_size, len(candidates))
        scaled = (candidates[start:stop] / math.sqrt(int(hotness[0]))).astype(np.float32)
        activations = np.broadcast_to(x32, (stop - start, *x32.shape))
        for module_w, module_b in zip(pool.modules, pool.biases, strict=True):
            weights = np.einsum("cm,mio->cio", scaled, module_w)
            biases = np.einsum("cm,mo->co", scaled, module_b)
            activations = np.maximum(
                np.einsum("cni,cio->cno", activations, weights) + biases[:, None, :],
                0.0,
            )
        outputs[start:stop] = np.einsum("cnh,ho->cno", activations, pool.readout)
    return outputs


def _validate_suite(suite: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    required = {
        "tokens",
        "token_type",
        "targets",
        "latents",
        "demo_counts",
        "boundaries",
        "task_spans",
        "base_mse",
    }
    missing = sorted(required - suite.keys())
    if missing:
        raise ValueError(f"suite is missing required arrays: {', '.join(missing)}")
    task_family_spec(metadata)
    num_sequences, num_tasks = suite["demo_counts"].shape
    if suite["task_spans"].shape != (num_sequences, num_tasks, 2):
        raise ValueError("task_spans shape does not match demo_counts")
    if suite["boundaries"].shape != (num_sequences, num_tasks):
        raise ValueError(
            "known-world oracle requires one exported boundary per task; "
            "the suite appears unsignalled or malformed"
        )
    for n in range(num_sequences):
        boundaries = suite["boundaries"][n].astype(np.int64)
        starts = suite["task_spans"][n, :, 0].astype(np.int64)
        if not np.array_equal(boundaries + 1, starts):
            raise ValueError(f"sequence {n} task spans do not immediately follow boundaries")
        if not np.all(suite["token_type"][n, boundaries] == TOKEN_BOUNDARY):
            raise ValueError(f"sequence {n} has a non-boundary token at an exported boundary")


def _nan_array(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float32)


def _candidate_index(candidates: np.ndarray, latent: np.ndarray) -> int:
    matches = np.flatnonzero(np.all(candidates == latent.astype(np.float32), axis=1))
    if len(matches) != 1:
        raise ValueError(f"true latent matches {len(matches)} candidates; expected exactly one")
    return int(matches[0])


def compute_oracle(
    suite: dict[str, np.ndarray],
    metadata: dict[str, Any],
    config: OracleConfig,
) -> dict[str, np.ndarray]:
    """Run the history-free survivor filter over every task in a frozen suite."""
    _validate_suite(suite, metadata)
    spec = task_family_spec(metadata)
    candidates = enumerate_candidates(spec["num_modules"], spec["hotness"])
    counts = suite["demo_counts"].astype(np.int64)
    num_sequences, num_tasks = counts.shape
    max_demos = int(counts.max())
    output_dim = int(suite["targets"].shape[-1])
    curve_shape = (num_sequences, num_tasks, max_demos)

    arrays: dict[str, np.ndarray] = {
        "candidates": candidates,
        "predictions": _nan_array((*curve_shape, output_dim)),
        "raw_mse": _nan_array(curve_shape),
        "nmse": _nan_array(curve_shape),
        "survivor_count_before": np.full(curve_shape, -1, dtype=np.int32),
        "survivor_count_after": np.full(curve_shape, -1, dtype=np.int32),
        "entropy_before": _nan_array(curve_shape),
        "entropy_after": _nan_array(curve_shape),
        "effective_hypothesis_count_before": _nan_array(curve_shape),
        "effective_hypothesis_count_after": _nan_array(curve_shape),
        "predictive_variance_diag": _nan_array((*curve_shape, output_dim)),
        "predictive_covariance_trace": _nan_array(curve_shape),
        "true_posterior_mass_after": _nan_array(curve_shape),
        "unique_identified_after": np.zeros(curve_shape, dtype=bool),
        "identification_step": np.full((num_sequences, num_tasks), NOT_IDENTIFIED, dtype=np.int32),
        "num_modules_seen_before": np.zeros((num_sequences, num_tasks), dtype=np.int16),
        "num_current_modules_seen": np.zeros((num_sequences, num_tasks), dtype=np.int16),
        "num_current_modules_unseen": np.zeros((num_sequences, num_tasks), dtype=np.int16),
        "all_current_modules_seen": np.zeros((num_sequences, num_tasks), dtype=bool),
        "demo_counts": counts.copy(),
        "task_positions": np.arange(1, num_tasks + 1, dtype=np.int16),
    }
    sensitivity_shape = (len(config.sensitivity_tolerances), *curve_shape)
    arrays["sensitivity_survivor_count_after"] = np.full(sensitivity_shape, -1, dtype=np.int32)
    arrays["sensitivity_true_survives_after"] = np.zeros(sensitivity_shape, dtype=bool)
    if config.store_full_covariance:
        arrays["predictive_covariance"] = _nan_array((*curve_shape, output_dim, output_dim))

    for n in range(num_sequences):
        pool = module_pool_from_suite(suite, n)
        seen_modules = np.zeros(spec["num_modules"], dtype=bool)
        for task in range(num_tasks):
            count = int(counts[n, task])
            start = int(suite["task_spans"][n, task, 0])
            positions = start + 2 * np.arange(count)
            if not np.all(suite["token_type"][n, positions] == TOKEN_X):
                raise ValueError(f"sequence {n}, task {task + 1} has malformed x-token positions")
            if not np.all(suite["token_type"][n, positions + 1] == TOKEN_Y):
                raise ValueError(f"sequence {n}, task {task + 1} has malformed y-token positions")

            current_latent = suite["latents"][n, task]
            true_index = _candidate_index(candidates, current_latent)
            current_support = current_latent != 0
            current_seen = current_support & seen_modules
            arrays["num_modules_seen_before"][n, task] = int(seen_modules.sum())
            arrays["num_current_modules_seen"][n, task] = int(current_seen.sum())
            arrays["num_current_modules_unseen"][n, task] = int(
                (current_support & ~seen_modules).sum()
            )
            arrays["all_current_modules_seen"][n, task] = bool(
                np.all(seen_modules[current_support])
            )

            input_dim = int(pool.modules[0].shape[1])
            x = suite["tokens"][n, positions, :input_dim]
            y = suite["targets"][n, positions]
            candidate_outputs = candidate_forward(
                pool, candidates, x, chunk_size=config.candidate_chunk_size
            )
            survivors = np.ones(len(candidates), dtype=bool)
            sensitivity_survivors = [
                np.ones(len(candidates), dtype=bool) for _ in config.sensitivity_tolerances
            ]
            denominator = max(float(suite["base_mse"][n, task].mean()), BASE_MSE_FLOOR)

            for demo in range(count):
                active_outputs = candidate_outputs[survivors, demo].astype(np.float64)
                prediction = active_outputs.mean(axis=0)
                centered = active_outputs - prediction
                covariance = centered.T @ centered / len(active_outputs)
                prediction32 = prediction.astype(np.float32)

                before = int(survivors.sum())
                arrays["predictions"][n, task, demo] = prediction32
                arrays["survivor_count_before"][n, task, demo] = before
                arrays["entropy_before"][n, task, demo] = math.log(before)
                arrays["effective_hypothesis_count_before"][n, task, demo] = before
                arrays["predictive_variance_diag"][n, task, demo] = np.diag(covariance)
                arrays["predictive_covariance_trace"][n, task, demo] = np.trace(covariance)
                if config.store_full_covariance:
                    arrays["predictive_covariance"][n, task, demo] = covariance

                mse = float(np.mean((prediction32 - y[demo]) ** 2))
                arrays["raw_mse"][n, task, demo] = mse
                arrays["nmse"][n, task, demo] = mse / denominator

                tolerance = config.atol + config.rtol * np.abs(y[demo])
                survivors &= np.all(
                    np.abs(candidate_outputs[:, demo] - y[demo]) <= tolerance,
                    axis=1,
                )
                after = int(survivors.sum())
                if after == 0:
                    max_residual = np.max(np.abs(candidate_outputs[:, demo] - y[demo]), axis=1)
                    raise RuntimeError(
                        "oracle has zero survivors: "
                        f"suite={metadata.get('suite', 'unknown')}, sequence={n}, "
                        f"task={task + 1}, demo={demo + 1}, "
                        f"minimum_max_abs_residual={float(max_residual.min()):.8g}, "
                        f"atol={config.atol}, rtol={config.rtol}"
                    )
                if not survivors[true_index]:
                    raise RuntimeError(
                        "true task was eliminated by the noise-free oracle: "
                        f"suite={metadata.get('suite', 'unknown')}, sequence={n}, "
                        f"task={task + 1}, demo={demo + 1}"
                    )
                arrays["survivor_count_after"][n, task, demo] = after
                arrays["entropy_after"][n, task, demo] = math.log(after)
                arrays["effective_hypothesis_count_after"][n, task, demo] = after
                arrays["true_posterior_mass_after"][n, task, demo] = 1.0 / after
                arrays["unique_identified_after"][n, task, demo] = after == 1
                if after == 1 and arrays["identification_step"][n, task] == NOT_IDENTIFIED:
                    arrays["identification_step"][n, task] = demo + 1

                for sensitivity_index, ((atol, rtol), sensitivity_mask) in enumerate(
                    zip(config.sensitivity_tolerances, sensitivity_survivors, strict=True)
                ):
                    sensitivity_tolerance = atol + rtol * np.abs(y[demo])
                    sensitivity_mask &= np.all(
                        np.abs(candidate_outputs[:, demo] - y[demo]) <= sensitivity_tolerance,
                        axis=1,
                    )
                    arrays["sensitivity_survivor_count_after"][sensitivity_index, n, task, demo] = (
                        int(sensitivity_mask.sum())
                    )
                    arrays["sensitivity_true_survives_after"][sensitivity_index, n, task, demo] = (
                        bool(sensitivity_mask[true_index])
                    )

            seen_modules |= current_support
    return arrays


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def cache_identity(
    suite_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    config: OracleConfig,
) -> dict[str, Any]:
    """Everything that makes one cache scientifically and numerically specific."""
    identity = {
        "schema_version": ORACLE_SCHEMA_VERSION,
        "suite_sha256": file_sha256(suite_path),
        "metadata_sha256": file_sha256(metadata_path),
        "data_config": metadata.get("config"),
        "task_family": task_family_spec(metadata),
        "oracle_config": asdict(config),
        "implementation_git_commit": _git_commit(),
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    # JSON is the persisted canonical form; normalize tuples here so an identity
    # compares equal before and after a sidecar round trip.
    return json.loads(json.dumps(identity, sort_keys=True))


def identity_hash(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def default_cache_path(cache_dir: Path, suite_path: Path, identity: dict[str, Any]) -> Path:
    return cache_dir / f"{suite_path.stem}-{identity_hash(identity)[:16]}.npz"


def cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def write_oracle_cache(
    cache_path: Path,
    arrays: dict[str, np.ndarray],
    identity: dict[str, Any],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, allow_pickle=False, **arrays)
    metadata = {
        "identity": identity,
        "identity_hash": identity_hash(identity),
        "created_at": datetime.now(UTC).isoformat(),
        "identification_step_semantics": (
            "one-based number of current-task labels observed; -1 means not identified"
        ),
        "array_schema": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }
    cache_metadata_path(cache_path).write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_oracle_cache(
    cache_path: Path,
    *,
    expected_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata_path = cache_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"oracle cache or sidecar is missing: {cache_path}")
    metadata = json.loads(metadata_path.read_text())
    if expected_identity is not None and metadata.get("identity") != expected_identity:
        raise ValueError(
            f"stale oracle cache identity: {cache_path}; regenerate it for this suite/settings"
        )
    with np.load(cache_path, allow_pickle=False) as data:
        arrays = dict(data)
    expected_schema = metadata.get("array_schema", {})
    actual_schema = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }
    if actual_schema != expected_schema:
        raise ValueError(f"oracle cache array schema does not match its sidecar: {cache_path}")
    return arrays, metadata


def generate_or_reuse_cache(
    suite_path: Path,
    metadata_path: Path,
    cache_dir: Path,
    config: OracleConfig,
) -> tuple[Path, bool]:
    """Return the exact cache path and whether it was newly computed."""
    metadata = load_suite_metadata(metadata_path)
    identity = cache_identity(suite_path, metadata_path, metadata, config)
    cache_path = default_cache_path(cache_dir, suite_path, identity)
    if cache_path.exists() or cache_metadata_path(cache_path).exists():
        load_oracle_cache(cache_path, expected_identity=identity)
        return cache_path, False
    suite = load_suite(suite_path.with_suffix(""))
    arrays = compute_oracle(suite, metadata, config)
    write_oracle_cache(cache_path, arrays, identity)
    return cache_path, True


def resolve_cache(
    suite_path: Path,
    metadata_path: Path,
    cache_dir: Path,
    config: OracleConfig,
    explicit_path: str | None = None,
) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    """Load an explicit cache or the content-addressed cache for this suite."""
    metadata = load_suite_metadata(metadata_path)
    identity = cache_identity(suite_path, metadata_path, metadata, config)
    cache_path = (
        Path(explicit_path)
        if explicit_path is not None
        else default_cache_path(cache_dir, suite_path, identity)
    )
    arrays, cache_metadata = load_oracle_cache(cache_path, expected_identity=identity)
    return cache_path, arrays, cache_metadata
