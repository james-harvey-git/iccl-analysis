"""Configuration, validation, and content-addressed structured-observer caches."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from iccl.analysis.bayes_oracle import file_sha256, load_suite_metadata
from iccl.analysis.structured_observer.kernel import validate_observer_device
from iccl.analysis.structured_observer.runner import (
    compute_structured_observers,
    make_feature_bank_from_spec,
)
from iccl.analysis.structured_observer.schedule import ScheduleConfig, StructuredSuiteSpec
from iccl.analysis.structured_observer.smc import SMCConfig
from iccl.data.export import load_suite
from iccl.data.teacher import DISCRETE_WEIGHT_VALUES

STRUCTURED_OBSERVER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class KernelSettings:
    num_features: int
    seed: int
    dtype: str
    relative_jitter: float
    max_relative_jitter: float


@dataclass(frozen=True)
class StructuredObserverSettings:
    modes: tuple[str, ...]
    device: str
    kernel: KernelSettings
    smc: SMCConfig
    smc_seeds: tuple[int, ...]
    sequence_limit: int | None


def settings_from_config(
    cfg: DictConfig,
    *,
    require_device_available: bool = True,
) -> StructuredObserverSettings:
    """Resolve Hydra values into the immutable numerical cache settings."""
    kernel = cfg.kernel
    smc = cfg.smc
    settings = StructuredObserverSettings(
        modes=tuple(str(mode) for mode in cfg.modes),
        device=str(cfg.device),
        kernel=KernelSettings(
            num_features=int(kernel.num_features),
            seed=int(kernel.seed),
            dtype=str(kernel.dtype),
            relative_jitter=float(kernel.relative_jitter),
            max_relative_jitter=float(kernel.max_relative_jitter),
        ),
        smc=SMCConfig(
            num_particles=int(smc.num_particles),
            ess_fraction=float(smc.ess_fraction),
            task_end_rejuvenation_sweeps=int(smc.task_end_rejuvenation_sweeps),
            max_completion_attempts=int(smc.max_completion_attempts),
            initial_gp_capacity=int(smc.initial_gp_capacity),
        ),
        smc_seeds=tuple(int(seed) for seed in smc.seeds),
        sequence_limit=(
            None if cfg.get("sequence_limit") is None else int(cfg.sequence_limit)
        ),
    )
    validate_observer_device(
        settings.device,
        require_available=require_device_available,
    )
    if settings.kernel.dtype != "float64":
        raise ValueError("structured observer v1 requires kernel.dtype=float64")
    if "full_history" in settings.modes and not settings.smc_seeds:
        raise ValueError("full_history mode requires at least one SMC seed")
    return settings


def structured_suite_spec(metadata: dict[str, Any]) -> StructuredSuiteSpec:
    """Validate and resolve the supported in-distribution HyperTeacher family."""
    if metadata.get("suite") != "in_dist":
        raise ValueError("structured observer v1 supports the frozen in_dist suite only")
    try:
        data = metadata["config"]
        sequence = data["sequence"]
        phases = sequence["phases"]
    except (KeyError, TypeError) as error:
        raise ValueError("suite metadata has no resolved data/sequence configuration") from error
    if data.get("weighting") != "discrete":
        raise ValueError("structured observer v1 requires weighting=discrete")
    if not sequence.get("signal_boundaries", False):
        raise ValueError("structured observer v1 requires signalled boundaries")
    if sequence.get("task_graph") != "random":
        raise ValueError("structured observer v1 requires task_graph=random")
    if not sequence.get("require_identifiable", False):
        raise ValueError("structured observer v1 requires identifiable schedules")
    hidden_dims = tuple(int(value) for value in data["hidden_dims"])
    if len(hidden_dims) != 1:
        raise ValueError("structured observer v1 requires exactly one hidden layer")
    hotness_values: set[int] = set()
    for phase in phases:
        low, high = (int(value) for value in phase["hotness"])
        if low != high:
            raise ValueError("structured observer v1 requires fixed task hotness")
        hotness_values.add(low)
    if hotness_values != {2}:
        raise ValueError("structured observer v1 requires exactly two-hot tasks")
    return {
        "input_dim": int(data["input_dim"]),
        "output_dim": int(data["output_dim"]),
        "hidden_dim": hidden_dims[0],
        "use_bias": bool(data["use_bias"]),
        "num_modules": int(data["num_modules"]),
        "num_tasks": sum(int(phase["num_tasks"]) for phase in phases),
        "hotness": 2,
        "scale": float(data["scale"]),
        "weight_values": [float(value) for value in DISCRETE_WEIGHT_VALUES],
        "require_coverage": True,
        "require_connected": True,
    }


def schedule_config_from_spec(spec: StructuredSuiteSpec) -> ScheduleConfig:
    return ScheduleConfig(
        num_modules=int(spec["num_modules"]),
        num_tasks=int(spec["num_tasks"]),
        hotness=int(spec["hotness"]),
        weight_values=tuple(float(value) for value in spec["weight_values"]),
        require_coverage=bool(spec["require_coverage"]),
        require_connected=bool(spec["require_connected"]),
    )


def validate_suite_arrays(suite: dict[str, np.ndarray], spec: StructuredSuiteSpec) -> None:
    """Check only driver and post-hoc arrays; privileged fields never enter observers."""
    required = {
        "tokens",
        "token_type",
        "targets",
        "demo_counts",
        "task_spans",
        "base_mse",
        "latents",
    }
    missing = sorted(required - suite.keys())
    if missing:
        raise ValueError(f"suite is missing required arrays: {', '.join(missing)}")
    num_sequences = suite["tokens"].shape[0]
    expected_tasks = int(spec["num_tasks"])
    if suite["demo_counts"].shape != (num_sequences, expected_tasks):
        raise ValueError("demo_counts does not match the metadata task count")
    if suite["latents"].shape != (
        num_sequences,
        expected_tasks,
        int(spec["num_modules"]),
    ):
        raise ValueError("latents does not match the metadata schedule dimensions")


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
    settings: StructuredObserverSettings,
) -> dict[str, Any]:
    """Describe every input to the deterministic approximate reference."""
    package_dir = Path(__file__).parent
    implementation_hashes = {
        path.name: file_sha256(path) for path in sorted(package_dir.glob("*.py"))
    }
    identity = {
        "schema_version": STRUCTURED_OBSERVER_SCHEMA_VERSION,
        "suite_sha256": file_sha256(suite_path),
        "metadata_sha256": file_sha256(metadata_path),
        "data_config": metadata.get("config"),
        "structured_suite_spec": structured_suite_spec(metadata),
        "settings": asdict(settings),
        "implementation_git_commit": _git_commit(),
        "implementation_sha256": implementation_hashes,
    }
    return json.loads(json.dumps(identity, sort_keys=True))


def identity_hash(identity: dict[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def default_cache_path(cache_dir: Path, suite_path: Path, identity: dict[str, Any]) -> Path:
    return cache_dir / f"{suite_path.stem}-{identity_hash(identity)[:16]}.npz"


def cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def write_cache(
    cache_path: Path,
    arrays: dict[str, np.ndarray],
    identity: dict[str, Any],
    *,
    feature_bank_hash: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, allow_pickle=False, **arrays)
    metadata = {
        "identity": identity,
        "identity_hash": identity_hash(identity),
        "feature_bank_sha256": feature_bank_hash,
        "created_at": datetime.now(UTC).isoformat(),
        "array_schema": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "interpretation": (
            "generator-aware random-feature GP reference; approximate and not a "
            "certified lower bound for the finite-width teacher or GDN"
        ),
    }
    cache_metadata_path(cache_path).write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_cache(
    cache_path: Path,
    *,
    expected_identity: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata_path = cache_metadata_path(cache_path)
    if not cache_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"structured-observer cache or sidecar is missing: {cache_path}")
    metadata = json.loads(metadata_path.read_text())
    if expected_identity is not None and metadata.get("identity") != expected_identity:
        raise ValueError(f"stale structured-observer cache identity: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as data:
        arrays = dict(data)
    actual_schema = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }
    if actual_schema != metadata.get("array_schema"):
        raise ValueError(f"cache array schema does not match its sidecar: {cache_path}")
    return arrays, metadata


def generate_or_reuse_cache(
    suite_path: Path,
    metadata_path: Path,
    cache_dir: Path,
    settings: StructuredObserverSettings,
    *,
    progress: bool = True,
) -> tuple[Path, bool]:
    metadata = load_suite_metadata(metadata_path)
    identity = cache_identity(suite_path, metadata_path, metadata, settings)
    cache_path = default_cache_path(cache_dir, suite_path, identity)
    if cache_path.exists() or cache_metadata_path(cache_path).exists():
        load_cache(cache_path, expected_identity=identity)
        return cache_path, False
    suite = load_suite(suite_path.with_suffix(""))
    spec = structured_suite_spec(metadata)
    validate_suite_arrays(suite, spec)
    feature_bank = make_feature_bank_from_spec(
        spec,
        num_features=settings.kernel.num_features,
        seed=settings.kernel.seed,
        device=settings.device,
        dtype=settings.kernel.dtype,
    )
    arrays = compute_structured_observers(
        suite,
        schedule_config=schedule_config_from_spec(spec),
        feature_bank=feature_bank,
        modes=settings.modes,
        smc_config=settings.smc,
        smc_seeds=settings.smc_seeds,
        relative_jitter=settings.kernel.relative_jitter,
        max_relative_jitter=settings.kernel.max_relative_jitter,
        sequence_limit=settings.sequence_limit,
        progress=print if progress else None,
    )
    write_cache(
        cache_path,
        arrays,
        identity,
        feature_bank_hash=feature_bank.content_hash(),
    )
    return cache_path, True


def resolve_cache(
    suite_path: Path,
    metadata_path: Path,
    cache_dir: Path,
    settings: StructuredObserverSettings,
    explicit_path: str | None = None,
) -> tuple[Path, dict[str, np.ndarray], dict[str, Any]]:
    metadata = load_suite_metadata(metadata_path)
    identity = cache_identity(suite_path, metadata_path, metadata, settings)
    cache_path = (
        Path(explicit_path)
        if explicit_path is not None
        else default_cache_path(cache_dir, suite_path, identity)
    )
    arrays, cache_metadata = load_cache(cache_path, expected_identity=identity)
    return cache_path, arrays, cache_metadata
