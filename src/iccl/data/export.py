"""Generate deterministic frozen validation and capability-suite archives.

Capability evaluation expands a deduplicated set of fixed-demo ``(M,T,D)``
cells into ordinary, composition and retention conditions. Metadata beside each
archive records its structural families, training-distribution status, pairing
and generation provenance so training monitoring and standalone evaluation use
the same examples.
"""

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from iccl.data.controls import (
    build_paired_composition_controls,
)
from iccl.data.curriculum import (
    CURRICULUM_SAMPLER_CODES,
    TASK_CATEGORY_CODES,
    TASK_ORIGIN_CODES,
)
from iccl.data.dataset import (
    collate_sequences,
    make_family,
    sequence_config_from,
    sequence_dataset_from_config,
    sequence_rng,
    to_tensors,
)
from iccl.data.eval_cells import EvalCell, resolve_eval_cells
from iccl.data.retention_position import (
    REHEARSAL_PROTOCOL,
    RETENTION_PROTOCOL,
    build_paired_position_group,
    build_rehearsal_position_group,
)
from iccl.data.sequences import SequenceSample, build_sequence

EVAL_SEED_OFFSET = 1_000_000
VALIDATION_SEED_OFFSET = 2_000_000
RETENTION_POSITION_SEED_OFFSET = 3_000_000
VALIDATION_SUITE = "validation"

EXPORTED_INFO = (
    "num_modules",
    "num_surplus_tasks",
    "num_prediction_tokens",
    "serialized_length",
    "curriculum_sampler",
    "generation_attempts",
    "task_origin",
    "pre_shuffle_index",
    "generation_category",
    "presentation_category",
    "history_prediction_tokens",
    "history_serialized_tokens",
    "target_first_prediction_index",
    "num_unique_supports_seen",
    "num_modules_covered",
    "original_task_position",
    "intervening_tasks",
    "prediction_token_delay",
    "serialized_token_delay",
    "target_support",
    "constituent_task_exposures",
    "constituent_demo_exposures",
    "prior_target_support_count",
    "prior_target_latent_count",
    "target_modules_seen_before",
    "target_module_pre_exposures",
    "target_module_post_exposures",
    "position_group_id",
    "world_index",
    "sequence_index",
    "rehearsal_mode",
    "support_status",
    "designated_constituent",
    "logical_task_id",
    "pair_id",
    "condition",
    "protocol",
    "exposure_scope",
    "rehearsal_positions",
    "background_task_indices",
    "background_num_modules",
    "background_connected",
    "background_covered",
    "history_connected",
    "history_covered",
)


def _stack(samples: list[SequenceSample], key: str) -> np.ndarray:
    return np.stack([sample.info[key] for sample in samples])


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_arrays(arrays: dict[str, np.ndarray], path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.with_suffix(".npz")
    np.savez_compressed(archive, **arrays)  # pyright: ignore[reportArgumentType]
    metadata = dict(
        meta,
        num_sequences=len(arrays["tokens"]),
        git_commit=_git_commit(),
        archive_sha256=_sha256(archive),
    )
    path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2, default=str))


def export_suite(samples: list[SequenceSample], path: Path, meta: dict[str, Any]) -> None:
    """Write one shape-homogeneous suite and its resolved metadata."""
    lengths = {len(sample.tokens) for sample in samples}
    if len(lengths) != 1:
        raise ValueError(f"suite sequences must share a length, got {sorted(lengths)}")
    for key in ("num_modules", "num_curriculum_tasks", "demo_counts"):
        if all(key in sample.info for sample in samples):
            reference = samples[0].info[key]
            if any(not np.array_equal(reference, sample.info[key]) for sample in samples[1:]):
                raise ValueError(f"suite {meta.get('suite', path.name)} mixes values for {key}")
    arrays = {
        key: np.stack([getattr(sample, key) for sample in samples])
        for key in ("tokens", "token_type", "targets", "loss_mask")
    }
    for key in (
        "latents",
        "demo_counts",
        "boundaries",
        "task_spans",
        "base_mse",
        "num_curriculum_tasks",
        *EXPORTED_INFO,
    ):
        if all(key in sample.info for sample in samples):
            arrays[key] = _stack(samples, key)
    if "world" in samples[0].info:
        pools = [sample.info["world"] for sample in samples]
        for layer in range(len(pools[0].modules)):
            arrays[f"world_modules_{layer}"] = np.stack([pool.modules[layer] for pool in pools])
            arrays[f"world_biases_{layer}"] = np.stack([pool.biases[layer] for pool in pools])
        arrays["world_readout"] = np.stack([pool.readout for pool in pools])
    _write_arrays(arrays, path, meta)


def export_training_validation(data_cfg: DictConfig, path: Path, *, count: int, seed: int) -> None:
    """Freeze a padded sample from the on-the-fly training distribution."""
    dataset = sequence_dataset_from_config(
        data_cfg, base_seed=seed + VALIDATION_SEED_OFFSET, num_sequences=count
    )
    samples = [dataset.build(index) for index in range(count)]
    arrays = {
        key: value.numpy()
        for key, value in collate_sequences([to_tensors(sample) for sample in samples]).items()
    }
    _write_arrays(
        arrays,
        path / VALIDATION_SUITE,
        {
            "suite": VALIDATION_SUITE,
            "capability": "validation",
            "condition": "training_distribution",
            "config": OmegaConf.to_container(data_cfg, resolve=True),
            "seed": seed,
        },
    )


def suite_name(capability: str, condition: str, cell: EvalCell) -> str:
    return f"{capability}__{condition}__{cell.cell_id}"


def _metadata(
    base: dict[str, Any],
    capability: str,
    condition: str,
    cell: EvalCell,
    *,
    sampling_kind: str,
    pair_group: str | None = None,
) -> dict[str, Any]:
    metadata = dict(
        base,
        suite=suite_name(capability, condition, cell),
        capability=capability,
        condition=condition,
        cell_id=cell.cell_id,
        family_memberships=list(cell.family_memberships),
        module_count_status=cell.status,
        num_modules=cell.num_modules,
        num_tasks=cell.num_tasks,
        num_surplus_tasks=cell.num_surplus,
        demos_per_task=cell.demos_per_task,
        demo_counts=cell.demo_counts,
        history_prediction_tokens=cell.prediction_tokens,
        history_serialized_tokens=cell.serialized_tokens,
        sampling_kind=sampling_kind,
    )
    if pair_group is not None:
        metadata["pair_group"] = pair_group
    return metadata


def _sequence_config(base: Any, cell: EvalCell) -> Any:
    return replace(
        base,
        phases=(),
        demos_per_task=cell.demos_per_task,
        task_graph="random",
        graph_ordered=False,
        surplus_tasks=cell.num_surplus,
    )


def balanced_repeat_positions(count: int, tasks: int, seed: int) -> np.ndarray:
    """Assign every history position equally, with deterministic shuffled remainders."""
    if tasks < 1 or count < tasks:
        raise ValueError(
            f"retention balancing requires num_sequences >= T >= 1, got {count}, {tasks}"
        )
    quotient, remainder = divmod(count, tasks)
    rng = sequence_rng(seed + RETENTION_POSITION_SEED_OFFSET, tasks)
    positions = np.repeat(np.arange(tasks, dtype=np.int64), quotient)
    if remainder:
        positions = np.concatenate([positions, rng.permutation(tasks)[:remainder]])
    rng.shuffle(positions)
    return positions


def _clear_archives(out_dir: Path) -> None:
    """Remove stale generated suites so removed schemas cannot be loaded accidentally."""
    for pattern in ("*.npz", "*.meta.json"):
        for path in out_dir.glob(pattern):
            path.unlink()


def export_eval_sets(cfg: DictConfig, *, out_dir: Path | None = None) -> int:
    """Export validation plus every configured capability for each physical cell."""
    data_cfg, eval_cfg = cfg.data, cfg.data.eval_sets
    cells = resolve_eval_cells(data_cfg)
    capabilities = tuple(str(value) for value in eval_cfg.capabilities)
    unknown = set(capabilities) - {"icl", "composition", "retention"}
    if unknown:
        raise ValueError(f"unknown evaluation capabilities: {sorted(unknown)}")
    if "composition" in capabilities:
        composition_controls = {str(value) for value in eval_cfg.composition.controls}
        invalid_controls = composition_controls - {"unexposed", "no_history"}
        if invalid_controls:
            raise ValueError(f"unknown composition controls: {sorted(invalid_controls)}")
        if "unexposed" not in composition_controls:
            raise ValueError("composition requires its unexposed control")
        invalid = [cell.cell_id for cell in cells if cell.num_modules < 4]
        if invalid:
            raise ValueError(f"composition requires M>=4; invalid cells: {invalid}")
    retention_controls = {str(value) for value in eval_cfg.retention.controls}
    invalid_controls = retention_controls - {"unexposed", "shared"}
    if invalid_controls:
        raise ValueError(f"unknown retention controls: {sorted(invalid_controls)}")
    if "retention" in capabilities and "unexposed" not in retention_controls:
        raise ValueError("retention requires its unexposed control")

    out_dir = Path(eval_cfg.out_dir) if out_dir is None else out_dir
    count = int(eval_cfg.num_sequences)
    if count < 1:
        raise ValueError(f"eval_sets.num_sequences must be positive, got {count}")
    worlds = int(eval_cfg.retention.num_worlds)
    monitor_count = int(eval_cfg.retention.monitor_num_sequences)
    if worlds < 1:
        raise ValueError("retention.num_worlds must be positive")
    if (
        "retention" in capabilities
        and not int(eval_cfg.canonical.task_count) <= monitor_count <= worlds
    ):
        raise ValueError(
            "retention monitoring requires canonical T <= monitor_num_sequences <= num_worlds"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_archives(out_dir)
    export_training_validation(data_cfg, out_dir, count=count, seed=int(cfg.seed))

    base_seed = int(cfg.seed) + EVAL_SEED_OFFSET
    base_sequence = sequence_config_from(data_cfg)
    base_meta = {
        "config": OmegaConf.to_container(data_cfg, resolve=True),
        "seed": int(cfg.seed),
        "enum_mappings": {
            "task_origin": TASK_ORIGIN_CODES,
            "task_category": TASK_CATEGORY_CODES,
            "curriculum_sampler": CURRICULUM_SAMPLER_CODES,
        },
    }
    written: set[str] = set()
    stream, suites_written = 0, 1

    def write(
        samples: list[SequenceSample],
        capability: str,
        condition: str,
        cell: EvalCell,
        sampling_kind: str,
        pair_group: str | None = None,
    ) -> None:
        nonlocal suites_written
        name = suite_name(capability, condition, cell)
        if name in written:
            raise ValueError(f"frozen suite name collision: {name}")
        written.add(name)
        metadata = _metadata(
            base_meta,
            capability,
            condition,
            cell,
            sampling_kind=sampling_kind,
            pair_group=pair_group,
        )
        if capability == "retention":
            metadata.update(protocol=RETENTION_PROTOCOL, num_worlds=worlds, sample_scope="full")
            if "canonical" in cell.family_memberships:
                positions = balanced_repeat_positions(monitor_count, cell.num_tasks, int(cfg.seed))
                metadata["monitor_indices"] = (
                    np.arange(monitor_count) * cell.num_tasks + positions
                ).tolist()
                metadata["monitor_selection"] = "balanced-world-position-v1"
        export_suite(samples, out_dir / name, metadata)
        suites_written += 1

    for cell in cells:
        family = make_family(data_cfg, extra_hotness=2, num_modules=cell.num_modules)
        sequence_cfg = _sequence_config(base_sequence, cell)

        if "icl" in capabilities:
            samples = [
                build_sequence(
                    family,
                    sequence_cfg,
                    sequence_rng(base_seed, stream * count + index),
                    include_world=True,
                    fixed_demo_counts=cell.demo_counts,
                )
                for index in range(count)
            ]
            stream += 1
            write(samples, "icl", "ordinary", cell, "natural")

        if "composition" in capabilities:
            conditions = {name: [] for name in ("exposed", "unexposed", "no_history")}
            pair_group = f"composition__{cell.cell_id}"
            for index in range(count):
                pair_id = stream * count + index
                triplet = build_paired_composition_controls(
                    family,
                    sequence_cfg,
                    sequence_rng(base_seed, pair_id),
                    target_demos=cell.demos_per_task,
                    constituent_task_exposures=int(eval_cfg.composition.constituent_task_exposures),
                    fixed_demo_counts=cell.demo_counts,
                )
                for name, sample in zip(conditions, triplet, strict=True):
                    sample.info["pair_id"] = pair_id
                    conditions[name].append(sample)
            stream += 1
            for condition, samples in conditions.items():
                if condition == "no_history" and condition not in eval_cfg.composition.controls:
                    continue
                kind = (
                    "constructively_constrained"
                    if condition == "exposed"
                    else "paired_counterfactual"
                )
                write(samples, "composition", condition, cell, kind, pair_group)

        if "retention" not in capabilities:
            continue
        modes = [str(value) for value in eval_cfg.retention.controls]
        if data_cfg.weighting == "binary":
            modes = [mode for mode in modes if mode != "shared"]
        conditions = {condition: [] for condition in ("repeat", *modes)}
        cell_seed = int.from_bytes(hashlib.sha256(cell.cell_id.encode()).digest()[:4], "little")
        for world_index in range(worlds):
            group = build_paired_position_group(
                family,
                sequence_cfg,
                sequence_rng(base_seed + RETENTION_POSITION_SEED_OFFSET + cell_seed, world_index),
                group_id=world_index,
                control_modes=tuple(modes),
            )
            for condition, samples in group.items():
                conditions[condition].extend(samples)
        for condition, samples in conditions.items():
            write(
                samples,
                "retention",
                condition,
                cell,
                "paired_history_intervention",
                f"retention__{cell.cell_id}",
            )

    print(f"exported {suites_written} suites to {out_dir}/")
    return suites_written


def export_rehearsal_sets(cfg: DictConfig, *, out_dir: Path | None = None) -> int:
    """Export the explicitly enabled canonical rehearsal experiment."""
    data_cfg, eval_cfg = cfg.data, cfg.data.eval_sets
    rehearsal_cfg = eval_cfg.rehearsal
    if not rehearsal_cfg.enabled:
        return 0
    worlds = int(rehearsal_cfg.num_worlds)
    if worlds < 1:
        raise ValueError("rehearsal.num_worlds must be positive")
    cell = next(
        cell for cell in resolve_eval_cells(data_cfg) if "canonical" in cell.family_memberships
    )
    modes = tuple(
        str(mode)
        for mode in eval_cfg.retention.controls
        if mode != "shared" or data_cfg.weighting != "binary"
    )
    family = make_family(data_cfg, extra_hotness=2, num_modules=cell.num_modules)
    sequence_cfg = _sequence_config(sequence_config_from(data_cfg), cell)
    conditions = {condition: [] for condition in ("repeat", *modes)}
    for group_id in range(worlds):
        generated = build_rehearsal_position_group(
            family,
            sequence_cfg,
            sequence_rng(
                int(cfg.seed) + EVAL_SEED_OFFSET + RETENTION_POSITION_SEED_OFFSET + 1, group_id
            ),
            group_id=group_id,
            control_modes=modes,
        )
        for condition, samples in generated.items():
            conditions[condition].extend(samples)
    base = {
        "config": OmegaConf.to_container(data_cfg, resolve=True),
        "seed": int(cfg.seed),
        "num_worlds": worlds,
        "protocol": REHEARSAL_PROTOCOL,
        "sample_scope": "full",
        "exposure_scope": "original_encounter",
    }
    out_dir = Path(eval_cfg.out_dir) if out_dir is None else out_dir
    for condition, samples in conditions.items():
        metadata = _metadata(
            base,
            "rehearsal",
            condition,
            cell,
            sampling_kind="paired_history_intervention",
            pair_group=f"rehearsal__{cell.cell_id}",
        )
        export_suite(samples, out_dir / metadata["suite"], metadata)
    return len(conditions)


def load_suite(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.with_suffix(".npz")) as data:
        return dict(data)


def suite_paths(
    eval_dir: Path, suite: str, explicit_path: str | Path | None = None
) -> tuple[Path, Path]:
    """Resolve and validate one array/metadata pair."""
    base = Path(explicit_path) if explicit_path is not None else eval_dir / suite
    arrays, metadata = base.with_suffix(".npz"), base.with_suffix(".meta.json")
    if not arrays.exists():
        raise FileNotFoundError(f"frozen suite not found: {arrays}")
    if not metadata.exists():
        raise FileNotFoundError(f"frozen suite metadata not found: {metadata}")
    return arrays, metadata


def load_suite_metadata(path: Path) -> dict[str, Any]:
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"suite metadata must be a JSON object: {path}")
    return metadata
