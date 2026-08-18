"""Generate reproducible frozen suites for validation and capability evaluation.

The exporter expands homogeneous cells from the data config into deterministic
sequence archives. Each array file is paired with metadata recording its
resolved configuration and generation provenance, so training-time monitoring
and standalone evaluation score the same examples.
"""

import json
import subprocess
from dataclasses import replace
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from iccl.data.controls import (
    build_paired_composition_controls,
    build_paired_retention_control,
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
from iccl.data.sequences import SequenceSample, build_sequence

EVAL_SEED_OFFSET = 1_000_000
VALIDATION_SEED_OFFSET = 2_000_000
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
    "intervening_tasks",
    "prediction_token_delay",
    "serialized_token_delay",
    "target_support",
    "constituent_task_exposures",
    "constituent_demo_exposures",
    "prior_target_support_count",
    "pair_id",
)


def _stack(samples: list[SequenceSample], key: str) -> np.ndarray:
    return np.stack([sample.info[key] for sample in samples])


def _write_arrays(arrays: dict[str, np.ndarray], path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), allow_pickle=False, **arrays)
    metadata = dict(meta, num_sequences=len(arrays["tokens"]), git_commit=_git_commit())
    path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2, default=str))


def export_suite(samples: list[SequenceSample], path: Path, meta: dict[str, Any]) -> None:
    """Write one equally-shaped suite and its resolved metadata."""
    lengths = {len(sample.tokens) for sample in samples}
    if len(lengths) != 1:
        raise ValueError(f"suite sequences must share a length, got {sorted(lengths)}")
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
        data_cfg,
        base_seed=seed + VALIDATION_SEED_OFFSET,
        num_sequences=count,
    )
    samples = [dataset.build(index) for index in range(count)]
    batch = collate_sequences([to_tensors(sample) for sample in samples])
    arrays = {key: value.numpy() for key, value in batch.items()}
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
    variant = f"__{cell.variant}" if cell.variant else ""
    return (
        f"{capability}__{condition}__{cell.slice}__{cell.status}__"
        f"m{cell.num_modules:02d}__t{cell.num_tasks:02d}__b{cell.prediction_tokens:04d}"
        f"{variant}"
    )


def _metadata(
    base: dict[str, Any],
    capability: str,
    condition: str,
    cell: EvalCell,
    *,
    sampling_kind: str,
    pair_group: str | None = None,
    demo_counts: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    metadata = dict(
        base,
        suite=suite_name(capability, condition, cell),
        capability=capability,
        condition=condition,
        structural_slice=cell.slice,
        variant=cell.variant,
        module_count_status=cell.status,
        num_modules=cell.num_modules,
        num_tasks=cell.num_tasks,
        num_surplus_tasks=cell.num_surplus,
        demo_counts=demo_counts or cell.demo_counts,
        history_prediction_tokens=cell.prediction_tokens,
        history_serialized_tokens=cell.serialized_tokens,
        sampling_kind=sampling_kind,
    )
    if pair_group:
        metadata["pair_group"] = pair_group
    return metadata


def _sequence_config(base: Any, cell: EvalCell) -> Any:
    return replace(
        base,
        phases=(),
        demos_per_task=cell.demo_counts[0],
        task_graph="random",
        graph_ordered=False,
        surplus_tasks=cell.num_surplus,
    )


def export_eval_sets(cfg: DictConfig) -> int:
    """Export every configured capability and structural cell."""
    data_cfg, eval_cfg = cfg.data, cfg.data.eval_sets
    cells = resolve_eval_cells(data_cfg, int(cfg.seed))
    capabilities = tuple(str(value) for value in eval_cfg.capabilities.enabled)
    unknown = set(capabilities) - {"icl", "composition", "retention"}
    if unknown:
        raise ValueError(f"unknown evaluation capabilities: {sorted(unknown)}")
    if (
        "composition" in capabilities
        and "matched_prefix" not in eval_cfg.capabilities.composition.controls
    ):
        raise ValueError("composition requires its matched_prefix control")

    out_dir = Path(eval_cfg.out_dir)
    count = int(eval_cfg.num_sequences)
    base_seed = int(cfg.seed) + EVAL_SEED_OFFSET
    export_training_validation(data_cfg, out_dir, count=count, seed=int(cfg.seed))
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
    matched_slices = {"fixed_surplus", "matched_task_count", "matched_prediction_tokens"}
    fixed_constituent_demos = {
        name: min(min(cell.demo_counts) for cell in cells if cell.slice == name)
        for name in matched_slices
        if any(cell.slice == name for cell in cells)
    }
    written: set[str] = set()
    stream, suites_written = 0, 1

    def write(
        samples: list[SequenceSample],
        capability: str,
        condition: str,
        cell: EvalCell,
        sampling_kind: str,
        *,
        pair_group: str | None = None,
        demo_counts: tuple[int, ...] | None = None,
    ) -> None:
        nonlocal suites_written
        name = suite_name(capability, condition, cell)
        if name in written:
            raise ValueError(f"frozen suite name collision: {name}")
        written.add(name)
        export_suite(
            samples,
            out_dir / name,
            _metadata(
                base_meta,
                capability,
                condition,
                cell,
                sampling_kind=sampling_kind,
                pair_group=pair_group,
                demo_counts=demo_counts,
            ),
        )
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
            conditions = {name: [] for name in ("constituent", "matched_prefix", "no_history")}
            pair_group = suite_name("composition", "paired", cell)
            for index in range(count):
                pair_id = stream * count + index
                triplet = build_paired_composition_controls(
                    family,
                    sequence_cfg,
                    sequence_rng(base_seed, pair_id),
                    target_demos=int(eval_cfg.capabilities.composition.target_demos),
                    constituent_task_exposures=int(
                        eval_cfg.capabilities.composition.constituent_task_exposures
                    ),
                    fixed_demo_counts=cell.demo_counts,
                    constituent_demo_count=fixed_constituent_demos.get(cell.slice),
                )
                for name, sample in zip(conditions, triplet, strict=True):
                    sample.info["pair_id"] = pair_id
                    conditions[name].append(sample)
            stream += 1
            resolved_demos = tuple(
                int(value)
                for value in conditions["constituent"][0].info["demo_counts"][: cell.num_tasks]
            )
            for condition, samples in conditions.items():
                if (
                    condition == "no_history"
                    and condition not in eval_cfg.capabilities.composition.controls
                ):
                    continue
                kind = (
                    "constructively_constrained"
                    if condition == "constituent"
                    else "paired_counterfactual"
                )
                write(
                    samples,
                    "composition",
                    condition,
                    cell,
                    kind,
                    pair_group=pair_group,
                    demo_counts=resolved_demos,
                )

        if "retention" in capabilities:
            modes = [str(value) for value in eval_cfg.capabilities.retention.controls]
            if "novel" not in modes:
                raise ValueError("retention requires its novel control")
            if data_cfg.weighting == "binary":
                modes = [mode for mode in modes if mode != "shared"]
            repeats: list[SequenceSample] = []
            controls: dict[str, list[SequenceSample]] = {mode: [] for mode in modes}
            pair_group = suite_name("retention", "paired", cell)
            for index in range(count):
                pair_id = stream * count + index
                rng = sequence_rng(base_seed, pair_id)
                for _ in range(sequence_cfg.max_attempts):
                    repeat = build_sequence(
                        family,
                        sequence_cfg,
                        rng,
                        revisit_demos=int(eval_cfg.capabilities.retention.revisit_demos),
                        include_world=True,
                        fixed_demo_counts=cell.demo_counts,
                    )
                    history = repeat.info["latents"][: cell.num_tasks]
                    supports = {tuple(np.flatnonzero(latent)) for latent in history}
                    if len(supports) < comb(cell.num_modules, sequence_cfg.hotness):
                        break
                else:
                    raise RuntimeError(
                        f"retention cell {pair_group} exhausted all supports in "
                        f"{sequence_cfg.max_attempts} sampled histories"
                    )
                repeat.info["pair_id"] = pair_id
                repeats.append(repeat)
                for mode in modes:
                    control = build_paired_retention_control(family, repeat, rng, mode=mode)
                    control.info["pair_id"] = pair_id
                    controls[mode].append(control)
            stream += 1
            write(
                repeats,
                "retention",
                "repeat",
                cell,
                "constructively_constrained",
                pair_group=pair_group,
            )
            for mode, samples in controls.items():
                write(
                    samples, "retention", mode, cell, "paired_counterfactual", pair_group=pair_group
                )

    print(f"exported {suites_written} suites x {count} sequences to {out_dir}/")
    return suites_written


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


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
