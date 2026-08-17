"""Export of frozen evaluation and analysis sets.

Each suite is written as one ``.npz`` of stacked arrays plus a ``meta.json``
recording the resolved config, git commit, and generation parameters, so
reported numbers and interp analyses always run on byte-identical sequences.
Worlds (module pools) are included as probe targets for the interp stage.

The suites: "in_dist" (standard curriculum sequences), "composite" (curriculum
plus a few-shot composite final task) with its paired "composite_control" (same
world and final task, no history), one "structural_<graph>" suite per
overlap-graph family in eval_sets.structural_graphs, "retention" (curriculum
sequences that re-demonstrate the first task at the end), and one
position-matched control per eval_sets.retention.controls mode, each repeating
its retention sequence with a different task in that final block.

Suites take consecutive blocks of the eval sequence-index space, so appending a
suite leaves the earlier ones' sequences untouched.
"""

import json
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from iccl.data.dataset import (
    make_family,
    module_count_config_from,
    sequence_config_from,
    sequence_rng,
)
from iccl.data.sequences import (
    CURRICULUM_SAMPLER_CODES,
    TASK_CATEGORY_CODES,
    TASK_ORIGIN_CODES,
    FinalTaskConfig,
    SequenceSample,
    build_paired_composition_controls,
    build_paired_control,
    build_paired_retention_control,
    build_sequence,
)

EVAL_SEED_OFFSET = 1_000_000  # keeps eval streams disjoint from training indices

# Suite name per retention-control mode, shared by the exporter and the metrics
# so the two agree on what is on disk.
RETENTION_CONTROL_SUITES = {"novel": "retention_control", "shared": "retention_control_shared"}

OPTIONAL_INFO_KEYS = (
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


def _stack_info(samples: list[SequenceSample], key: str) -> np.ndarray:
    return np.stack([s.info[key] for s in samples])


def export_suite(
    samples: list[SequenceSample],
    path: Path,
    meta: dict[str, Any],
    *,
    include_extended_info: bool = False,
) -> None:
    """Write a suite of equally-shaped sequences to ``<path>.npz`` + ``<path>.meta.json``."""
    lengths = {s.tokens.shape[0] for s in samples}
    if len(lengths) != 1:
        raise ValueError(f"suite sequences must share a length, got {sorted(lengths)}")
    arrays: dict[str, np.ndarray] = {
        "tokens": np.stack([s.tokens for s in samples]),
        "token_type": np.stack([s.token_type for s in samples]),
        "targets": np.stack([s.targets for s in samples]),
        "loss_mask": np.stack([s.loss_mask for s in samples]),
        "latents": _stack_info(samples, "latents"),
        "demo_counts": _stack_info(samples, "demo_counts"),
        "boundaries": _stack_info(samples, "boundaries"),
        "task_spans": _stack_info(samples, "task_spans"),
        "base_mse": _stack_info(samples, "base_mse"),
        "num_curriculum_tasks": _stack_info(samples, "num_curriculum_tasks"),
    }
    if include_extended_info:
        for key in OPTIONAL_INFO_KEYS:
            if all(key in sample.info for sample in samples):
                arrays[key] = _stack_info(samples, key)
    if "world" in samples[0].info:
        pools = [s.info["world"] for s in samples]
        for layer in range(len(pools[0].modules)):
            arrays[f"world_modules_{layer}"] = np.stack([p.modules[layer] for p in pools])
            arrays[f"world_biases_{layer}"] = np.stack([p.biases[layer] for p in pools])
        arrays["world_readout"] = np.stack([p.readout for p in pools])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), allow_pickle=False, **arrays)
    meta = dict(meta, num_sequences=len(samples), git_commit=_git_commit())
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _export_legacy_eval_sets(cfg: DictConfig) -> int:
    """Write every frozen eval suite to ``cfg.data.eval_sets.out_dir``.

    Returns the number of suites written. The shared-module retention control is
    skipped under a binary weighting, where a task is its support and the
    control would duplicate the retention suite.
    """
    data_cfg = cfg.data
    composite_cfg = data_cfg.eval_sets.composite
    family = make_family(data_cfg, extra_hotness=composite_cfg.hotness)
    seq_cfg = sequence_config_from(data_cfg)
    final = FinalTaskConfig(
        mode="composite", hotness=composite_cfg.hotness, num_demos=composite_cfg.num_demos
    )
    out_dir = Path(data_cfg.eval_sets.out_dir)
    num = data_cfg.eval_sets.num_sequences
    meta = {"config": OmegaConf.to_container(data_cfg, resolve=True), "seed": cfg.seed}
    base_seed = cfg.seed + EVAL_SEED_OFFSET

    in_dist, composite, control = [], [], []
    for i in range(num):
        rng = sequence_rng(base_seed, i)
        in_dist.append(build_sequence(family, seq_cfg, rng, include_world=True))
        rng = sequence_rng(base_seed, num + i)
        seq = build_sequence(family, seq_cfg, rng, final_task=final, include_world=True)
        composite.append(seq)
        control.append(build_paired_control(family, seq_cfg, seq, final, rng))

    export_suite(in_dist, out_dir / "in_dist", dict(meta, suite="in_dist"))
    export_suite(composite, out_dir / "composite", dict(meta, suite="composite"))
    export_suite(control, out_dir / "composite_control", dict(meta, suite="composite_control"))

    num_suites = 3
    for graph_offset, graph in enumerate(data_cfg.eval_sets.structural_graphs):
        graph_cfg = replace(seq_cfg, task_graph=graph)
        suite = [
            build_sequence(
                family,
                graph_cfg,
                sequence_rng(base_seed, (2 + graph_offset) * num + i),
                include_world=True,
            )
            for i in range(num)
        ]
        export_suite(
            suite, out_dir / f"structural_{graph}", dict(meta, suite=f"structural_{graph}")
        )
        num_suites += 1

    retention_base = (2 + len(data_cfg.eval_sets.structural_graphs)) * num
    retention = [
        build_sequence(
            family,
            seq_cfg,
            sequence_rng(base_seed, retention_base + i),
            revisit_demos=data_cfg.eval_sets.retention.revisit_demos,
            include_world=True,
        )
        for i in range(num)
    ]
    export_suite(retention, out_dir / "retention", dict(meta, suite="retention"))
    num_suites += 1

    modes = list(data_cfg.eval_sets.retention.controls)
    if "shared" in modes and data_cfg.weighting == "binary":
        print(
            "skipping the shared-module retention control: under weighting=binary a task is "
            "its support, so the control would duplicate the retention suite"
        )
        modes.remove("shared")
    for offset, mode in enumerate(modes):
        base = retention_base + (1 + offset) * num
        suite = [
            build_paired_retention_control(
                family, retention[i], sequence_rng(base_seed, base + i), mode=mode
            )
            for i in range(num)
        ]
        name = RETENTION_CONTROL_SUITES[mode]
        suite_meta: dict[str, Any] = dict(meta, suite=name, control_mode=mode)
        # The shared control reuses the revisited task's modules with different
        # weights; how different bounds how much episodic help leaks into it, so
        # the frozen set records the spread.
        if mode == "shared":
            distances = [
                float(np.linalg.norm(c.info["latents"][-1] - r.info["latents"][-1]))
                for c, r in zip(suite, retention, strict=True)
            ]
            suite_meta["latent_distance_to_revisited"] = {
                "mean": float(np.mean(distances)),
                "min": float(np.min(distances)),
            }
        export_suite(suite, out_dir / name, suite_meta)
        num_suites += 1

    print(f"exported {num_suites} suites x {num} sequences to {out_dir}/")
    return num_suites


@dataclass(frozen=True)
class EvalCell:
    """One homogeneous frozen history cell."""

    slice: str
    status: str
    num_modules: int
    num_tasks: int
    num_surplus: int
    demo_counts: tuple[int, ...]
    variant: str = ""

    @property
    def prediction_tokens(self) -> int:
        return sum(self.demo_counts)

    @property
    def serialized_tokens(self) -> int:
        return 2 * self.prediction_tokens + self.num_tasks


def evaluation_module_counts(data_cfg: DictConfig) -> tuple[tuple[int, ...], dict[int, str]]:
    """Resolve the explicit seen, interpolation, and OOD module-count union."""
    train = module_count_config_from(data_cfg)
    eval_cfg = data_cfg.eval_sets.module_counts
    if eval_cfg.seen_selection != "endpoints_and_canonical":
        raise ValueError(
            "eval_sets.module_counts.seen_selection must be endpoints_and_canonical, got "
            f"{eval_cfg.seen_selection!r}"
        )
    canonical = int(eval_cfg.canonical_seen)
    if canonical not in train.allowed:
        raise ValueError(
            f"canonical_seen={canonical} is absent from training support {train.allowed}"
        )
    seen = tuple(sorted({train.min, canonical, train.max}))
    held_out = train.held_out
    ood = tuple(int(value) for value in eval_cfg.get("ood", []))
    if len(set(ood)) != len(ood):
        raise ValueError(f"eval OOD module counts contain duplicates: {ood}")
    invalid_ood = [value for value in ood if value <= train.max]
    if invalid_ood:
        raise ValueError(
            f"OOD module counts must be greater than M_max={train.max}, got {invalid_ood}"
        )
    overlap = (set(seen) & set(held_out)) | (set(seen) & set(ood)) | (set(held_out) & set(ood))
    if overlap:
        raise ValueError(f"evaluation module-count groups overlap at {sorted(overlap)}")
    status = {value: "seen" for value in seen}
    status.update({value: "heldout" for value in held_out})
    status.update({value: "ood" for value in ood})
    values = tuple(sorted(status))
    if not values:
        raise ValueError("evaluation module-count set is empty")
    return values, status


def _even_demo_counts(total: int, tasks: int, seed: int) -> tuple[int, ...]:
    if tasks < 1 or total < tasks:
        raise ValueError(
            f"prediction-token budget must assign >=1 demo per task, got B={total}, T={tasks}"
        )
    quotient, remainder = divmod(total, tasks)
    counts = np.full(tasks, quotient, dtype=np.int64)
    if remainder:
        rng = np.random.Generator(
            np.random.Philox(key=np.array([seed, tasks], dtype=np.uint64))
        )
        positions = rng.choice(tasks, size=remainder, replace=False)
        counts[positions] += 1
    return tuple(int(value) for value in counts)


def _cell(
    slice_name: str,
    status: str,
    num_modules: int,
    num_tasks: int,
    demo_counts: tuple[int, ...],
    variant: str = "",
) -> EvalCell:
    surplus = num_tasks - (num_modules - 1)
    if surplus < 0:
        raise ValueError(
            f"slice={slice_name} cannot connectedly cover M={num_modules} with T={num_tasks}"
        )
    if len(demo_counts) != num_tasks or any(count < 1 for count in demo_counts):
        raise ValueError(
            f"slice={slice_name} needs {num_tasks} positive demo counts, got {demo_counts}"
        )
    return EvalCell(
        slice_name,
        status,
        num_modules,
        num_tasks,
        surplus,
        demo_counts,
        variant,
    )


def resolve_eval_cells(data_cfg: DictConfig, seed: int) -> list[EvalCell]:
    """Expand named one-dimensional structural slices into homogeneous cells."""
    module_values, statuses = evaluation_module_counts(data_cfg)
    slices = data_cfg.eval_sets.structural_slices
    cells: list[EvalCell] = []
    for slice_name in slices.enabled:
        config = slices[slice_name]
        match slice_name:
            case "fixed_surplus":
                surplus = int(config.surplus_tasks)
                demos = int(config.history_demos_per_task)
                for modules in module_values:
                    tasks = modules - 1 + surplus
                    cells.append(
                        _cell(slice_name, statuses[modules], modules, tasks, (demos,) * tasks)
                    )
            case "matched_task_count":
                tasks = int(config.task_count)
                demos = int(config.history_demos_per_task)
                for modules in module_values:
                    cells.append(
                        _cell(slice_name, statuses[modules], modules, tasks, (demos,) * tasks)
                    )
            case "matched_prediction_tokens":
                budget = int(config.prediction_tokens)
                surplus = int(config.surplus_tasks)
                for modules in module_values:
                    tasks = modules - 1 + surplus
                    demos = _even_demo_counts(budget, tasks, seed + modules)
                    cells.append(_cell(slice_name, statuses[modules], modules, tasks, demos))
            case "task_count":
                modules = int(config.module_count)
                if modules not in statuses:
                    raise ValueError(f"task_count slice M={modules} is not in M_eval")
                demos = int(config.history_demos_per_task)
                for tasks_raw in config["values"]:
                    tasks = int(tasks_raw)
                    cells.append(
                        _cell(
                            slice_name,
                            statuses[modules],
                            modules,
                            tasks,
                            (demos,) * tasks,
                        )
                    )
            case "history_demos":
                modules, tasks = int(config.module_count), int(config.task_count)
                if modules not in statuses:
                    raise ValueError(f"history_demos slice M={modules} is not in M_eval")
                for demos_raw in config["values"]:
                    demos = int(demos_raw)
                    cells.append(
                        _cell(
                            slice_name,
                            statuses[modules],
                            modules,
                            tasks,
                            (demos,) * tasks,
                            variant=f"d{demos:03d}",
                        )
                    )
            case "matched_serialized_prefix":
                modules = int(config.module_count)
                if modules not in statuses:
                    raise ValueError(
                        f"matched_serialized_prefix slice M={modules} is not in M_eval"
                    )
                length = int(config.serialized_tokens)
                for tasks_raw in config.task_counts:
                    tasks = int(tasks_raw)
                    remainder = length - tasks
                    if remainder < 0 or remainder % 2:
                        raise ValueError(
                            "matched serialized prefix needs non-negative even L-T, got "
                            f"L={length}, T={tasks}"
                        )
                    demos = _even_demo_counts(remainder // 2, tasks, seed + tasks)
                    cells.append(_cell(slice_name, statuses[modules], modules, tasks, demos))
            case "demo_allocation":
                modules, tasks = int(config.module_count), int(config.task_count)
                if modules not in statuses:
                    raise ValueError(f"demo_allocation slice M={modules} is not in M_eval")
                budget = int(config.prediction_tokens)
                uniform = _even_demo_counts(budget, tasks, seed)
                if len(set(uniform)) != 1 or tasks % 2:
                    raise ValueError(
                        "demo allocation patterns require an even task count and an "
                        f"integer mean, got T={tasks}, B={budget}"
                    )
                mean = uniform[0]
                low, high = mean // 2, mean + (mean - mean // 2)
                alternating = tuple(low if i % 2 == 0 else high for i in range(tasks))
                front = (high,) * (tasks // 2) + (low,) * (tasks // 2)
                patterns = {
                    "uniform": uniform,
                    "alternating": alternating,
                    "front_loaded": front,
                    "back_loaded": tuple(reversed(front)),
                }
                for pattern in config.patterns:
                    if pattern not in patterns:
                        raise ValueError(f"unknown demo-allocation pattern: {pattern}")
                    counts = patterns[pattern]
                    if sum(counts) != budget or min(counts) < 1:
                        raise ValueError(
                            f"demo-allocation pattern {pattern} does not preserve B={budget}: "
                            f"{counts}"
                        )
                    cells.append(
                        _cell(
                            slice_name,
                            statuses[modules],
                            modules,
                            tasks,
                            counts,
                            variant=str(pattern),
                        )
                    )
            case _:
                raise ValueError(f"unknown structural slice: {slice_name}")
    return cells


def _suite_name(capability: str, condition: str, cell: EvalCell) -> str:
    suffix = f"__{cell.variant}" if cell.variant else ""
    return (
        f"{capability}__{condition}__{cell.slice}__{cell.status}__"
        f"m{cell.num_modules:02d}__t{cell.num_tasks:02d}__b{cell.prediction_tokens:04d}"
        f"{suffix}"
    )


def _cell_metadata(
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
        suite=_suite_name(capability, condition, cell),
        capability=capability,
        condition=condition,
        structural_slice=cell.slice,
        module_count_status=cell.status,
        num_modules=cell.num_modules,
        num_tasks=cell.num_tasks,
        num_surplus_tasks=cell.num_surplus,
        demo_counts=cell.demo_counts,
        history_prediction_tokens=cell.prediction_tokens,
        history_serialized_tokens=cell.serialized_tokens,
        sampling_kind=sampling_kind,
    )
    if pair_group is not None:
        metadata["pair_group"] = pair_group
    return metadata


def _cell_sequence_config(base: Any, cell: EvalCell) -> Any:
    return replace(
        base,
        phases=(),
        demos_per_task=cell.demo_counts[0],
        task_graph="random",
        graph_ordered=False,
        surplus_tasks=cell.num_surplus,
    )


def _export_variable_eval_sets(cfg: DictConfig) -> int:
    data_cfg = cfg.data
    eval_cfg = data_cfg.eval_sets
    cells = resolve_eval_cells(data_cfg, int(cfg.seed))
    capabilities = tuple(str(value) for value in eval_cfg.capabilities.enabled)
    unknown = set(capabilities) - {"icl", "composition", "retention"}
    if unknown:
        raise ValueError(f"unknown evaluation capabilities: {sorted(unknown)}")
    if "matched_prefix" not in eval_cfg.capabilities.composition.controls:
        raise ValueError("composition capability requires the matched_prefix control")

    out_dir = Path(eval_cfg.out_dir)
    num_sequences = int(eval_cfg.num_sequences)
    base_seed = int(cfg.seed) + EVAL_SEED_OFFSET
    base_seq_cfg = sequence_config_from(data_cfg)
    base_meta = {
        "config": OmegaConf.to_container(data_cfg, resolve=True),
        "seed": cfg.seed,
        "enum_mappings": {
            "task_origin": TASK_ORIGIN_CODES,
            "task_category": TASK_CATEGORY_CODES,
            "curriculum_sampler": CURRICULUM_SAMPLER_CODES,
        },
    }
    written: set[str] = set()
    suite_count = 0
    stream_block = 0

    def write(
        samples: list[SequenceSample],
        capability: str,
        condition: str,
        cell: EvalCell,
        *,
        sampling_kind: str,
        pair_group: str | None = None,
    ) -> None:
        nonlocal suite_count
        name = _suite_name(capability, condition, cell)
        if name in written:
            raise ValueError(f"frozen suite name collision: {name}")
        written.add(name)
        metadata = _cell_metadata(
            base_meta,
            capability,
            condition,
            cell,
            sampling_kind=sampling_kind,
            pair_group=pair_group,
        )
        export_suite(samples, out_dir / name, metadata, include_extended_info=True)
        suite_count += 1

    for cell in cells:
        family = make_family(data_cfg, extra_hotness=2, num_modules=cell.num_modules)
        seq_cfg = _cell_sequence_config(base_seq_cfg, cell)
        if "icl" in capabilities:
            samples = [
                build_sequence(
                    family,
                    seq_cfg,
                    sequence_rng(base_seed, stream_block * num_sequences + i),
                    include_world=True,
                    fixed_demo_counts=cell.demo_counts,
                )
                for i in range(num_sequences)
            ]
            stream_block += 1
            write(samples, "icl", "ordinary", cell, sampling_kind="natural")

        if "composition" in capabilities:
            constituent: list[SequenceSample] = []
            matched: list[SequenceSample] = []
            no_history: list[SequenceSample] = []
            pair_group = _suite_name("composition", "paired", cell)
            for i in range(num_sequences):
                pair_id = stream_block * num_sequences + i
                triplet = build_paired_composition_controls(
                    family,
                    seq_cfg,
                    sequence_rng(base_seed, pair_id),
                    target_demos=int(eval_cfg.capabilities.composition.target_demos),
                    constituent_task_exposures=int(
                        eval_cfg.capabilities.composition.constituent_task_exposures
                    ),
                    fixed_demo_counts=cell.demo_counts,
                )
                for sample in triplet:
                    sample.info["pair_id"] = pair_id
                constituent.append(triplet[0])
                matched.append(triplet[1])
                no_history.append(triplet[2])
            stream_block += 1
            write(
                constituent,
                "composition",
                "constituent",
                cell,
                sampling_kind="constructively_constrained",
                pair_group=pair_group,
            )
            write(
                matched,
                "composition",
                "matched_prefix",
                cell,
                sampling_kind="paired_counterfactual",
                pair_group=pair_group,
            )
            if "no_history" in eval_cfg.capabilities.composition.controls:
                write(
                    no_history,
                    "composition",
                    "no_history",
                    cell,
                    sampling_kind="paired_counterfactual",
                    pair_group=pair_group,
                )

        if "retention" in capabilities:
            repeats: list[SequenceSample] = []
            controls: dict[str, list[SequenceSample]] = {"novel": [], "shared": []}
            modes = [str(value) for value in eval_cfg.capabilities.retention.controls]
            if "novel" not in modes:
                raise ValueError("retention capability requires the novel control")
            if data_cfg.weighting == "binary":
                modes = [mode for mode in modes if mode != "shared"]
            pair_group = _suite_name("retention", "paired", cell)
            for i in range(num_sequences):
                pair_id = stream_block * num_sequences + i
                rng = sequence_rng(base_seed, pair_id)
                repeat = build_sequence(
                    family,
                    seq_cfg,
                    rng,
                    revisit_demos=int(eval_cfg.capabilities.retention.revisit_demos),
                    include_world=True,
                    fixed_demo_counts=cell.demo_counts,
                )
                repeat.info["pair_id"] = pair_id
                repeats.append(repeat)
                for mode in modes:
                    control = build_paired_retention_control(family, repeat, rng, mode=mode)
                    control.info["pair_id"] = pair_id
                    controls[mode].append(control)
            stream_block += 1
            write(
                repeats,
                "retention",
                "repeat",
                cell,
                sampling_kind="constructively_constrained",
                pair_group=pair_group,
            )
            for mode in modes:
                write(
                    controls[mode],
                    "retention",
                    mode,
                    cell,
                    sampling_kind="paired_counterfactual",
                    pair_group=pair_group,
                )

    print(f"exported {suite_count} suites x {num_sequences} sequences to {out_dir}/")
    return suite_count


def export_eval_sets(cfg: DictConfig) -> int:
    """Export the legacy suite family or the configured variable-world matrix."""
    if cfg.data.eval_sets.get("capabilities") is None:
        return _export_legacy_eval_sets(cfg)
    return _export_variable_eval_sets(cfg)


def load_suite(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.with_suffix(".npz")) as data:
        return dict(data)


def suite_paths(
    eval_dir: Path,
    suite_name: str,
    explicit_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve and validate the array and metadata paths for one frozen suite."""
    suite_path = Path(explicit_path) if explicit_path is not None else eval_dir / suite_name
    suite_path = suite_path.with_suffix(".npz")
    metadata_path = suite_path.with_suffix(".meta.json")
    if not suite_path.exists():
        raise FileNotFoundError(f"frozen suite not found: {suite_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"frozen suite metadata not found: {metadata_path}")
    return suite_path, metadata_path


def load_suite_metadata(path: Path) -> dict[str, Any]:
    """Load a frozen suite's JSON-object metadata sidecar."""
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
