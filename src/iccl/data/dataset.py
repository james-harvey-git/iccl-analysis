"""Torch dataset wrapper with deterministic, order-independent seeding.

Sequence ``i`` under base seed ``s`` is generated on CPU from a
``numpy.random.Philox`` generator keyed on ``(s, i)``, so the stream is a pure
function of (config, seed, index) — independent of dataloader workers, batch
size, device, and access order. The golden-stream tests pin the stream against
accidental sampler changes.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import IterableDataset, get_worker_info

from iccl.data.sequences import (
    TOKEN_PAD,
    DemoCountConfig,
    FinalTaskConfig,
    PhaseConfig,
    SequenceConfig,
    SequenceSample,
    assert_feasible,
    build_sequence,
)
from iccl.data.teacher import HyperTeacher, TeacherConfig


def sequence_rng(base_seed: int, index: int) -> np.random.Generator:
    """The canonical per-sequence RNG: Philox keyed on (base_seed, index)."""
    return np.random.Generator(np.random.Philox(key=np.array([base_seed, index], dtype=np.uint64)))


@dataclass(frozen=True)
class ModuleCountConfig:
    """Uniform distribution over explicitly allowed module counts."""

    min: int
    max: int
    held_out: tuple[int, ...]
    allowed: tuple[int, ...]

    def sample(self, rng: np.random.Generator) -> int:
        """Sample uniformly over allowed values; a singleton consumes no RNG."""
        if len(self.allowed) == 1:
            return self.allowed[0]
        return self.allowed[int(rng.integers(len(self.allowed)))]


def module_count_config_from(cfg: DictConfig) -> ModuleCountConfig:
    """Normalize scalar or bounded ``num_modules`` configuration."""
    spec = cfg.num_modules
    if isinstance(spec, int):
        if spec < 2:
            raise ValueError(f"num_modules must be >=2, got {spec}")
        return ModuleCountConfig(spec, spec, (), (spec,))

    minimum, maximum = int(spec.min), int(spec.max)
    held_out = tuple(int(value) for value in spec.get("held_out", []))
    if minimum < 2 or minimum > maximum:
        raise ValueError(
            f"num_modules requires 2 <= min <= max, got min={minimum}, max={maximum}"
        )
    if len(set(held_out)) != len(held_out):
        raise ValueError(f"num_modules.held_out contains duplicates: {held_out}")
    invalid = [value for value in held_out if not minimum < value < maximum]
    if invalid:
        raise ValueError(
            "num_modules.held_out values must be strictly inside the training envelope; "
            f"got {invalid} for [{minimum}, {maximum}]"
        )
    excluded = set(held_out)
    allowed = tuple(value for value in range(minimum, maximum + 1) if value not in excluded)
    if not allowed:
        raise ValueError("num_modules training support is empty")
    return ModuleCountConfig(minimum, maximum, tuple(sorted(held_out)), allowed)


def teacher_config_from(cfg: DictConfig, num_modules: int | None = None) -> TeacherConfig:
    concrete_modules = (
        module_count_config_from(cfg).allowed[0] if num_modules is None else num_modules
    )
    return TeacherConfig(
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        use_bias=cfg.use_bias,
        num_modules=concrete_modules,
        scale=cfg.scale,
        weighting=cfg.weighting,
    )


def sequence_config_from(cfg: DictConfig) -> SequenceConfig:
    demos = cfg.sequence.demos_per_task
    if isinstance(demos, int):
        if demos < 1:
            raise ValueError(f"demos_per_task must be >=1, got {demos}")
        demo_spec: int | tuple[int, int] | DemoCountConfig = demos
    elif hasattr(demos, "get") and demos.get("scope") is not None:
        if demos.scope not in {"per_sequence", "per_task"}:
            raise ValueError(
                "demos_per_task.scope must be per_sequence or per_task, got "
                f"{demos.scope!r}"
            )
        if int(demos.min) < 1 or int(demos.min) > int(demos.max):
            raise ValueError(
                "demos_per_task requires 1 <= min <= max, got "
                f"[{demos.min}, {demos.max}]"
            )
        demo_spec = DemoCountConfig(
            min=int(demos.min), max=int(demos.max), scope=str(demos.scope)
        )
    else:
        demo_values = tuple(int(value) for value in demos)
        if len(demo_values) != 2:
            raise ValueError(
                f"demos_per_task range must contain [min, max], got {demo_values}"
            )
        if demo_values[0] < 1 or demo_values[0] > demo_values[1]:
            raise ValueError(
                f"demos_per_task requires 1 <= min <= max, got {demo_values}"
            )
        demo_spec = (demo_values[0], demo_values[1])

    surplus_raw = cfg.sequence.get("surplus_tasks")
    surplus: int | tuple[int, int] | None
    if surplus_raw is None:
        surplus = None
    elif isinstance(surplus_raw, int):
        if surplus_raw < 0:
            raise ValueError(f"surplus_tasks must be >=0, got {surplus_raw}")
        surplus = surplus_raw
    else:
        surplus_values = tuple(int(value) for value in surplus_raw)
        if len(surplus_values) != 2:
            raise ValueError(
                f"surplus_tasks range must contain [min, max], got {surplus_values}"
            )
        if surplus_values[0] < 0 or surplus_values[0] > surplus_values[1]:
            raise ValueError(
                f"surplus_tasks requires 0 <= min <= max, got {surplus_values}"
            )
        surplus = (surplus_values[0], surplus_values[1])

    sequence = SequenceConfig(
        phases=tuple(
            PhaseConfig(num_tasks=p.num_tasks, hotness=tuple(p.hotness))
            for p in cfg.sequence.get("phases", [])
        ),
        demos_per_task=demo_spec,
        signal_boundaries=cfg.sequence.signal_boundaries,
        require_identifiable=cfg.sequence.require_identifiable,
        require_full_rank=cfg.sequence.get("require_full_rank", False),
        task_graph=cfg.sequence.get("task_graph", "random"),
        graph_ordered=cfg.sequence.get("graph_ordered", False),
        curriculum_sampler=cfg.sequence.get("curriculum_sampler", "rejection"),
        hotness=int(cfg.sequence.get("hotness", 2)),
        surplus_tasks=surplus,
    )
    return sequence


def make_family(
    cfg: DictConfig, extra_hotness: int = 0, *, num_modules: int | None = None
) -> HyperTeacher:
    """Build the task family, enumerating patterns up to the largest hotness any
    phase can request (plus headroom for higher-hotness eval tasks)."""
    phases = cfg.sequence.get("phases", [])
    curriculum_hotness = (
        max(int(p.hotness[1]) for p in phases)
        if phases
        else int(cfg.sequence.get("hotness", 2))
    )
    max_hotness = max(curriculum_hotness, extra_hotness)
    return HyperTeacher(
        teacher_config_from(cfg, num_modules=num_modules), max_hotness=max_hotness
    )


class SequenceDataset(IterableDataset):
    """On-the-fly stream of ICCL sequences.

    Infinite by default; ``num_sequences`` makes it finite (eval use). With
    multiple dataloader workers, worker w yields indices w, w+W, w+2W, … —
    contents per index are identical regardless of worker count.
    ``start_index`` offsets the stream (checkpoint resume: pass the number of
    samples already consumed to continue at the exact offset).
    """

    def __init__(
        self,
        family: HyperTeacher,
        seq_cfg: SequenceConfig,
        base_seed: int,
        num_sequences: int | None = None,
        final_task: FinalTaskConfig | None = None,
        start_index: int = 0,
        module_counts: ModuleCountConfig | None = None,
        families: dict[int, HyperTeacher] | None = None,
    ) -> None:
        self.family = family
        self.seq_cfg = seq_cfg
        self.base_seed = base_seed
        self.num_sequences = num_sequences
        self.final_task = final_task
        self.start_index = start_index
        self.module_counts = module_counts or ModuleCountConfig(
            family.cfg.num_modules,
            family.cfg.num_modules,
            (),
            (family.cfg.num_modules,),
        )
        self.families = families or {family.cfg.num_modules: family}
        missing = set(self.module_counts.allowed) - self.families.keys()
        if missing:
            raise ValueError(
                f"no HyperTeacher family configured for module counts {sorted(missing)}"
            )

    def build(self, index: int, **kwargs: Any) -> SequenceSample:
        rng = sequence_rng(self.base_seed, index)
        num_modules = self.module_counts.sample(rng)
        return build_sequence(
            self.families[num_modules], self.seq_cfg, rng, final_task=self.final_task, **kwargs
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        start = worker.id if worker is not None else 0
        step = worker.num_workers if worker is not None else 1
        index = self.start_index + start
        end = None if self.num_sequences is None else self.start_index + self.num_sequences
        while end is None or index < end:
            yield to_tensors(self.build(index))
            index += step


def sequence_dataset_from_config(
    cfg: DictConfig,
    *,
    base_seed: int,
    num_sequences: int | None = None,
    final_task: FinalTaskConfig | None = None,
    start_index: int = 0,
    extra_hotness: int = 0,
) -> SequenceDataset:
    """Build a dataset and one cached immutable task family per allowed ``M``."""
    module_counts = module_count_config_from(cfg)
    sequence = sequence_config_from(cfg)
    for num_modules in module_counts.allowed:
        assert_feasible(sequence, num_modules)
    families = {
        num_modules: make_family(cfg, extra_hotness, num_modules=num_modules)
        for num_modules in module_counts.allowed
    }
    return SequenceDataset(
        families[module_counts.allowed[0]],
        sequence,
        base_seed=base_seed,
        num_sequences=num_sequences,
        final_task=final_task,
        start_index=start_index,
        module_counts=module_counts,
        families=families,
    )


def to_tensors(sample: SequenceSample) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.from_numpy(sample.tokens),
        "token_type": torch.from_numpy(sample.token_type),
        "targets": torch.from_numpy(sample.targets),
        "loss_mask": torch.from_numpy(sample.loss_mask),
    }


def collate_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack sequences, right-padding to the batch's longest sequence (pad rows
    are all-zero with token type TOKEN_PAD and zero loss mask)."""
    max_len = max(item["tokens"].shape[0] for item in batch)
    out: dict[str, torch.Tensor] = {}
    for key in ("tokens", "token_type", "targets", "loss_mask"):
        padded = []
        for item in batch:
            tensor = item[key]
            pad_len = max_len - tensor.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len, *tensor.shape[1:])
                fill = TOKEN_PAD if key == "token_type" else 0
                tensor = torch.cat([tensor, torch.full(pad_shape, fill, dtype=tensor.dtype)])
            padded.append(tensor)
        out[key] = torch.stack(padded)
    return out
