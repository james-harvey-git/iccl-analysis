"""Torch dataset wrapper with deterministic, order-independent seeding.

Sequence ``i`` under base seed ``s`` is generated on CPU from a
``numpy.random.Philox`` generator keyed on ``(s, i)``, so the stream is a pure
function of (config, seed, index) — independent of dataloader workers, batch
size, device, and access order. The golden-stream tests pin the stream against
accidental sampler changes.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import IterableDataset, get_worker_info

from iccl.data.sequences import (
    TOKEN_PAD,
    FinalTaskConfig,
    PhaseConfig,
    SequenceConfig,
    SequenceSample,
    build_sequence,
)
from iccl.data.teacher import HyperTeacher, TeacherConfig


def sequence_rng(base_seed: int, index: int) -> np.random.Generator:
    """The canonical per-sequence RNG: Philox keyed on (base_seed, index)."""
    return np.random.Generator(np.random.Philox(key=np.array([base_seed, index], dtype=np.uint64)))


def teacher_config_from(cfg: DictConfig) -> TeacherConfig:
    return TeacherConfig(
        input_dim=cfg.input_dim,
        output_dim=cfg.output_dim,
        hidden_dims=tuple(cfg.hidden_dims),
        use_bias=cfg.use_bias,
        num_modules=cfg.num_modules,
        scale=cfg.scale,
        weighting=cfg.weighting,
    )


def sequence_config_from(cfg: DictConfig) -> SequenceConfig:
    demos = cfg.sequence.demos_per_task
    return SequenceConfig(
        phases=tuple(
            PhaseConfig(num_tasks=p.num_tasks, hotness=tuple(p.hotness))
            for p in cfg.sequence.phases
        ),
        demos_per_task=demos if isinstance(demos, int) else tuple(demos),
        signal_boundaries=cfg.sequence.signal_boundaries,
        require_identifiable=cfg.sequence.require_identifiable,
        require_full_rank=cfg.sequence.get("require_full_rank", False),
        task_graph=cfg.sequence.get("task_graph", "random"),
        graph_ordered=cfg.sequence.get("graph_ordered", False),
    )


def make_family(cfg: DictConfig, extra_hotness: int = 0) -> HyperTeacher:
    """Build the task family, enumerating patterns up to the largest hotness any
    phase can request (plus headroom for higher-hotness eval tasks)."""
    max_hotness = max(max(p.hotness[1] for p in cfg.sequence.phases), extra_hotness)
    return HyperTeacher(teacher_config_from(cfg), max_hotness=max_hotness)


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
    ) -> None:
        self.family = family
        self.seq_cfg = seq_cfg
        self.base_seed = base_seed
        self.num_sequences = num_sequences
        self.final_task = final_task
        self.start_index = start_index

    def build(self, index: int, **kwargs: Any) -> SequenceSample:
        rng = sequence_rng(self.base_seed, index)
        return build_sequence(self.family, self.seq_cfg, rng, final_task=self.final_task, **kwargs)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        worker = get_worker_info()
        start = worker.id if worker is not None else 0
        step = worker.num_workers if worker is not None else 1
        index = self.start_index + start
        end = None if self.num_sequences is None else self.start_index + self.num_sequences
        while end is None or index < end:
            yield to_tensors(self.build(index))
            index += step


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
