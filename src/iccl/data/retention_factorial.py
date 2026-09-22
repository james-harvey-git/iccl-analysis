"""Independent preceding-history and delay interventions on frozen task blocks."""

from dataclasses import replace

import numpy as np

from iccl.data.curriculum import CURRICULUM_SAMPLER_CODES, SequenceConfig, weighted_edge
from iccl.data.dataset import sequence_rng
from iccl.data.retention_position import _conditions, _inputs, _world
from iccl.data.sequences import SequenceSample, build_sequence
from iccl.data.teacher import HyperTeacher, sample_module_pool

FACTORIAL_PROTOCOL = "history-intervention-factorial-v1"
FACTORIAL_SEED_OFFSET = 5_000_000


def factorial_axis(values: object) -> tuple[int, ...]:
    """Canonicalize explicit, distinct nonnegative integer task counts."""
    from omegaconf import ListConfig

    if not isinstance(values, (list, tuple, ListConfig)) or not values:
        raise ValueError("factorial axes require nonempty lists of nonnegative integers")
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("factorial axes require nonnegative integers")
    if len(set(values)) != len(values):
        raise ValueError("factorial axes must not contain duplicate coordinates")
    return tuple(sorted(values))


def build_factorial_cell(
    family: HyperTeacher,
    cfg: SequenceConfig,
    *,
    seed: int,
    world: int,
    preceding: int,
    delay: int,
    control_modes: tuple[str, ...] = ("unexposed", "shared"),
) -> dict[str, SequenceSample]:
    """Assemble one matched cell using grid-independent indexed random streams.

    Logical blocks 0/1 are original/final; even IDs >=2 are preceding blocks,
    odd IDs >=3 are intervening blocks. Bank prefixes never depend on grid size.
    """
    demos = cfg.demos_per_task
    if (
        family.cfg.num_modules < 4
        or cfg.hotness != 2
        or not isinstance(demos, int)
        or demos < 1
        or min(preceding, delay) < 0
    ):
        raise ValueError("factorial retention requires M>=4, 2-hot tasks, D>=1 and p,d>=0")
    if cfg.require_full_rank:
        raise ValueError("factorial retention is incompatible with require_full_rank")
    rng = sequence_rng(seed + FACTORIAL_SEED_OFFSET, world)
    streams = rng.integers(0, 2**32, size=5).tolist()
    pool = sample_module_pool(family.cfg, sequence_rng(streams[0], 0))
    target, remaining, controls = _world(
        family, sequence_rng(streams[1], 0), control_modes, cfg.max_attempts
    )
    original_x, final_x = _inputs(family, sequence_rng(streams[2], 0), 2, demos)

    def bank(count: int, stream: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        latents, inputs = [], []
        for index in range(count):
            block_rng = sequence_rng(stream, index)
            a, b = block_rng.choice(remaining, 2, replace=False)
            latents.append(weighted_edge(family, (int(a), int(b)), block_rng))
            inputs.append(_inputs(family, block_rng, 1, demos)[0])
        return latents, inputs

    before, before_x = bank(preceding, streams[3])
    after, after_x = bank(delay, streams[4])
    tasks = preceding + delay + 1
    logical = np.array([*(2 + 2 * np.arange(preceding)), 0, *(3 + 2 * np.arange(delay))])
    repeat = build_sequence(
        family,
        replace(cfg, require_identifiable=False, phases=(), surplus_tasks=None),
        rng,
        revisit_demos=demos,
        revisit_task_index=preceding,
        include_world=True,
        world=pool,
        fixed_curriculum_latents=np.stack([*before, target, *after]),
        fixed_demo_counts=(demos,) * tasks,
        fixed_task_inputs=tuple([*before_x, original_x, *after_x, final_x]),
    )
    conditions = _conditions(
        family,
        repeat,
        controls,
        group_id=world,
        pair_id=world,
        mode="natural",
        core_indices=np.delete(np.arange(tasks), preceding),
        logical_ids=logical,
        slots=np.empty(0, dtype=np.int64),
        designated=-1,
        attempts=1,
    )
    for sample in conditions.values():
        sample.info.update(
            protocol=FACTORIAL_PROTOCOL,
            curriculum_sampler=CURRICULUM_SAMPLER_CODES["independent"],
            num_surplus_tasks=-1,  # No constructive surplus; suite metadata uses null.
        )
        sample.info["logical_task_id"][-1] = 1
    return conditions
