"""Frozen worlds paired across encounter interventions, delays and rehearsal."""

from dataclasses import replace

import numpy as np

from iccl.data.controls import (
    build_paired_retention_control,
    exact_latent_occurrences,
    sample_retention_control_latent,
)
from iccl.data.curriculum import (
    SequenceConfig,
    check_compositional,
    check_connected,
    sample_curriculum,
    weighted_edge,
)
from iccl.data.sequences import SequenceSample, build_sequence
from iccl.data.teacher import HyperTeacher, sample_module_pool

RETENTION_PROTOCOL = "history-intervention-v1"
REHEARSAL_PROTOCOL = "history-intervention-rehearsal-v1"
REHEARSAL_MODES = ("none", "one", "both")


def _dimensions(family: HyperTeacher, cfg: SequenceConfig) -> tuple[int, int]:
    if not isinstance(cfg.surplus_tasks, int) or not isinstance(cfg.demos_per_task, int):
        raise ValueError("retention requires fixed surplus_tasks and demos_per_task")
    tasks, demos = family.cfg.num_modules - 1 + cfg.surplus_tasks, cfg.demos_per_task
    if family.cfg.num_modules < 4 or cfg.hotness != 2 or cfg.surplus_tasks < 0 or demos < 1:
        raise ValueError("retention requires M>=4, T>=M-1, 2-hot tasks and D>=1")
    if cfg.require_full_rank:
        raise ValueError("retention history interventions are incompatible with require_full_rank")
    if cfg.curriculum_sampler not in {"constructive", "rejection"}:
        raise ValueError("retention requires constructive or rejection background sampling")
    return tasks, demos


def _background(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    remaining: np.ndarray,
    tasks: int,
) -> tuple[np.ndarray, int]:
    """Sample a connected, fully covered curriculum on the non-target support."""
    restricted = HyperTeacher(replace(family.cfg, num_modules=len(remaining)), max_hotness=2)
    background_cfg = replace(
        cfg,
        phases=(),
        surplus_tasks=tasks - (len(remaining) - 1),
        require_identifiable=True,
        require_full_rank=False,
    )
    background = sample_curriculum(restricted, background_cfg, rng)
    latents = np.zeros((tasks, family.cfg.num_modules), dtype=np.float32)
    latents[:, remaining] = background.latents
    return latents, background.generation_attempts


def _annotate(
    sample: SequenceSample,
    *,
    condition: str,
    group_id: int,
    pair_id: int,
    mode: str,
    core_indices: np.ndarray,
    logical_ids: np.ndarray,
    slots: np.ndarray,
    designated: int,
    attempts: int,
) -> None:
    history, target = sample.info["latents"][:-1], sample.info["latents"][-1]
    position = int(sample.info["original_task_position"])
    target_support = np.flatnonzero(target)
    remaining = np.flatnonzero(target == 0)
    active = history != 0
    core = active[core_indices][:, remaining]
    full_coverage = check_compositional(active, len(target))
    connected = check_connected(active)
    sample.info.update(
        condition=condition,
        protocol=RETENTION_PROTOCOL if mode == "natural" else REHEARSAL_PROTOCOL,
        exposure_scope="history" if mode == "natural" else "original_encounter",
        position_group_id=group_id,
        pair_id=pair_id,
        world_index=group_id,
        sequence_index=pair_id,
        target_support=target_support,
        target_modules_seen_before=active[:position, target_support].any(axis=0),
        target_module_pre_exposures=active[:position, target_support].sum(axis=0),
        target_module_post_exposures=active[position + 1 :, target_support].sum(axis=0),
        constituent_task_exposures=active[:, target_support].sum(axis=0),
        constituent_demo_exposures=(
            active[:, target_support] * sample.info["demo_counts"][:-1, None]
        ).sum(axis=0),
        prior_target_latent_count=exact_latent_occurrences(history, target),
        prior_target_support_count=int(np.all(active == (target != 0), axis=1).sum()),
        rehearsal_mode=mode,
        rehearsal_positions=slots,
        designated_constituent=designated,
        logical_task_id=np.append(logical_ids, -1),
        background_task_indices=core_indices,
        background_num_modules=len(remaining),
        background_connected=check_connected(core),
        background_covered=check_compositional(core, len(remaining)),
        history_connected=connected,
        history_covered=full_coverage,
        support_status=("connected" if connected else "disconnected")
        + ("_covered" if full_coverage else "_partial"),
        generation_attempts=attempts,
    )


def _conditions(
    family: HyperTeacher,
    repeat: SequenceSample,
    controls: dict[str, np.ndarray],
    **metadata: object,
) -> dict[str, SequenceSample]:
    conditions = {"repeat": repeat} | {
        mode: build_paired_retention_control(family, repeat, mode=mode, latent=latent)
        for mode, latent in controls.items()
    }
    for condition, sample in conditions.items():
        _annotate(sample, condition=condition, **metadata)  # pyright: ignore[reportArgumentType]
    return conditions


def _world(
    family: HyperTeacher,
    rng: np.random.Generator,
    modes: tuple[str, ...],
    max_attempts: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if set(modes) - {"unexposed", "shared"} or "unexposed" not in modes:
        raise ValueError("retention controls require unexposed and optionally shared")
    support = np.sort(rng.choice(family.cfg.num_modules, size=2, replace=False))
    target = weighted_edge(family, (int(support[0]), int(support[1])), rng)
    controls = {
        mode: sample_retention_control_latent(family, target, rng, mode, max_attempts)
        for mode in modes
    }
    return target, np.flatnonzero(target == 0), controls


def _inputs(
    family: HyperTeacher, rng: np.random.Generator, tasks: int, demos: int
) -> tuple[np.ndarray, ...]:
    return tuple(
        rng.uniform(-1.0, 1.0, size=(demos, family.cfg.input_dim)).astype(np.float32)
        for _ in range(tasks)
    )


def build_paired_position_group(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    *,
    group_id: int,
    control_modes: tuple[str, ...] = ("unexposed", "shared"),
) -> dict[str, list[SequenceSample]]:
    """Move a frozen original encounter through every position of one world."""
    tasks, demos = _dimensions(family, cfg)
    pool = sample_module_pool(family.cfg, rng)
    target, remaining, controls = _world(family, rng, control_modes, cfg.max_attempts)
    background, attempts = _background(family, cfg, rng, remaining, tasks - 1)
    history = np.concatenate([background, target[None]])
    inputs = _inputs(family, rng, tasks + 1, demos)
    samples = {condition: [] for condition in ("repeat", *control_modes)}
    for position in range(tasks):
        order = np.insert(np.arange(tasks - 1), position, tasks - 1)
        repeat = build_sequence(
            family,
            cfg,
            rng,
            revisit_demos=demos,
            revisit_task_index=position,
            include_world=True,
            world=pool,
            fixed_curriculum_latents=history[order],
            fixed_demo_counts=(demos,) * tasks,
            fixed_task_inputs=tuple(inputs[index] for index in order) + (inputs[-1],),
        )
        for condition, sample in _conditions(
            family,
            repeat,
            controls,
            group_id=group_id,
            pair_id=group_id * tasks + position,
            mode="natural",
            core_indices=np.delete(np.arange(tasks), position),
            logical_ids=order,
            slots=np.empty(0, dtype=np.int64),
            designated=-1,
            attempts=attempts,
        ).items():
            samples[condition].append(sample)
    return samples


def build_rehearsal_position_group(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    *,
    group_id: int,
    control_modes: tuple[str, ...] = ("unexposed", "shared"),
) -> dict[str, list[SequenceSample]]:
    """Pair first/middle encounters and none/one/both subsequent rehearsals."""
    tasks, demos = _dimensions(family, cfg)
    middle = (tasks - 1) // 2
    if tasks < family.cfg.num_modules or tasks - 1 - middle < 2:
        raise ValueError("rehearsal requires T>=M and at least two post-middle slots")
    pool = sample_module_pool(family.cfg, rng)
    target, remaining, controls = _world(family, rng, control_modes, cfg.max_attempts)
    support = np.flatnonzero(target)
    designated = int(support[group_id % 2])
    final_inputs = _inputs(family, rng, 1, demos)[0]
    samples = {condition: [] for condition in ("repeat", *control_modes)}
    for position_index, position in enumerate((0, middle)):
        core, attempts = _background(family, cfg, rng, remaining, tasks - 3)
        slots = rng.choice(np.arange(position + 1, tasks), size=2, replace=False)
        core_indices = np.array([i for i in range(tasks) if i != position and i not in slots])
        history = np.zeros((tasks, family.cfg.num_modules), dtype=np.float32)
        history[core_indices], history[position] = core, target
        for slot in slots:
            pair = rng.choice(remaining, size=2, replace=False)
            history[slot] = weighted_edge(family, (int(pair[0]), int(pair[1])), rng)
        rehearsals = [
            weighted_edge(family, (int(module), int(rng.choice(remaining))), rng)
            for module in (designated, int(support[(group_id + 1) % 2]))
        ]
        inputs = _inputs(family, rng, tasks, demos) + (final_inputs,)
        for mode_index, mode in enumerate(REHEARSAL_MODES):
            varied = history.copy()
            for index in range(mode_index):
                varied[slots[index]] = rehearsals[index]
            repeat = build_sequence(
                family,
                cfg,
                rng,
                revisit_demos=demos,
                revisit_task_index=position,
                include_world=True,
                world=pool,
                fixed_curriculum_latents=varied,
                fixed_demo_counts=(demos,) * tasks,
                fixed_task_inputs=inputs,
            )
            for condition, sample in _conditions(
                family,
                repeat,
                controls,
                group_id=group_id,
                pair_id=group_id * 6 + position_index * 3 + mode_index,
                mode=mode,
                core_indices=core_indices,
                logical_ids=np.arange(tasks),
                slots=slots,
                designated=designated,
                attempts=attempts,
            ).items():
                samples[condition].append(sample)
    return samples
