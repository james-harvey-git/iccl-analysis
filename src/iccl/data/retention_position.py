"""Paired behavioral diagnostics for serial position and module rehearsal.

The all-position family permutes complete, frozen task blocks within one world.
The rehearsal family holds a target and world fixed while controlling whether
neither, one, or both target modules occur between the target and its revisit.
"""

from math import comb

import numpy as np

from iccl.data.controls import (
    build_paired_retention_control,
    exact_latent_occurrences,
    sample_retention_control_latent,
)
from iccl.data.curriculum import SequenceConfig, check_compositional, check_connected, weighted_edge
from iccl.data.sequences import SequenceSample, build_sequence
from iccl.data.teacher import HyperTeacher, sample_module_pool

REHEARSAL_MODES = ("none", "one", "both")


def _support(latent: np.ndarray) -> tuple[int, ...]:
    return tuple(int(module) for module in np.flatnonzero(latent))


def _dimensions(family: HyperTeacher, cfg: SequenceConfig) -> tuple[int, int]:
    if not isinstance(cfg.surplus_tasks, int) or not isinstance(cfg.demos_per_task, int):
        raise ValueError("position diagnostics require fixed surplus_tasks and demos_per_task")
    return family.cfg.num_modules - 1 + cfg.surplus_tasks, cfg.demos_per_task


def _task_inputs(sample: SequenceSample, tasks: int, input_dim: int) -> tuple[np.ndarray, ...]:
    return tuple(
        sample.tokens[start + 2 * np.arange(count), :input_dim].copy()
        for (start, _), count in zip(
            sample.info["task_spans"][:tasks],
            sample.info["demo_counts"][:tasks],
            strict=True,
        )
    )


def _control_latents(
    family: HyperTeacher,
    histories: list[np.ndarray],
    target: np.ndarray,
    rng: np.random.Generator,
    modes: tuple[str, ...],
    max_attempts: int,
) -> dict[str, np.ndarray]:
    combined = np.concatenate(histories)
    return {
        mode: sample_retention_control_latent(family, combined, target, rng, mode, max_attempts)
        for mode in modes
    }


def _annotate(
    sample: SequenceSample,
    history: np.ndarray,
    target: np.ndarray,
    position: int,
    *,
    group_id: int,
    pair_id: int,
    mode: str,
    support_status: str,
    designated_constituent: int = -1,
    logical_task_ids: np.ndarray | None = None,
) -> None:
    target_support = np.asarray(_support(target), dtype=np.int64)
    active = history != 0
    before, after = active[:position], active[position + 1 :]
    sample.info.update(
        position_group_id=group_id,
        pair_id=pair_id,
        world_index=group_id,
        sequence_index=pair_id,
        target_support=target_support,
        target_modules_seen_before=before[:, target_support].any(axis=0),
        target_module_pre_exposures=before[:, target_support].sum(axis=0),
        target_module_post_exposures=after[:, target_support].sum(axis=0),
        prior_target_latent_count=exact_latent_occurrences(history, target),
        prior_target_support_count=sum(
            _support(latent) == tuple(target_support) for latent in history
        ),
        rehearsal_mode=mode,
        support_status=support_status,
        designated_constituent=designated_constituent,
    )
    if logical_task_ids is not None:
        sample.info["logical_task_id"] = np.append(logical_task_ids.astype(np.int64), -1)


def _conditions(
    family: HyperTeacher,
    repeat: SequenceSample,
    rng: np.random.Generator,
    control_latents: dict[str, np.ndarray],
) -> dict[str, SequenceSample]:
    return {"repeat": repeat} | {
        mode: build_paired_retention_control(family, repeat, rng, mode=mode, fixed_latent=latent)
        for mode, latent in control_latents.items()
    }


def build_paired_position_group(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    *,
    group_id: int,
    control_modes: tuple[str, ...] = ("novel", "shared"),
) -> dict[str, list[SequenceSample]]:
    """Move one frozen target block through all positions in one sampled history."""
    tasks, demos = _dimensions(family, cfg)
    base = build_sequence(
        family,
        cfg,
        rng,
        include_world=True,
        fixed_demo_counts=(demos,) * tasks,
    )
    history = base.info["latents"][:tasks]
    supports = [_support(latent) for latent in history]
    counts = {support: supports.count(support) for support in set(supports)}
    eligible = [index for index, support in enumerate(supports) if counts[support] == 1]
    if not eligible or len(counts) == comb(family.cfg.num_modules, 2):
        raise RuntimeError("paired position history has no unique target or reserved novel support")
    target_index = eligible[int(rng.integers(len(eligible)))]
    target = history[target_index]
    controls = _control_latents(family, [history], target, rng, control_modes, cfg.max_attempts)
    history_inputs = _task_inputs(base, tasks, family.cfg.input_dim)
    final_inputs = rng.uniform(-1.0, 1.0, size=(demos, family.cfg.input_dim)).astype(np.float32)
    pool = base.info["world"]
    samples = {condition: [] for condition in ("repeat", *control_modes)}
    others = [index for index in range(tasks) if index != target_index]

    for position in range(tasks):
        order = others.copy()
        order.insert(position, target_index)
        order_array = np.asarray(order, dtype=np.int64)
        permuted = history[order_array]
        repeat = build_sequence(
            family,
            cfg,
            rng,
            revisit_demos=demos,
            revisit_task_index=position,
            include_world=True,
            world=pool,
            fixed_curriculum_latents=permuted,
            fixed_demo_counts=(demos,) * tasks,
            fixed_task_inputs=tuple(history_inputs[index] for index in order) + (final_inputs,),
        )
        pair_id = group_id * tasks + position
        _annotate(
            repeat,
            permuted,
            target,
            position,
            group_id=group_id,
            pair_id=pair_id,
            mode="natural",
            support_status="connected_id",
            logical_task_ids=order_array,
        )
        for condition, sample in _conditions(family, repeat, rng, controls).items():
            samples[condition].append(sample)
    return samples


def _random_edge(modules: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    chosen = rng.choice(modules, size=2, replace=False)
    return int(chosen[0]), int(chosen[1])


def _weighted_edges(
    family: HyperTeacher,
    edges: list[tuple[int, int]],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    return [weighted_edge(family, edge, rng) for edge in edges]


def _controlled_histories(
    family: HyperTeacher,
    target: np.ndarray,
    position: int,
    tasks: int,
    rng: np.random.Generator,
    designated: int,
) -> dict[str, np.ndarray]:
    modules = family.cfg.num_modules
    target_modules = _support(target)
    remaining = rng.permutation([m for m in range(modules) if m not in target_modules])
    extras = tasks - (modules - 1)
    if extras < 0:
        raise ValueError(
            f"controlled rehearsal requires canonical T>=M-1, got M={modules}, T={tasks}"
        )

    if position == 0:
        common_edges = [
            (int(remaining[index]), int(remaining[index + 1]))
            for index in range(len(remaining) - 2)
        ] + [_random_edge(remaining, rng) for _ in range(extras)]
        prefix: list[np.ndarray] = []
        bridge = weighted_edge(family, (int(remaining[0]), int(remaining[-1])), rng)
        variable = {
            "none": [
                weighted_edge(family, _random_edge(remaining, rng), rng),
                bridge,
            ],
            "one": [
                weighted_edge(family, (designated, int(remaining[0])), rng),
                bridge,
            ],
            "both": [
                weighted_edge(family, (target_modules[0], int(remaining[0])), rng),
                weighted_edge(family, (target_modules[1], int(remaining[-1])), rng),
            ],
        }
    else:
        if position > len(remaining) or len(remaining[position:]) < 2:
            raise ValueError(
                "controlled middle position needs at least two unreached non-target modules"
            )
        prefix_edges = [(target_modules[0], int(remaining[0]))] + [
            (int(remaining[index]), int(remaining[index + 1])) for index in range(position - 1)
        ]
        tail = remaining[position:]
        common_edges = [
            (int(tail[index]), int(tail[index + 1])) for index in range(len(tail) - 2)
        ] + [_random_edge(remaining, rng) for _ in range(extras)]
        prefix = _weighted_edges(family, prefix_edges, rng)
        bridge = weighted_edge(family, (int(tail[0]), int(tail[-1])), rng)
        variable = {
            "none": [
                weighted_edge(family, (int(remaining[position - 1]), int(tail[0])), rng),
                bridge,
            ],
            "one": [
                weighted_edge(family, (designated, int(tail[0])), rng),
                bridge,
            ],
            "both": [
                weighted_edge(family, (target_modules[0], int(tail[0])), rng),
                weighted_edge(family, (target_modules[1], int(tail[-1])), rng),
            ],
        }

    common = _weighted_edges(family, common_edges, rng)
    suffix_order = rng.permutation(len(common) + 2)
    histories = {}
    for mode in REHEARSAL_MODES:
        suffix = np.stack(common + variable[mode])[suffix_order]
        histories[mode] = np.stack(prefix + [target] + list(suffix))
    return histories


def _validate_controlled(
    history: np.ndarray,
    target: np.ndarray,
    position: int,
    mode: str,
    support_status: str,
) -> None:
    target_support = _support(target)
    post = history[position + 1 :] != 0
    exposures = post[:, target_support].sum(axis=0)
    expected = {"none": [0, 0], "both": [1, 1]}
    if mode in expected and exposures.tolist() != expected[mode]:
        raise AssertionError(f"{mode} rehearsal produced constituent exposures {exposures}")
    if mode == "one" and sorted(exposures.tolist()) != [0, 1]:
        raise AssertionError(f"one rehearsal produced constituent exposures {exposures}")
    if exact_latent_occurrences(history, target) != 1:
        raise AssertionError("target latent must occur once in rehearsal history")
    if sum(_support(latent) == target_support for latent in history) != 1:
        raise AssertionError("target support must occur once in rehearsal history")
    supports = history != 0
    if not check_compositional(supports, history.shape[1]):
        raise AssertionError("rehearsal history does not cover every module")
    connected = check_connected(supports)
    if connected != (support_status == "connected_id"):
        raise AssertionError(f"rehearsal history connectivity disagrees with {support_status}")


def build_rehearsal_position_group(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    *,
    group_id: int,
    control_modes: tuple[str, ...] = ("novel", "shared"),
) -> dict[str, list[SequenceSample]]:
    """Build paired first/middle by none/one/both constituent rehearsals."""
    tasks, demos = _dimensions(family, cfg)
    middle = (tasks - 1) // 2
    if family.cfg.num_modules < 4 or middle < 1 or tasks - 1 - middle < 2:
        raise ValueError("controlled rehearsal needs M>=4 and two post-middle task slots")
    pool = sample_module_pool(family.cfg, rng)
    chosen = sorted(
        int(value) for value in rng.choice(family.cfg.num_modules, size=2, replace=False)
    )
    target_modules = (chosen[0], chosen[1])
    target = weighted_edge(family, target_modules, rng)
    designated = target_modules[group_id % 2]
    histories = {
        position: _controlled_histories(family, target, position, tasks, rng, designated)
        for position in (0, middle)
    }
    all_histories = [history for modes in histories.values() for history in modes.values()]
    controls = _control_latents(family, all_histories, target, rng, control_modes, cfg.max_attempts)
    final_inputs = rng.uniform(-1.0, 1.0, size=(demos, family.cfg.input_dim)).astype(np.float32)
    samples = {condition: [] for condition in ("repeat", *control_modes)}

    for position_index, (position, modes) in enumerate(histories.items()):
        task_inputs = tuple(
            rng.uniform(-1.0, 1.0, size=(demos, family.cfg.input_dim)).astype(np.float32)
            for _ in range(tasks)
        )
        for mode_index, (mode, history) in enumerate(modes.items()):
            status = "disconnected_ood" if position == 0 and mode == "none" else "connected_id"
            _validate_controlled(history, target, position, mode, status)
            repeat = build_sequence(
                family,
                cfg,
                rng,
                revisit_demos=demos,
                revisit_task_index=position,
                include_world=True,
                world=pool,
                fixed_curriculum_latents=history,
                fixed_demo_counts=(demos,) * tasks,
                fixed_task_inputs=task_inputs + (final_inputs,),
            )
            pair_id = group_id * 6 + position_index * 3 + mode_index
            _annotate(
                repeat,
                history,
                target,
                position,
                group_id=group_id,
                pair_id=pair_id,
                mode=mode,
                support_status=status,
                designated_constituent=(designated if mode == "one" else -1),
            )
            for condition, sample in _conditions(family, repeat, rng, controls).items():
                samples[condition].append(sample)
    return samples
