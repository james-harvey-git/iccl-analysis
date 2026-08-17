"""Serialize sampled ICCL curricula into causal demonstration streams."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iccl.data.curriculum import (
    CURRICULUM_SAMPLER_CODES,
    TASK_ORIGIN_CODES,
    CurriculumSample,
    SequenceConfig,
    assert_feasible,
    sample_curriculum,
    task_categories,
)
from iccl.data.teacher import HyperTeacher, ModulePool, sample_module_pool, teacher_forward

TOKEN_X = 0
TOKEN_Y = 1
TOKEN_BOUNDARY = 2
TOKEN_PAD = 3


@dataclass(frozen=True)
class FinalTaskConfig:
    """Task appended after a curriculum for a capability probe."""

    mode: str
    hotness: int
    num_demos: int


@dataclass
class SequenceSample:
    tokens: np.ndarray
    token_type: np.ndarray
    targets: np.ndarray
    loss_mask: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)


def sample_final_latent(
    family: HyperTeacher,
    final: FinalTaskConfig,
    curriculum_latents: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int,
) -> np.ndarray:
    if final.mode == "any":
        return family.apply_weighting(rng, family.sample_pattern(rng, final.hotness))
    if final.mode != "composite":
        raise ValueError(f"unknown final-task mode: {final.mode}")

    covered = np.flatnonzero(curriculum_latents.any(axis=0))
    if len(covered) < final.hotness:
        raise ValueError(
            f"composite final task needs {final.hotness} covered modules, got {len(covered)}"
        )
    demonstrated = {tuple(np.flatnonzero(latent)) for latent in curriculum_latents}
    for _ in range(max_attempts):
        chosen = rng.choice(covered, size=final.hotness, replace=False)
        if tuple(sorted(chosen)) not in demonstrated:
            pattern = np.zeros(family.cfg.num_modules, dtype=np.int8)
            pattern[chosen] = 1
            return family.apply_weighting(rng, pattern)
    raise RuntimeError(f"no undemonstrated composite support found in {max_attempts} attempts")


def _draw_demo_count(lo: int, hi: int, rng: np.random.Generator) -> int:
    if lo < 1 or hi < lo:
        raise ValueError(f"demos_per_task requires 1 <= min <= max, got [{lo}, {hi}]")
    return lo if lo == hi else int(rng.integers(lo, hi + 1))


def _curriculum_demo_counts(
    cfg: SequenceConfig, num_tasks: int, rng: np.random.Generator
) -> list[int]:
    spec = cfg.demos_per_task
    if isinstance(spec, int):
        if spec < 1:
            raise ValueError(f"demos_per_task must be >=1, got {spec}")
        return [spec] * num_tasks
    if isinstance(spec, tuple):
        return [_draw_demo_count(*spec, rng) for _ in range(num_tasks)]
    if spec.scope == "per_sequence":
        return [_draw_demo_count(spec.min, spec.max, rng)] * num_tasks
    if spec.scope == "per_task":
        return [_draw_demo_count(spec.min, spec.max, rng) for _ in range(num_tasks)]
    raise ValueError(f"unknown demos_per_task scope: {spec.scope!r}")


def _fixed_curriculum(latents: np.ndarray, num_modules: int) -> CurriculumSample:
    if latents.ndim != 2 or latents.shape[1] != num_modules:
        raise ValueError(
            f"fixed_curriculum_latents must have shape [tasks, {num_modules}], got {latents.shape}"
        )
    tasks = len(latents)
    categories = task_categories(latents)
    return CurriculumSample(
        latents=latents,
        task_origins=np.full(tasks, TASK_ORIGIN_CODES["ordinary"], dtype=np.int8),
        pre_shuffle_indices=np.arange(tasks, dtype=np.int64),
        generation_categories=categories,
        presentation_categories=categories.copy(),
        generation_attempts=0,
        num_surplus_tasks=max(0, tasks - (num_modules - 1)),
    )


def build_sequence(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    *,
    final_task: FinalTaskConfig | None = None,
    revisit_demos: int = 0,
    include_world: bool = False,
    world: ModulePool | None = None,
    fixed_final_latent: np.ndarray | None = None,
    fixed_curriculum_latents: np.ndarray | None = None,
    fixed_demo_counts: tuple[int, ...] | None = None,
) -> SequenceSample:
    """Build one sequence from a fresh world or supplied paired-control state."""
    assert_feasible(cfg, family.cfg.num_modules)
    pool = world if world is not None else sample_module_pool(family.cfg, rng)

    if fixed_curriculum_latents is not None:
        curriculum = _fixed_curriculum(
            fixed_curriculum_latents.astype(np.float32), family.cfg.num_modules
        )
    else:
        curriculum = sample_curriculum(family, cfg, rng)
    curriculum_latents = curriculum.latents

    latents = [latent for latent in curriculum_latents]
    origins = list(curriculum.task_origins)
    source_order = list(curriculum.pre_shuffle_indices)
    generation_categories = list(curriculum.generation_categories)
    if final_task is not None:
        final_latent = (
            fixed_final_latent.astype(np.float32)
            if fixed_final_latent is not None
            else sample_final_latent(family, final_task, curriculum_latents, rng, cfg.max_attempts)
        )
        latents.append(final_latent)
        origins.append(TASK_ORIGIN_CODES["final"])
        source_order.append(len(source_order))
    elif fixed_final_latent is not None:
        raise ValueError("fixed_final_latent requires final_task")
    if revisit_demos > 0:
        if not len(curriculum_latents):
            raise ValueError("revisit requires curriculum history")
        latents.append(curriculum_latents[0])
        origins.append(TASK_ORIGIN_CODES["revisit"])
        source_order.append(len(source_order))

    stacked_latents = np.stack(latents)
    presentation_categories = task_categories(stacked_latents)
    generation_categories.extend(
        int(value) for value in presentation_categories[len(generation_categories) :]
    )

    if fixed_demo_counts is None:
        demo_counts = _curriculum_demo_counts(cfg, len(curriculum_latents), rng)
    else:
        if len(fixed_demo_counts) != len(curriculum_latents) or min(fixed_demo_counts) < 1:
            raise ValueError(
                "fixed_demo_counts must contain one positive count per curriculum task"
            )
        demo_counts = list(fixed_demo_counts)
    if final_task is not None:
        if final_task.num_demos < 1:
            raise ValueError(f"final task demos must be positive, got {final_task.num_demos}")
        demo_counts.append(final_task.num_demos)
    if revisit_demos > 0:
        demo_counts.append(revisit_demos)

    token_dim = max(family.cfg.input_dim, family.cfg.output_dim)
    sequence_length = 2 * sum(demo_counts) + int(cfg.signal_boundaries) * len(latents)
    tokens = np.zeros((sequence_length, token_dim), dtype=np.float32)
    token_type = np.full(sequence_length, TOKEN_PAD, dtype=np.int64)
    targets = np.zeros((sequence_length, family.cfg.output_dim), dtype=np.float32)
    loss_mask = np.zeros(sequence_length, dtype=np.float32)

    boundaries: list[int] = []
    task_spans: list[tuple[int, int]] = []
    base_mse: list[np.ndarray] = []
    position = 0
    for latent, demos in zip(latents, demo_counts, strict=True):
        if cfg.signal_boundaries:
            boundaries.append(position)
            token_type[position] = TOKEN_BOUNDARY
            position += 1
        start = position
        x = rng.uniform(-1.0, 1.0, size=(demos, family.cfg.input_dim)).astype(np.float32)
        y = teacher_forward(pool, latent, x)
        base_mse.append(((y - y.mean(axis=0)) ** 2).mean(axis=0))
        for demo in range(demos):
            tokens[position, : family.cfg.input_dim] = x[demo]
            token_type[position] = TOKEN_X
            targets[position] = y[demo]
            loss_mask[position] = 1.0
            position += 1
            tokens[position, : family.cfg.output_dim] = y[demo]
            token_type[position] = TOKEN_Y
            position += 1
        task_spans.append((start, position))

    tasks = len(latents)
    history_prediction_tokens = np.zeros(tasks, dtype=np.int64)
    history_serialized_tokens = np.zeros(tasks, dtype=np.int64)
    unique_supports_seen = np.zeros(tasks, dtype=np.int64)
    modules_covered = np.zeros(tasks, dtype=np.int64)
    supports: set[tuple[int, ...]] = set()
    covered: set[int] = set()
    prediction_prefix = serialized_prefix = 0
    for index, (latent, demos) in enumerate(zip(latents, demo_counts, strict=True)):
        history_prediction_tokens[index] = prediction_prefix
        history_serialized_tokens[index] = serialized_prefix
        unique_supports_seen[index] = len(supports)
        modules_covered[index] = len(covered)
        support = tuple(int(module) for module in np.flatnonzero(latent))
        supports.add(support)
        covered.update(support)
        prediction_prefix += demos
        serialized_prefix += 2 * demos + int(cfg.signal_boundaries)

    sampler = (
        cfg.curriculum_sampler
        if cfg.surplus_tasks is not None
        else ("structured" if cfg.task_graph != "random" else "rejection")
    )
    info: dict[str, Any] = {
        "latents": stacked_latents,
        "demo_counts": np.asarray(demo_counts, dtype=np.int64),
        "boundaries": np.asarray(boundaries, dtype=np.int64),
        "task_spans": np.asarray(task_spans, dtype=np.int64),
        "base_mse": np.stack(base_mse),
        "num_curriculum_tasks": len(curriculum_latents),
        "num_modules": family.cfg.num_modules,
        "num_surplus_tasks": curriculum.num_surplus_tasks,
        "num_prediction_tokens": sum(demo_counts),
        "serialized_length": sequence_length,
        "curriculum_sampler": CURRICULUM_SAMPLER_CODES[sampler],
        "generation_attempts": curriculum.generation_attempts,
        "task_origin": np.asarray(origins, dtype=np.int8),
        "pre_shuffle_index": np.asarray(source_order, dtype=np.int64),
        "generation_category": np.asarray(generation_categories, dtype=np.int8),
        "presentation_category": presentation_categories,
        "history_prediction_tokens": history_prediction_tokens,
        "history_serialized_tokens": history_serialized_tokens,
        "target_first_prediction_index": np.asarray(task_spans, dtype=np.int64)[:, 0],
        "num_unique_supports_seen": unique_supports_seen,
        "num_modules_covered": modules_covered,
    }
    if revisit_demos > 0:
        original_last = task_spans[0][0] + 2 * (demo_counts[0] - 1)
        revisit_first = task_spans[-1][0]
        info.update(
            intervening_tasks=len(curriculum_latents) - 1,
            prediction_token_delay=int(history_prediction_tokens[-1] - demo_counts[0]),
            serialized_token_delay=revisit_first - original_last,
        )
    if include_world:
        info["world"] = pool
    return SequenceSample(tokens, token_type, targets, loss_mask, info)
