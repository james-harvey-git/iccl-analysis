"""Paired counterfactual sequences for composition and retention."""

from typing import Any

import numpy as np

from iccl.data.curriculum import (
    TASK_CATEGORY_CODES,
    TASK_ORIGIN_CODES,
    SequenceConfig,
    decode_prufer,
    task_categories,
    weighted_edge,
)
from iccl.data.sequences import (
    FinalTaskConfig,
    SequenceSample,
    build_sequence,
    sample_final_latent,
)
from iccl.data.teacher import HyperTeacher, sample_module_pool, teacher_forward


def exact_latent_occurrences(history: np.ndarray, latent: np.ndarray) -> int:
    """Count exact task-latent occurrences in a curriculum history."""
    return sum(np.array_equal(previous, latent) for previous in history)


def _retention_control_latent(
    family: HyperTeacher,
    history: np.ndarray,
    revisited: np.ndarray,
    rng: np.random.Generator,
    mode: str,
    max_attempts: int,
) -> np.ndarray:
    if mode == "novel":
        return sample_final_latent(
            family,
            FinalTaskConfig("composite", int(np.count_nonzero(revisited)), 1),
            history,
            rng,
            max_attempts,
        )
    if mode != "shared":
        raise ValueError(f"unknown retention-control mode: {mode}")
    if family.cfg.weighting == "binary":
        raise ValueError("same-support/new-weights control is undefined for weighting=binary")
    pattern = (revisited > 0).astype(np.int8)
    for _ in range(max_attempts):
        latent = family.apply_weighting(rng, pattern)
        if exact_latent_occurrences(history, latent) == 0:
            return latent
    raise RuntimeError(f"no history-novel same-support task found in {max_attempts} attempts")


def build_paired_retention_control(
    family: HyperTeacher,
    sequence: SequenceSample,
    rng: np.random.Generator,
    *,
    mode: str,
    max_attempts: int = 1000,
) -> SequenceSample:
    """Replace only the final task and targets in a retention sequence."""
    if "world" not in sequence.info:
        raise ValueError("paired retention control requires the sampled world")
    latents = sequence.info["latents"]
    curriculum = int(sequence.info["num_curriculum_tasks"])
    if curriculum == 0 or len(latents) <= curriculum:
        raise ValueError("paired retention control requires a revisit block")
    start = int(sequence.info["task_spans"][-1, 0])
    demos = int(sequence.info["demo_counts"][-1])
    replacement = _retention_control_latent(
        family, latents[:curriculum], latents[-1], rng, mode, max_attempts
    )
    positions = start + 2 * np.arange(demos)
    x = sequence.tokens[positions, : family.cfg.input_dim]
    y = teacher_forward(sequence.info["world"], replacement, x)

    tokens, targets = sequence.tokens.copy(), sequence.targets.copy()
    tokens[positions + 1, : family.cfg.output_dim] = y
    targets[positions] = y
    info = dict(sequence.info)
    info["latents"] = np.concatenate([latents[:-1], replacement[None].astype(np.float32)])
    info["base_mse"] = sequence.info["base_mse"].copy()
    info["base_mse"][-1] = ((y - y.mean(axis=0)) ** 2).mean(axis=0)
    return SequenceSample(
        tokens, sequence.token_type.copy(), targets, sequence.loss_mask.copy(), info
    )


def _composition_latents(
    family: HyperTeacher,
    rng: np.random.Generator,
    *,
    tasks: int,
    exposures: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], np.ndarray]:
    modules = family.cfg.num_modules
    minimum_tasks = modules - 1 + 2 * (exposures - 1)
    if modules < 4 or exposures < 1 or tasks < minimum_tasks:
        raise ValueError(
            "composition requires M>=4, positive constituent exposure, and "
            f"T>={minimum_tasks}; got M={modules}, exposures={exposures}, T={tasks}"
        )

    target_values = sorted(int(value) for value in rng.choice(modules, size=2, replace=False))
    target = (target_values[0], target_values[1])
    remaining = np.array([module for module in range(modules) if module not in target])
    prufer = (
        rng.integers(0, len(remaining), size=len(remaining) - 2, dtype=np.int64)
        if len(remaining) > 2
        else np.empty(0, dtype=np.int64)
    )
    ordinary = [
        (int(remaining[left]), int(remaining[right]))
        for left, right in decode_prufer(prufer, len(remaining))
    ]
    constituent = [
        (module, int(remaining[int(rng.integers(len(remaining)))]))
        for module in target
        for _ in range(exposures)
    ]
    while len(ordinary) + len(constituent) < tasks:
        pair = rng.choice(remaining, size=2, replace=False)
        ordinary.append((int(pair[0]), int(pair[1])))
    history = np.stack([weighted_edge(family, edge, rng) for edge in constituent + ordinary])
    constituent_mask = np.array([True] * len(constituent) + [False] * len(ordinary))
    order = rng.permutation(tasks)
    return (
        history[order],
        weighted_edge(family, target, rng),
        target,
        constituent_mask[order],
    )


def _refresh_metadata(info: dict[str, Any], latents: np.ndarray) -> None:
    info["presentation_category"] = task_categories(latents)
    info["generation_category"] = info["presentation_category"].copy()
    supports: set[tuple[int, ...]] = set()
    modules: set[int] = set()
    unique_counts = np.zeros(len(latents), dtype=np.int64)
    covered_counts = np.zeros(len(latents), dtype=np.int64)
    for index, latent in enumerate(latents):
        unique_counts[index], covered_counts[index] = len(supports), len(modules)
        support = tuple(int(module) for module in np.flatnonzero(latent))
        supports.add(support)
        modules.update(support)
    info["num_unique_supports_seen"] = unique_counts
    info["num_modules_covered"] = covered_counts


def _replace_history(
    family: HyperTeacher, sequence: SequenceSample, replacements: np.ndarray
) -> SequenceSample:
    curriculum = int(sequence.info["num_curriculum_tasks"])
    expected = sequence.info["latents"][:curriculum].shape
    if "world" not in sequence.info or replacements.shape != expected:
        raise ValueError(f"history replacements must have shape {expected} and a sampled world")
    tokens, targets = sequence.tokens.copy(), sequence.targets.copy()
    info = dict(sequence.info)
    latents, base_mse = sequence.info["latents"].copy(), sequence.info["base_mse"].copy()
    latents[:curriculum] = replacements
    for task, latent in enumerate(replacements):
        start = int(sequence.info["task_spans"][task, 0])
        demos = int(sequence.info["demo_counts"][task])
        positions = start + 2 * np.arange(demos)
        x = sequence.tokens[positions, : family.cfg.input_dim]
        y = teacher_forward(sequence.info["world"], latent, x)
        tokens[positions + 1, : family.cfg.output_dim] = y
        targets[positions] = y
        base_mse[task] = ((y - y.mean(axis=0)) ** 2).mean(axis=0)
    info.update(latents=latents, base_mse=base_mse)
    _refresh_metadata(info, latents)
    return SequenceSample(
        tokens, sequence.token_type.copy(), targets, sequence.loss_mask.copy(), info
    )


def _without_history(sequence: SequenceSample) -> SequenceSample:
    """Return the exact final block with its history prefix removed."""
    final = len(sequence.info["latents"]) - 1
    if "world" not in sequence.info or final < int(sequence.info["num_curriculum_tasks"]):
        raise ValueError("no-history control requires a final task and sampled world")
    boundaries = sequence.info["boundaries"]
    start = int(boundaries[-1]) if len(boundaries) else int(sequence.info["task_spans"][-1, 0])
    offset = int(bool(len(boundaries)))
    demos = int(sequence.info["demo_counts"][-1])
    tokens = sequence.tokens[start:].copy()
    info: dict[str, Any] = {
        "latents": sequence.info["latents"][-1:].copy(),
        "demo_counts": np.array([demos]),
        "boundaries": np.array([0]) if offset else np.empty(0, dtype=np.int64),
        "task_spans": np.array([[offset, len(tokens)]]),
        "base_mse": sequence.info["base_mse"][-1:].copy(),
        "num_curriculum_tasks": 0,
        "num_modules": sequence.info["num_modules"],
        "num_surplus_tasks": 0,
        "num_prediction_tokens": demos,
        "serialized_length": len(tokens),
        "curriculum_sampler": sequence.info["curriculum_sampler"],
        "generation_attempts": 0,
        "task_origin": np.array([TASK_ORIGIN_CODES["final"]], dtype=np.int8),
        "pre_shuffle_index": np.array([0]),
        "generation_category": np.array([TASK_CATEGORY_CODES["novel_support"]], dtype=np.int8),
        "presentation_category": np.array([TASK_CATEGORY_CODES["novel_support"]], dtype=np.int8),
        "history_prediction_tokens": np.array([0]),
        "history_serialized_tokens": np.array([0]),
        "target_first_prediction_index": np.array([offset]),
        "num_unique_supports_seen": np.array([0]),
        "num_modules_covered": np.array([0]),
        "world": sequence.info["world"],
    }
    return SequenceSample(
        tokens,
        sequence.token_type[start:].copy(),
        sequence.targets[start:].copy(),
        sequence.loss_mask[start:].copy(),
        info,
    )


def build_paired_composition_controls(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    *,
    target_demos: int,
    constituent_task_exposures: int,
    fixed_demo_counts: tuple[int, ...],
) -> tuple[SequenceSample, SequenceSample, SequenceSample]:
    """Build constituent-history, matched-prefix, and no-history conditions."""
    history, final_latent, target, mask = _composition_latents(
        family,
        rng,
        tasks=len(fixed_demo_counts),
        exposures=constituent_task_exposures,
    )
    pool = sample_module_pool(family.cfg, rng)
    final = FinalTaskConfig("composite", 2, target_demos)
    constituent = build_sequence(
        family,
        cfg,
        rng,
        final_task=final,
        include_world=True,
        world=pool,
        fixed_final_latent=final_latent,
        fixed_curriculum_latents=history,
        fixed_demo_counts=fixed_demo_counts,
    )

    allowed = np.array([module for module in range(family.cfg.num_modules) if module not in target])
    replacements = history.copy()
    for index, latent in enumerate(history):
        if np.any(latent[list(target)]):
            pair = rng.choice(allowed, size=2, replace=False)
            replacements[index] = weighted_edge(family, (int(pair[0]), int(pair[1])), rng)
    matched = _replace_history(family, constituent, replacements)
    no_history = _without_history(constituent)

    for sample in (constituent, matched, no_history):
        sample.info["target_support"] = np.asarray(target, dtype=np.int64)
    for sample in (constituent, matched):
        prefix = sample.info["latents"][: int(sample.info["num_curriculum_tasks"])] != 0
        demos = sample.info["demo_counts"][: len(prefix)]
        sample.info["constituent_task_exposures"] = prefix[:, target].sum(axis=0)
        sample.info["constituent_demo_exposures"] = (prefix[:, target] * demos[:, None]).sum(axis=0)
        sample.info["prior_target_support_count"] = int(
            sum(tuple(np.flatnonzero(latent)) == target for latent in prefix)
        )
    no_history.info["constituent_task_exposures"] = np.zeros(2, dtype=np.int64)
    no_history.info["constituent_demo_exposures"] = np.zeros(2, dtype=np.int64)
    no_history.info["prior_target_support_count"] = 0
    return constituent, matched, no_history
