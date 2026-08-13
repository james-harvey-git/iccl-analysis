"""Latent-task priors, validity constraints, and causal canonicalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement

import numpy as np

from iccl.data.sequences import check_compositional, check_connected


@dataclass(frozen=True)
class ScheduleConfig:
    """The supported discrete ICCL schedule family."""

    num_modules: int
    num_tasks: int
    hotness: int
    weight_values: tuple[float, ...]
    require_coverage: bool = True
    require_connected: bool = True


def enumerate_task_latents(cfg: ScheduleConfig) -> np.ndarray:
    """Enumerate every labelled-module task under the discrete prior."""
    candidates: list[np.ndarray] = []
    for modules in combinations(range(cfg.num_modules), cfg.hotness):
        coefficient_grids = np.meshgrid(
            *([np.asarray(cfg.weight_values, dtype=np.float64)] * cfg.hotness),
            indexing="ij",
        )
        for coefficients in np.stack(coefficient_grids, axis=-1).reshape(-1, cfg.hotness):
            latent = np.zeros(cfg.num_modules, dtype=np.float64)
            latent[np.asarray(modules)] = coefficients
            candidates.append(latent)
    return np.stack(candidates)


def sample_task_latent(rng: np.random.Generator, cfg: ScheduleConfig) -> np.ndarray:
    """Sample one labelled-module task from the i.i.d. task prior."""
    active = rng.choice(cfg.num_modules, size=cfg.hotness, replace=False)
    latent = np.zeros(cfg.num_modules, dtype=np.float64)
    latent[active] = rng.choice(cfg.weight_values, size=cfg.hotness, replace=True)
    return latent


def is_valid_schedule(schedule: np.ndarray, cfg: ScheduleConfig) -> bool:
    """Check shape, support coverage, and module co-occurrence connectivity."""
    if schedule.shape != (cfg.num_tasks, cfg.num_modules):
        return False
    supports = schedule != 0.0
    if not bool((supports.sum(axis=1) == cfg.hotness).all()):
        return False
    if cfg.require_coverage and not check_compositional(supports, cfg.num_modules):
        return False
    if cfg.require_connected and not check_connected(supports):
        return False
    return True


def sample_valid_schedule(
    rng: np.random.Generator,
    cfg: ScheduleConfig,
    *,
    max_attempts: int,
) -> tuple[np.ndarray, int]:
    """Sample the full i.i.d. schedule prior conditioned on global validity."""
    for attempt in range(1, max_attempts + 1):
        schedule = np.stack([sample_task_latent(rng, cfg) for _ in range(cfg.num_tasks)])
        if is_valid_schedule(schedule, cfg):
            return schedule, attempt
    raise RuntimeError(
        f"failed to sample a valid schedule in {max_attempts} attempts; "
        "increase max_completion_attempts or check schedule feasibility"
    )


def sample_conditional_completion(
    rng: np.random.Generator,
    prefix: np.ndarray,
    cfg: ScheduleConfig,
    *,
    max_attempts: int,
) -> tuple[np.ndarray, int]:
    """Sample an i.i.d. future tail conditioned on a fixed prefix and validity."""
    if prefix.ndim != 2 or prefix.shape[1] != cfg.num_modules:
        raise ValueError("prefix must have shape [observed_tasks, num_modules]")
    if prefix.shape[0] > cfg.num_tasks:
        raise ValueError("prefix is longer than the schedule")
    remaining = cfg.num_tasks - prefix.shape[0]
    if remaining == 0:
        if is_valid_schedule(prefix, cfg):
            return prefix.copy(), 1
        raise RuntimeError("fixed complete schedule is invalid")
    for attempt in range(1, max_attempts + 1):
        tail = np.stack([sample_task_latent(rng, cfg) for _ in range(remaining)])
        candidate = np.concatenate([prefix, tail], axis=0)
        if is_valid_schedule(candidate, cfg):
            return candidate, attempt
    raise RuntimeError(
        f"failed to complete a schedule prefix in {max_attempts} attempts"
    )


def canonicalize_schedule_prefix(schedule: np.ndarray, observed_tasks: int) -> np.ndarray:
    """Relabel modules using only columns of the causally observed prefix.

    Columns are sorted lexicographically by their observed coefficients. Python's
    stable sort leaves modules with identical observed prefixes tied in their
    previous order. No future task row participates in the permutation.
    """
    if schedule.ndim != 2:
        raise ValueError("schedule must have shape [tasks, modules]")
    if not 0 <= observed_tasks <= schedule.shape[0]:
        raise ValueError("observed_tasks is outside the schedule")
    prefix = schedule[:observed_tasks]
    order = sorted(
        range(schedule.shape[1]),
        key=lambda module: tuple(float(value) for value in prefix[:, module]),
    )
    return schedule[:, np.asarray(order)].copy()


def canonical_task_classes(cfg: ScheduleConfig) -> tuple[np.ndarray, np.ndarray]:
    """Collapse labelled tasks into coefficient multisets and their prior masses.

    This observer currently supports the specified two-hot family. Equal
    coefficient pairs each have probability ``1 / V^2``; unequal unordered
    pairs each have probability ``2 / V^2`` because either labelled ordering
    maps to the same exchangeable class.
    """
    if cfg.hotness != 2:
        raise ValueError("canonical current-task classes currently require hotness=2")
    if cfg.num_modules < 2:
        raise ValueError("two-hot tasks require at least two modules")
    values = tuple(float(value) for value in cfg.weight_values)
    latents: list[np.ndarray] = []
    probabilities: list[float] = []
    for first, second in combinations_with_replacement(values, 2):
        latent = np.zeros(cfg.num_modules, dtype=np.float64)
        latent[:2] = (first, second)
        latents.append(latent)
        multiplicity = 1 if math.isclose(first, second) else 2
        probabilities.append(multiplicity / len(values) ** 2)
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if not np.isclose(probabilities_array.sum(), 1.0):
        raise RuntimeError("canonical task-class prior does not sum to one")
    return np.stack(latents), np.log(probabilities_array)


def prefix_key(schedule: np.ndarray, observed_tasks: int) -> bytes:
    """Return a stable byte key for a canonically relabelled observed prefix."""
    canonical = canonicalize_schedule_prefix(schedule, observed_tasks)
    return canonical[:observed_tasks].astype(np.float32, copy=False).tobytes()


def systematic_resampling_indices(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw low-variance SMC ancestor indices from normalized weights."""
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("weights must be a non-empty vector")
    if bool((weights < 0.0).any()) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must be non-negative and normalized")
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right").astype(np.int64)
