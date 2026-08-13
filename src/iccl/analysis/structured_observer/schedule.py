"""Latent-task priors, validity constraints, and causal canonicalization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from itertools import combinations, combinations_with_replacement
from typing import TypedDict

import numpy as np

from iccl.data.sequences import check_compositional, check_connected


class StructuredSuiteSpec(TypedDict):
    """Resolved metadata for the supported structured-observer task family."""

    input_dim: int
    output_dim: int
    hidden_dim: int
    use_bias: bool
    num_modules: int
    num_tasks: int
    hotness: int
    scale: float
    weight_values: list[float]
    require_coverage: bool
    require_connected: bool


@dataclass(frozen=True)
class ScheduleConfig:
    """The supported discrete ICCL schedule family."""

    num_modules: int
    num_tasks: int
    hotness: int
    weight_values: tuple[float, ...]
    require_coverage: bool = True
    require_connected: bool = True


GraphState = tuple[int, ...]


class ConditionedSchedulePrior:
    """Exact task-prefix sampler conditioned on eventual schedule validity.

    The supported observer family has two active modules per task, so each task
    adds one edge to the module co-occurrence graph. Dynamic programming counts
    the valid edge-sequence suffixes reachable from every graph state. Sampling
    an edge in proportion to its suffix count therefore draws directly from the
    i.i.d. task prior conditioned on the final validity event, without proposing
    or retaining unobserved future tasks.
    """

    def __init__(self, cfg: ScheduleConfig) -> None:
        if cfg.hotness != 2:
            raise ValueError("conditioned prefix sampling currently requires hotness=2")
        if cfg.num_modules < 2:
            raise ValueError("two-hot tasks require at least two modules")
        if cfg.num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if not cfg.weight_values:
            raise ValueError("weight_values must be non-empty")
        if any(value == 0.0 for value in cfg.weight_values):
            raise ValueError("weight_values must be non-zero")
        self.cfg = cfg
        self.supports = tuple(combinations(range(cfg.num_modules), 2))
        self._completion_counts: dict[tuple[GraphState, int], int] = {}
        if self.completion_count(self.initial_state, cfg.num_tasks) == 0:
            raise ValueError("schedule constraints have no valid task sequence")

    @property
    def initial_state(self) -> GraphState:
        """Graph state before any task support has been observed."""
        return (-1,) * self.cfg.num_modules

    def add_support(self, state: GraphState, support: tuple[int, int]) -> GraphState:
        """Return the canonical connected-component state after one support."""
        if len(state) != self.cfg.num_modules:
            raise ValueError("graph state does not match num_modules")
        first, second = support
        if not 0 <= first < second < self.cfg.num_modules:
            raise ValueError("support must be an ordered pair of module indices")
        merged_labels = {state[first], state[second]} - {-1}
        merged_nodes = {first, second}
        for module, label in enumerate(state):
            if label in merged_labels:
                merged_nodes.add(module)
        merged_label = min(merged_nodes)
        updated = list(state)
        for module in merged_nodes:
            updated[module] = merged_label
        return tuple(updated)

    def state_from_prefix(self, prefix: np.ndarray) -> GraphState:
        """Build the graph state induced by a causally observed latent prefix."""
        if prefix.ndim != 2 or prefix.shape[1] != self.cfg.num_modules:
            raise ValueError("prefix must have shape [observed_tasks, num_modules]")
        if prefix.shape[0] > self.cfg.num_tasks:
            raise ValueError("prefix is longer than the schedule")
        state = self.initial_state
        for latent in prefix:
            active = np.flatnonzero(latent)
            if len(active) != 2:
                raise ValueError("every prefix task must activate exactly two modules")
            state = self.add_support(state, (int(active[0]), int(active[1])))
        return state

    def _is_valid_final_state(self, state: GraphState) -> bool:
        used_labels = {label for label in state if label >= 0}
        if self.cfg.require_coverage and any(label < 0 for label in state):
            return False
        if self.cfg.require_connected and len(used_labels) != 1:
            return False
        return True

    def completion_count(self, state: GraphState, remaining_tasks: int) -> int:
        """Count labelled support suffixes that end in a valid graph state."""
        if remaining_tasks < 0:
            raise ValueError("remaining_tasks cannot be negative")
        key = (state, remaining_tasks)
        if key in self._completion_counts:
            return self._completion_counts[key]
        if remaining_tasks == 0:
            count = int(self._is_valid_final_state(state))
        else:
            count = sum(
                self.completion_count(
                    self.add_support(state, support),
                    remaining_tasks - 1,
                )
                for support in self.supports
            )
        self._completion_counts[key] = count
        return count

    def support_counts(
        self,
        state: GraphState,
        remaining_after_task: int,
    ) -> np.ndarray:
        """Return valid-suffix counts for each possible next support."""
        counts = np.asarray(
            [
                self.completion_count(
                    self.add_support(state, support),
                    remaining_after_task,
                )
                for support in self.supports
            ],
            dtype=np.float64,
        )
        if counts.sum() <= 0.0:
            raise RuntimeError("observed prefix has no valid schedule completion")
        return counts

    def sample_next_batch(
        self,
        rng: np.random.Generator,
        prefixes: np.ndarray,
    ) -> np.ndarray:
        """Sample one validity-conditioned next task for each observed prefix."""
        if prefixes.ndim != 3 or prefixes.shape[2] != self.cfg.num_modules:
            raise ValueError(
                "prefixes must have shape [hypotheses, observed_tasks, num_modules]"
            )
        observed_tasks = prefixes.shape[1]
        if observed_tasks >= self.cfg.num_tasks:
            raise ValueError("cannot sample beyond the configured task schedule")
        states = [self.state_from_prefix(prefix) for prefix in prefixes]
        groups: dict[GraphState, list[int]] = {}
        for index, state in enumerate(states):
            groups.setdefault(state, []).append(index)
        latents = np.zeros(
            (prefixes.shape[0], self.cfg.num_modules),
            dtype=np.float64,
        )
        remaining = self.cfg.num_tasks - observed_tasks - 1
        for state, indices in groups.items():
            counts = self.support_counts(state, remaining)
            probabilities = counts / counts.sum()
            chosen = rng.choice(
                len(self.supports),
                size=len(indices),
                replace=True,
                p=probabilities,
            )
            coefficients = rng.choice(
                self.cfg.weight_values,
                size=(len(indices), 2),
                replace=True,
            )
            for row, support_index, values in zip(
                indices,
                chosen,
                coefficients,
                strict=True,
            ):
                latents[row, np.asarray(self.supports[int(support_index)])] = values
        return latents

    def sample_schedule(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a complete valid schedule through causal prefix transitions."""
        schedule = np.zeros(
            (self.cfg.num_tasks, self.cfg.num_modules),
            dtype=np.float64,
        )
        for task_index in range(self.cfg.num_tasks):
            schedule[task_index] = self.sample_next_batch(
                rng,
                schedule[None, :task_index],
            )[0]
        return schedule


@cache
def conditioned_schedule_prior(cfg: ScheduleConfig) -> ConditionedSchedulePrior:
    """Share immutable dynamic-programming tables across observer instances."""
    return ConditionedSchedulePrior(cfg)


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
        latent[-2:] = (first, second)
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
