"""Task-curriculum configuration and sampling."""

import math
from dataclasses import dataclass

import numpy as np

from iccl.data.teacher import HyperTeacher

TASK_ORIGIN_CODES = {"ordinary": 0, "backbone": 1, "surplus": 2, "final": 3, "revisit": 4}
TASK_CATEGORY_CODES = {
    "novel_support": 0,
    "seen_support_new_weights": 1,
    "exact_repeat": 2,
}
CURRICULUM_SAMPLER_CODES = {"rejection": 0, "constructive": 1, "structured": 2}


@dataclass(frozen=True)
class PhaseConfig:
    num_tasks: int
    hotness: tuple[int, int]


@dataclass(frozen=True)
class DemoCountConfig:
    """Inclusive demo-count range sampled once per sequence or per task."""

    min: int
    max: int
    scope: str


DemoCountSpec = int | tuple[int, int] | DemoCountConfig
SurplusTaskSpec = int | tuple[int, int]


@dataclass(frozen=True)
class SequenceConfig:
    phases: tuple[PhaseConfig, ...]
    demos_per_task: DemoCountSpec
    signal_boundaries: bool
    require_identifiable: bool
    require_full_rank: bool = False
    task_graph: str = "random"
    graph_ordered: bool = False
    curriculum_sampler: str = "rejection"
    hotness: int = 2
    surplus_tasks: SurplusTaskSpec | None = None
    max_attempts: int = 1000


@dataclass(frozen=True)
class CurriculumSample:
    latents: np.ndarray
    task_origins: np.ndarray
    pre_shuffle_indices: np.ndarray
    generation_categories: np.ndarray
    presentation_categories: np.ndarray
    generation_attempts: int
    num_surplus_tasks: int


def check_compositional(supports: np.ndarray, num_modules: int) -> bool:
    """Whether every module appears in at least one task support."""
    return supports.shape[1] == num_modules and bool((supports.sum(axis=0) > 0).all())


def check_connected(supports: np.ndarray) -> bool:
    """Whether used modules form one co-occurrence component."""
    used = supports.any(axis=0)
    if not used.any():
        return False
    reach = (supports.T.astype(np.int64) @ supports.astype(np.int64)) > 0
    reach |= np.eye(supports.shape[1], dtype=bool)
    for _ in range(max(1, math.ceil(math.log2(supports.shape[1])))):
        reach = (reach.astype(np.int64) @ reach.astype(np.int64)) > 0
    return bool(reach[np.ix_(used, used)].all())


def check_full_rank(latents: np.ndarray) -> bool:
    """Whether the latent matrix has full rank over its used modules."""
    used = latents.any(axis=0)
    return int(np.linalg.matrix_rank(latents[:, used])) == int(used.sum())


def draw_inclusive(spec: int | tuple[int, int], rng: np.random.Generator, *, name: str) -> int:
    """Draw an inclusive integer range without consuming RNG for a scalar."""
    lo, hi = (spec, spec) if isinstance(spec, int) else spec
    if lo < 0 or lo > hi:
        raise ValueError(f"{name} requires 0 <= min <= max, got [{lo}, {hi}]")
    return lo if lo == hi else int(rng.integers(lo, hi + 1))


def task_categories(latents: np.ndarray) -> np.ndarray:
    """Classify each task relative to preceding tasks in the supplied order."""
    categories = np.empty(len(latents), dtype=np.int8)
    seen: dict[tuple[int, ...], list[np.ndarray]] = {}
    for index, latent in enumerate(latents):
        support = tuple(int(module) for module in np.flatnonzero(latent))
        previous = seen.get(support)
        if previous is None:
            category = "novel_support"
        elif any(np.array_equal(latent, other) for other in previous):
            category = "exact_repeat"
        else:
            category = "seen_support_new_weights"
        categories[index] = TASK_CATEGORY_CODES[category]
        seen.setdefault(support, []).append(latent)
    return categories


def decode_prufer(sequence: np.ndarray, num_modules: int) -> list[tuple[int, int]]:
    """Decode a Prüfer sequence into labelled tree edges."""
    degrees = np.ones(num_modules, dtype=np.int64)
    for node in sequence:
        degrees[int(node)] += 1
    edges: list[tuple[int, int]] = []
    for node_raw in sequence:
        node = int(node_raw)
        leaf = int(np.flatnonzero(degrees == 1)[0])
        edges.append((leaf, node))
        degrees[[leaf, node]] -= 1
    remaining = np.flatnonzero(degrees == 1)
    edges.append((int(remaining[0]), int(remaining[1])))
    return edges


def weighted_edge(
    family: HyperTeacher, edge: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    pattern = np.zeros(family.cfg.num_modules, dtype=np.int8)
    pattern[list(edge)] = 1
    return family.apply_weighting(rng, pattern)


def _structured_base_size(task_graph: str, num_modules: int) -> int:
    if task_graph in {"chain", "star"}:
        return num_modules - 1
    if task_graph == "ring":
        return num_modules
    raise ValueError(f"unknown task_graph: {task_graph}")


def assert_feasible(cfg: SequenceConfig, num_modules: int) -> None:
    """Reject structurally impossible curricula before sampling."""
    if cfg.max_attempts < 1:
        raise ValueError(f"max_attempts must be positive, got {cfg.max_attempts}")
    if cfg.surplus_tasks is not None:
        if cfg.phases:
            raise ValueError("phases and surplus_tasks are mutually exclusive")
        if cfg.task_graph != "random":
            raise ValueError("relative curricula use curriculum_sampler, not task_graph")
        if cfg.curriculum_sampler not in {"constructive", "rejection"}:
            raise ValueError(f"unknown curriculum_sampler: {cfg.curriculum_sampler}")
        if cfg.hotness != 2:
            raise ValueError(f"relative curricula require exactly 2-hot tasks, got {cfg.hotness}")
        minimum = cfg.surplus_tasks if isinstance(cfg.surplus_tasks, int) else cfg.surplus_tasks[0]
        if minimum < 0:
            raise ValueError(f"surplus_tasks must be non-negative, got {minimum}")
        if cfg.require_full_rank and minimum < 1:
            raise ValueError("full rank requires T>=M, so surplus_tasks must be >=1")
        return

    num_tasks = sum(phase.num_tasks for phase in cfg.phases)
    if cfg.task_graph != "random":
        if any(phase.hotness != (2, 2) for phase in cfg.phases):
            raise ValueError(f"task_graph={cfg.task_graph} requires 2-hot phases")
        minimum = _structured_base_size(cfg.task_graph, num_modules)
        if num_tasks < minimum:
            raise ValueError(
                f"task_graph={cfg.task_graph} over M={num_modules} needs >= {minimum} tasks, "
                f"got {num_tasks}"
            )
        return
    if not cfg.require_identifiable:
        return
    max_hotness_sum = sum(phase.num_tasks * phase.hotness[1] for phase in cfg.phases)
    max_cover = 1 + max_hotness_sum - num_tasks
    if max_hotness_sum < num_modules or max_cover < num_modules:
        raise ValueError(
            f"phases cannot connectedly cover M={num_modules}; maximum cover is {max_cover}"
        )


def _result(
    latents: np.ndarray,
    origins: np.ndarray,
    *,
    attempts: int,
    surplus: int,
    order: np.ndarray | None = None,
) -> CurriculumSample:
    order = np.arange(len(latents), dtype=np.int64) if order is None else order
    presented = latents[order]
    return CurriculumSample(
        latents=presented,
        task_origins=origins[order],
        pre_shuffle_indices=order,
        generation_categories=task_categories(latents)[order],
        presentation_categories=task_categories(presented),
        generation_attempts=attempts,
        num_surplus_tasks=surplus,
    )


def _structured_latents(
    family: HyperTeacher, cfg: SequenceConfig, rng: np.random.Generator
) -> np.ndarray:
    modules = family.cfg.num_modules
    tasks = sum(phase.num_tasks for phase in cfg.phases)
    permutation = rng.permutation(modules)
    if cfg.task_graph == "chain":
        base = [(permutation[i], permutation[i + 1]) for i in range(modules - 1)]
    elif cfg.task_graph == "ring":
        base = [(permutation[i], permutation[(i + 1) % modules]) for i in range(modules)]
    elif cfg.task_graph == "star":
        base = [(permutation[0], permutation[i]) for i in range(1, modules)]
    else:
        raise ValueError(f"unknown task_graph: {cfg.task_graph}")
    edges = base + [base[int(rng.integers(len(base)))] for _ in range(tasks - len(base))]
    if not cfg.graph_ordered:
        edges = [edges[index] for index in rng.permutation(len(edges))]
    return np.stack([weighted_edge(family, edge, rng) for edge in edges])


def _constructive(
    family: HyperTeacher, cfg: SequenceConfig, rng: np.random.Generator, surplus: int
) -> CurriculumSample:
    modules = family.cfg.num_modules
    for attempt in range(1, cfg.max_attempts + 1):
        prufer = (
            rng.integers(0, modules, size=modules - 2, dtype=np.int64)
            if modules > 2
            else np.empty(0, dtype=np.int64)
        )
        latents = [weighted_edge(family, edge, rng) for edge in decode_prufer(prufer, modules)]
        latents.extend(
            family.apply_weighting(rng, family.sample_pattern(rng, cfg.hotness))
            for _ in range(surplus)
        )
        stacked = np.stack(latents)
        if cfg.require_full_rank and not check_full_rank(stacked):
            continue
        origins = np.array(
            [TASK_ORIGIN_CODES["backbone"]] * (modules - 1)
            + [TASK_ORIGIN_CODES["surplus"]] * surplus,
            dtype=np.int8,
        )
        order = rng.permutation(len(stacked)).astype(np.int64)
        sample = _result(stacked, origins, attempts=attempt, surplus=surplus, order=order)
        supports = sample.latents > 0
        if not check_compositional(supports, modules) or not check_connected(supports):
            raise AssertionError("constructive backbone violated coverage or connectivity")
        return sample
    raise RuntimeError(
        f"constructive sampler failed after {cfg.max_attempts} attempts "
        f"for M={modules}, S={surplus}, weighting={family.cfg.weighting}"
    )


def _rejection(
    family: HyperTeacher, cfg: SequenceConfig, rng: np.random.Generator, tasks: int, surplus: int
) -> CurriculumSample:
    for attempt in range(1, cfg.max_attempts + 1):
        latents = []
        if cfg.surplus_tasks is None:
            for phase in cfg.phases:
                for _ in range(phase.num_tasks):
                    hotness = int(rng.integers(phase.hotness[0], phase.hotness[1] + 1))
                    latents.append(family.apply_weighting(rng, family.sample_pattern(rng, hotness)))
        else:
            latents = [
                family.apply_weighting(rng, family.sample_pattern(rng, cfg.hotness))
                for _ in range(tasks)
            ]
        stacked = np.stack(latents)
        supports = stacked > 0
        valid = (
            not cfg.require_identifiable
            or check_compositional(supports, family.cfg.num_modules)
            and check_connected(supports)
        ) and (not cfg.require_full_rank or check_full_rank(stacked))
        if valid:
            ordinary = tasks - surplus
            origins = np.array(
                [TASK_ORIGIN_CODES["ordinary"]] * ordinary
                + [TASK_ORIGIN_CODES["surplus"]] * surplus,
                dtype=np.int8,
            )
            return _result(stacked, origins, attempts=attempt, surplus=surplus)
    raise RuntimeError(
        f"rejection sampler failed after {cfg.max_attempts} attempts for "
        f"M={family.cfg.num_modules}, S={surplus}, T={tasks}, hotness={cfg.hotness}, "
        f"full_rank={cfg.require_full_rank}"
    )


def sample_curriculum(
    family: HyperTeacher, cfg: SequenceConfig, rng: np.random.Generator
) -> CurriculumSample:
    """Sample a fixed-phase, constructive, or relative rejection curriculum."""
    if cfg.surplus_tasks is not None:
        surplus = draw_inclusive(cfg.surplus_tasks, rng, name="surplus_tasks")
        tasks = family.cfg.num_modules - 1 + surplus
        return (
            _constructive(family, cfg, rng, surplus)
            if cfg.curriculum_sampler == "constructive"
            else _rejection(family, cfg, rng, tasks, surplus)
        )
    if cfg.task_graph != "random":
        latents = _structured_latents(family, cfg, rng)
        origins = np.full(len(latents), TASK_ORIGIN_CODES["ordinary"], dtype=np.int8)
        return _result(latents, origins, attempts=1, surplus=0)
    tasks = sum(phase.num_tasks for phase in cfg.phases)
    return _rejection(family, cfg, rng, tasks, surplus=0)
