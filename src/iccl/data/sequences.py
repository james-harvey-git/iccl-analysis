"""ICCL sequence construction on top of the HyperTeacher task family.

A sequence is one continual-learning episode: a freshly sampled module pool
("world"), a curriculum of tasks drawn per phase, and per-task demonstration
runs serialized into a token stream. Identifiability of the primitives is
enforced by rejection-resampling the task set until its support covers all
modules and its module co-occurrence graph is connected.

Token layout (two tokens per demonstration): the stream alternates x-token,
y-token, with a dedicated all-zero boundary token opening each task when
boundaries are signalled. Loss is computed at x-token positions only, where the
target is that demonstration's y.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iccl.data.teacher import HyperTeacher, ModulePool, sample_module_pool, teacher_forward

TOKEN_X = 0
TOKEN_Y = 1
TOKEN_BOUNDARY = 2
TOKEN_PAD = 3

TASK_ORIGIN_CODES = {
    "legacy": 0,
    "backbone": 1,
    "surplus": 2,
    "final": 3,
    "revisit": 4,
}

TASK_CATEGORY_CODES = {
    "first": 0,
    "novel_support": 1,
    "seen_support_new_weights": 2,
    "exact_repeat": 3,
}

CURRICULUM_SAMPLER_CODES = {
    "rejection": 0,
    "constructive": 1,
    "structured": 2,
}


@dataclass(frozen=True)
class PhaseConfig:
    num_tasks: int
    hotness: tuple[int, int]  # inclusive [lo, hi] range


@dataclass(frozen=True)
class FinalTaskConfig:
    """Evaluation task appended after the curriculum phases.

    Modes: "composite" (constituent modules covered by the demonstrated
    support, pattern itself never demonstrated in-sequence), "any"
    (unconstrained pattern — e.g. for higher-hotness evals).
    """

    mode: str
    hotness: int
    num_demos: int


@dataclass(frozen=True)
class DemoCountConfig:
    """Inclusive demonstration-count distribution.

    ``per_sequence`` draws one count for the curriculum; ``per_task`` draws
    independently for each curriculum task. Equal bounds are a fixed value and
    deliberately consume no random number.
    """

    min: int
    max: int
    scope: str


DemoCountSpec = int | tuple[int, int] | DemoCountConfig
SurplusTaskSpec = int | tuple[int, int]


@dataclass(frozen=True)
class SequenceConfig:
    """``task_graph`` selects the overlap-graph family of the curriculum's task
    set: "random" draws tasks i.i.d. per phase (rejection-sampled for
    identifiability); "chain", "ring", and "star" construct the co-occurrence
    graph explicitly over a random module permutation, which requires every
    phase to be exactly 2-hot (tasks are graph edges). ``graph_ordered`` keeps
    the structured construction order (e.g. a chain presented as a path);
    otherwise task order is shuffled."""

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


@dataclass
class SequenceSample:
    tokens: np.ndarray  # [seq_len, max(input_dim, output_dim)] float32
    token_type: np.ndarray  # [seq_len] int64
    targets: np.ndarray  # [seq_len, output_dim] float32, nonzero at x-positions
    loss_mask: np.ndarray  # [seq_len] float32, 1 at x-positions
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurriculumSample:
    """Sampled curriculum plus its construction provenance."""

    latents: np.ndarray
    task_origins: np.ndarray
    pre_shuffle_indices: np.ndarray
    generation_categories: np.ndarray
    presentation_categories: np.ndarray
    generation_attempts: int
    num_surplus_tasks: int


def check_compositional(supports: np.ndarray, num_modules: int) -> bool:
    """True if every module of the pool appears in at least one task support."""
    return bool((supports.sum(axis=0) > 0).all()) and supports.shape[1] == num_modules


def check_connected(supports: np.ndarray) -> bool:
    """True if the module co-occurrence graph over used modules is one connected
    component (boolean-matrix-powering port of the source's check_connected)."""
    used = supports.any(axis=0)
    if not used.any():
        return False
    adjacency = (supports.T.astype(np.int64) @ supports.astype(np.int64)) > 0
    reach = adjacency | np.eye(supports.shape[1], dtype=bool)
    for _ in range(max(1, math.ceil(math.log2(supports.shape[1])))):
        reach = (reach.astype(np.int64) @ reach.astype(np.int64)) > 0
    return bool(reach[np.ix_(used, used)].all())


def check_full_rank(latents: np.ndarray) -> bool:
    """True if the task-latent matrix has full column rank over used modules
    (stricter identifiability criterion for non-binary weightings)."""
    used = latents.any(axis=0)
    return int(np.linalg.matrix_rank(latents[:, used])) == int(used.sum())


def _inclusive_draw(
    spec: int | tuple[int, int], rng: np.random.Generator, *, name: str
) -> int:
    """Draw from an inclusive integer range, preserving fixed-value RNG streams."""
    if isinstance(spec, int):
        value = spec
    else:
        lo, hi = spec
        if lo > hi:
            raise ValueError(f"{name} requires min <= max, got [{lo}, {hi}]")
        value = lo if lo == hi else int(rng.integers(lo, hi + 1))
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _task_categories(latents: np.ndarray) -> np.ndarray:
    """Classify tasks relative to tasks preceding them in the supplied order."""
    categories = np.empty(len(latents), dtype=np.int8)
    seen: dict[tuple[int, ...], list[np.ndarray]] = {}
    for i, latent in enumerate(latents):
        support = tuple(int(m) for m in np.flatnonzero(latent))
        previous = seen.get(support)
        if i == 0:
            category = "first"
        elif previous is None:
            category = "novel_support"
        elif any(np.array_equal(latent, other) for other in previous):
            category = "exact_repeat"
        else:
            category = "seen_support_new_weights"
        categories[i] = TASK_CATEGORY_CODES[category]
        seen.setdefault(support, []).append(latent)
    return categories


def _decode_prufer(sequence: np.ndarray, num_modules: int) -> list[tuple[int, int]]:
    """Decode a Prüfer sequence into the corresponding labelled tree edges."""
    if num_modules < 2:
        raise ValueError(f"a 2-hot curriculum needs at least 2 modules, got {num_modules}")
    degrees = np.ones(num_modules, dtype=np.int64)
    for node in sequence:
        degrees[int(node)] += 1
    edges: list[tuple[int, int]] = []
    for node_raw in sequence:
        node = int(node_raw)
        leaf = int(np.flatnonzero(degrees == 1)[0])
        edges.append((leaf, node))
        degrees[leaf] -= 1
        degrees[node] -= 1
    remaining = np.flatnonzero(degrees == 1)
    edges.append((int(remaining[0]), int(remaining[1])))
    return edges


def _weighted_edge(
    family: HyperTeacher, edge: tuple[int, int], rng: np.random.Generator
) -> np.ndarray:
    pattern = np.zeros(family.cfg.num_modules, dtype=np.int8)
    pattern[list(edge)] = 1
    return family.apply_weighting(rng, pattern)


def _structured_base_size(task_graph: str, num_modules: int) -> int:
    """Number of edges in the family's covering skeleton."""
    match task_graph:
        case "chain" | "star":
            return num_modules - 1
        case "ring":
            return num_modules
        case _:
            raise ValueError(f"unknown task_graph: {task_graph}")


def assert_feasible(cfg: SequenceConfig, num_modules: int) -> None:
    """Fail fast on configurations that cannot produce valid sequences.

    For "random" graphs with identifiability required: a connected union of
    tasks with hotness k_i covers at most 1 + sum(k_i - 1) modules. For
    structured graphs: every phase must be exactly 2-hot (tasks are edges) and
    there must be at least as many tasks as skeleton edges."""
    if cfg.surplus_tasks is not None:
        if cfg.phases:
            raise ValueError("absolute phases and relative surplus_tasks are mutually exclusive")
        if cfg.task_graph != "random":
            raise ValueError("variable-world curricula use curriculum_sampler, not task_graph")
        if cfg.curriculum_sampler not in {"constructive", "rejection"}:
            raise ValueError(f"unknown curriculum_sampler: {cfg.curriculum_sampler}")
        if cfg.hotness != 2:
            raise ValueError(
                f"variable-world curricula require exactly 2-hot tasks, got hotness={cfg.hotness}"
            )
        minimum_surplus = (
            cfg.surplus_tasks
            if isinstance(cfg.surplus_tasks, int)
            else cfg.surplus_tasks[0]
        )
        if minimum_surplus < 0:
            raise ValueError(f"surplus_tasks must be non-negative, got {minimum_surplus}")
        if cfg.require_full_rank and minimum_surplus < 1:
            raise ValueError(
                f"full rank over M={num_modules} needs T>=M, so surplus_tasks must be >=1"
            )
        return

    num_tasks = sum(p.num_tasks for p in cfg.phases)
    if cfg.task_graph != "random":
        if any(p.hotness != (2, 2) for p in cfg.phases):
            raise ValueError(f"task_graph={cfg.task_graph} requires all phases to be 2-hot")
        base = _structured_base_size(cfg.task_graph, num_modules)
        if num_tasks < base:
            raise ValueError(
                f"task_graph={cfg.task_graph} over {num_modules} modules needs "
                f">= {base} tasks, got {num_tasks}"
            )
        return
    if not cfg.require_identifiable:
        return
    max_hotness_sum = sum(p.num_tasks * p.hotness[1] for p in cfg.phases)
    max_connected_cover = 1 + max_hotness_sum - num_tasks
    if max_hotness_sum < num_modules or max_connected_cover < num_modules:
        raise ValueError(
            f"phases cannot connectedly cover {num_modules} modules: "
            f"max connected cover is {max_connected_cover} "
            f"(need more tasks or higher hotness)"
        )


def _structured_latents(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Construct the curriculum for a structured overlap-graph family.

    The family's covering skeleton is laid over a random permutation of module
    labels; tasks beyond the skeleton duplicate uniformly-drawn skeleton edges.
    Coverage and connectivity hold by construction. 2-hot only: for hotness > 2
    the overlap structure is a hypergraph and each family would need an overlap
    parameter (e.g. a hyperchain's shared-module count); see the dataset spec.
    """
    num_modules = family.cfg.num_modules
    num_tasks = sum(p.num_tasks for p in cfg.phases)
    perm = rng.permutation(num_modules)
    match cfg.task_graph:
        case "chain":
            base = [(perm[i], perm[i + 1]) for i in range(num_modules - 1)]
        case "ring":
            base = [(perm[i], perm[(i + 1) % num_modules]) for i in range(num_modules)]
        case "star":
            base = [(perm[0], perm[i]) for i in range(1, num_modules)]
        case _:
            raise ValueError(f"unknown task_graph: {cfg.task_graph}")
    edges = base + [base[rng.integers(len(base))] for _ in range(num_tasks - len(base))]
    if not cfg.graph_ordered:
        edges = [edges[i] for i in rng.permutation(len(edges))]
    latents = []
    for a, b in edges:
        pattern = np.zeros(num_modules, dtype=np.int8)
        pattern[[a, b]] = 1
        latents.append(family.apply_weighting(rng, pattern))
    return np.stack(latents)


def _constructive_curriculum(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    num_surplus: int,
    max_attempts: int,
) -> CurriculumSample:
    """Uniform labelled spanning-tree backbone plus unbiased ordinary surplus tasks."""
    num_modules = family.cfg.num_modules
    for attempt in range(1, max_attempts + 1):
        prufer = (
            rng.integers(0, num_modules, size=num_modules - 2, dtype=np.int64)
            if num_modules > 2
            else np.empty(0, dtype=np.int64)
        )
        edges = _decode_prufer(prufer, num_modules)
        generated = [_weighted_edge(family, edge, rng) for edge in edges]
        for _ in range(num_surplus):
            pattern = family.sample_pattern(rng, cfg.hotness)
            generated.append(family.apply_weighting(rng, pattern))
        latents = np.stack(generated)
        if cfg.require_full_rank and not check_full_rank(latents):
            continue

        origins = np.array(
            [TASK_ORIGIN_CODES["backbone"]] * (num_modules - 1)
            + [TASK_ORIGIN_CODES["surplus"]] * num_surplus,
            dtype=np.int8,
        )
        generation_categories = _task_categories(latents)
        order = rng.permutation(len(latents))
        presented = latents[order]
        return CurriculumSample(
            latents=presented,
            task_origins=origins[order],
            pre_shuffle_indices=order.astype(np.int64),
            generation_categories=generation_categories[order],
            presentation_categories=_task_categories(presented),
            generation_attempts=attempt,
            num_surplus_tasks=num_surplus,
        )
    raise RuntimeError(
        f"no full-rank constructive curriculum found in {max_attempts} attempts for "
        f"M={num_modules}, S={num_surplus}, weighting={family.cfg.weighting}"
    )


def _rejection_curriculum(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    num_surplus: int,
    max_attempts: int,
) -> CurriculumSample:
    """I.i.d. ordinary tasks conditioned on the configured structural checks."""
    num_tasks = family.cfg.num_modules - 1 + num_surplus
    for attempt in range(1, max_attempts + 1):
        latents = np.stack(
            [
                family.apply_weighting(rng, family.sample_pattern(rng, cfg.hotness))
                for _ in range(num_tasks)
            ]
        )
        if cfg.require_identifiable:
            supports = latents > 0
            if not check_compositional(supports, family.cfg.num_modules):
                continue
            if not check_connected(supports):
                continue
            if cfg.require_full_rank and not check_full_rank(latents):
                continue
        return CurriculumSample(
            latents=latents,
            task_origins=np.full(num_tasks, TASK_ORIGIN_CODES["legacy"], dtype=np.int8),
            pre_shuffle_indices=np.arange(num_tasks, dtype=np.int64),
            generation_categories=_task_categories(latents),
            presentation_categories=_task_categories(latents),
            generation_attempts=attempt,
            num_surplus_tasks=num_surplus,
        )
    raise RuntimeError(
        f"no identifiable task set found in {max_attempts} attempts for "
        f"M={family.cfg.num_modules}, S={num_surplus}"
    )


def _sample_curriculum_latents(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> CurriculumSample:
    """Sample the phase tasks' latents [T, M]: constructively for structured
    graph families, otherwise i.i.d. with rejection-resampling until the
    identifiability checks pass (when required)."""
    if cfg.surplus_tasks is not None:
        num_surplus = _inclusive_draw(cfg.surplus_tasks, rng, name="surplus_tasks")
        if cfg.curriculum_sampler == "constructive":
            return _constructive_curriculum(family, cfg, rng, num_surplus, max_attempts)
        return _rejection_curriculum(family, cfg, rng, num_surplus, max_attempts)

    if cfg.task_graph != "random":
        latents = _structured_latents(family, cfg, rng)
        num_tasks = len(latents)
        return CurriculumSample(
            latents=latents,
            task_origins=np.full(num_tasks, TASK_ORIGIN_CODES["legacy"], dtype=np.int8),
            pre_shuffle_indices=np.arange(num_tasks, dtype=np.int64),
            generation_categories=_task_categories(latents),
            presentation_categories=_task_categories(latents),
            generation_attempts=1,
            num_surplus_tasks=0,
        )
    for attempt in range(1, max_attempts + 1):
        latents = []
        for phase in cfg.phases:
            lo, hi = phase.hotness
            for _ in range(phase.num_tasks):
                hotness = int(rng.integers(lo, hi + 1))
                pattern = family.sample_pattern(rng, hotness)
                latents.append(family.apply_weighting(rng, pattern))
        stacked = np.stack(latents)
        if not cfg.require_identifiable:
            num_tasks = len(stacked)
            return CurriculumSample(
                latents=stacked,
                task_origins=np.full(num_tasks, TASK_ORIGIN_CODES["legacy"], dtype=np.int8),
                pre_shuffle_indices=np.arange(num_tasks, dtype=np.int64),
                generation_categories=_task_categories(stacked),
                presentation_categories=_task_categories(stacked),
                generation_attempts=attempt,
                num_surplus_tasks=0,
            )
        supports = stacked > 0
        if not check_compositional(supports, family.cfg.num_modules):
            continue
        if not check_connected(supports):
            continue
        if cfg.require_full_rank and not check_full_rank(stacked):
            continue
        num_tasks = len(stacked)
        return CurriculumSample(
            latents=stacked,
            task_origins=np.full(num_tasks, TASK_ORIGIN_CODES["legacy"], dtype=np.int8),
            pre_shuffle_indices=np.arange(num_tasks, dtype=np.int64),
            generation_categories=_task_categories(stacked),
            presentation_categories=_task_categories(stacked),
            generation_attempts=attempt,
            num_surplus_tasks=0,
        )
    raise RuntimeError(
        f"no identifiable task set found in {max_attempts} attempts; "
        f"the phase configuration is likely too tight for connected coverage"
    )


def _sample_final_latent(
    family: HyperTeacher,
    final: FinalTaskConfig,
    curriculum_latents: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> np.ndarray:
    match final.mode:
        case "any":
            pattern = family.sample_pattern(rng, final.hotness)
            return family.apply_weighting(rng, pattern)
        case "composite":
            covered = np.flatnonzero(curriculum_latents.any(axis=0))
            if len(covered) < final.hotness:
                raise ValueError(
                    f"composite final task needs {final.hotness} covered modules, "
                    f"but only {len(covered)} are covered by the curriculum"
                )
            demonstrated = {tuple(np.flatnonzero(lat)) for lat in curriculum_latents}
            for _ in range(max_attempts):
                chosen = rng.choice(covered, size=final.hotness, replace=False)
                if tuple(sorted(chosen)) not in demonstrated:
                    pattern = np.zeros(family.cfg.num_modules, dtype=np.int8)
                    pattern[chosen] = 1
                    return family.apply_weighting(rng, pattern)
            raise RuntimeError("no undemonstrated composite pattern found")
        case _:
            raise ValueError(f"unknown final-task mode: {final.mode}")


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
        lo, hi = spec
        return [_draw_demo_count(lo, hi, rng) for _ in range(num_tasks)]
    if spec.scope == "per_sequence":
        count = _draw_demo_count(spec.min, spec.max, rng)
        return [count] * num_tasks
    if spec.scope == "per_task":
        return [_draw_demo_count(spec.min, spec.max, rng) for _ in range(num_tasks)]
    raise ValueError(
        f"demos_per_task.scope must be per_sequence or per_task, got {spec.scope!r}"
    )


def build_sequence(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    *,
    final_task: FinalTaskConfig | None = None,
    history: bool = True,
    revisit_demos: int = 0,
    include_world: bool = False,
    world: ModulePool | None = None,
    fixed_final_latent: np.ndarray | None = None,
) -> SequenceSample:
    """Build one ICCL sequence.

    ``history=False`` drops the curriculum phases entirely (the no-history
    control for few-shot composite evals); it requires a ``fixed_final_latent``
    and ``world``, typically taken from a paired full-history sequence so the
    control shares the exact same composed teacher. ``revisit_demos`` appends a
    fresh demonstration run of the first curriculum task at the end of the
    sequence (retention eval).
    """
    assert_feasible(cfg, family.cfg.num_modules)
    pool = world if world is not None else sample_module_pool(family.cfg, rng)

    if history:
        curriculum = _sample_curriculum_latents(family, cfg, rng)
        curriculum_latents = curriculum.latents
    else:
        if fixed_final_latent is None or world is None:
            raise ValueError("history=False requires fixed_final_latent and world")
        curriculum_latents = np.zeros((0, family.cfg.num_modules), dtype=np.float32)
        curriculum = CurriculumSample(
            latents=curriculum_latents,
            task_origins=np.empty(0, dtype=np.int8),
            pre_shuffle_indices=np.empty(0, dtype=np.int64),
            generation_categories=np.empty(0, dtype=np.int8),
            presentation_categories=np.empty(0, dtype=np.int8),
            generation_attempts=0,
            num_surplus_tasks=0,
        )

    latents = [lat for lat in curriculum_latents]
    task_origins = list(curriculum.task_origins)
    pre_shuffle_indices = list(curriculum.pre_shuffle_indices)
    generation_categories = list(curriculum.generation_categories)
    if final_task is not None:
        if fixed_final_latent is not None:
            latents.append(fixed_final_latent.astype(np.float32))
        else:
            latents.append(_sample_final_latent(family, final_task, curriculum_latents, rng))
        task_origins.append(TASK_ORIGIN_CODES["final"])
        pre_shuffle_indices.append(len(pre_shuffle_indices))
    elif fixed_final_latent is not None:
        latents.append(fixed_final_latent.astype(np.float32))
        task_origins.append(TASK_ORIGIN_CODES["final"])
        pre_shuffle_indices.append(len(pre_shuffle_indices))
    if revisit_demos > 0:
        if len(curriculum_latents) == 0:
            raise ValueError("revisit requires curriculum history")
        latents.append(curriculum_latents[0])
        task_origins.append(TASK_ORIGIN_CODES["revisit"])
        pre_shuffle_indices.append(len(pre_shuffle_indices))

    stacked_latents = np.stack(latents)
    presentation_categories = _task_categories(stacked_latents)
    generation_categories.extend(
        int(category) for category in presentation_categories[len(generation_categories) :]
    )

    demo_counts = _curriculum_demo_counts(cfg, len(curriculum_latents), rng)
    if final_task is not None or fixed_final_latent is not None:
        if final_task is None:
            raise ValueError("fixed_final_latent requires final_task metadata")
        if final_task.num_demos < 1:
            raise ValueError(f"final task needs at least one demo, got {final_task.num_demos}")
        demo_counts.append(final_task.num_demos)
    if revisit_demos > 0:
        demo_counts.append(revisit_demos)

    token_dim = max(family.cfg.input_dim, family.cfg.output_dim)
    seq_len = sum(2 * n for n in demo_counts)
    if cfg.signal_boundaries:
        seq_len += len(latents)

    tokens = np.zeros((seq_len, token_dim), dtype=np.float32)
    token_type = np.full(seq_len, TOKEN_PAD, dtype=np.int64)
    targets = np.zeros((seq_len, family.cfg.output_dim), dtype=np.float32)
    loss_mask = np.zeros(seq_len, dtype=np.float32)

    boundaries = []
    task_spans = []
    base_mse = []
    pos = 0
    for latent, num_demos in zip(latents, demo_counts, strict=True):
        if cfg.signal_boundaries:
            boundaries.append(pos)
            token_type[pos] = TOKEN_BOUNDARY
            pos += 1
        start = pos
        x = rng.uniform(-1.0, 1.0, size=(num_demos, family.cfg.input_dim)).astype(np.float32)
        y = teacher_forward(pool, latent, x)
        base_mse.append(((y - y.mean(axis=0)) ** 2).mean(axis=0))
        for j in range(num_demos):
            tokens[pos, : family.cfg.input_dim] = x[j]
            token_type[pos] = TOKEN_X
            targets[pos] = y[j]
            loss_mask[pos] = 1.0
            pos += 1
            tokens[pos, : family.cfg.output_dim] = y[j]
            token_type[pos] = TOKEN_Y
            pos += 1
        task_spans.append((start, pos))

    history_prediction_tokens = np.zeros(len(latents), dtype=np.int64)
    history_serialized_tokens = np.zeros(len(latents), dtype=np.int64)
    unique_supports_seen = np.zeros(len(latents), dtype=np.int64)
    modules_covered = np.zeros(len(latents), dtype=np.int64)
    seen_supports: set[tuple[int, ...]] = set()
    covered_modules: set[int] = set()
    prediction_prefix = 0
    serialized_prefix = 0
    for i, (latent, count) in enumerate(zip(latents, demo_counts, strict=True)):
        history_prediction_tokens[i] = prediction_prefix
        history_serialized_tokens[i] = serialized_prefix
        unique_supports_seen[i] = len(seen_supports)
        modules_covered[i] = len(covered_modules)
        support = tuple(int(m) for m in np.flatnonzero(latent))
        seen_supports.add(support)
        covered_modules.update(support)
        prediction_prefix += count
        serialized_prefix += 2 * count + int(cfg.signal_boundaries)

    sampler_name = (
        cfg.curriculum_sampler
        if cfg.surplus_tasks is not None
        else ("structured" if cfg.task_graph != "random" else "rejection")
    )
    info: dict[str, Any] = {
        "latents": stacked_latents,
        "demo_counts": np.array(demo_counts, dtype=np.int64),
        "boundaries": np.array(boundaries, dtype=np.int64),
        "task_spans": np.array(task_spans, dtype=np.int64),
        "base_mse": np.stack(base_mse),
        "num_curriculum_tasks": len(curriculum_latents),
        "num_modules": family.cfg.num_modules,
        "num_surplus_tasks": curriculum.num_surplus_tasks,
        "num_prediction_tokens": int(sum(demo_counts)),
        "serialized_length": seq_len,
        "curriculum_sampler": CURRICULUM_SAMPLER_CODES[sampler_name],
        "generation_attempts": curriculum.generation_attempts,
        "task_origin": np.array(task_origins, dtype=np.int8),
        "pre_shuffle_index": np.array(pre_shuffle_indices, dtype=np.int64),
        "generation_category": np.array(generation_categories, dtype=np.int8),
        "presentation_category": presentation_categories,
        "history_prediction_tokens": history_prediction_tokens,
        "history_serialized_tokens": history_serialized_tokens,
        "target_first_prediction_index": np.array(task_spans, dtype=np.int64)[:, 0],
        "num_unique_supports_seen": unique_supports_seen,
        "num_modules_covered": modules_covered,
    }
    if revisit_demos > 0:
        original_last_prediction = int(task_spans[0][0] + 2 * (demo_counts[0] - 1))
        revisit_first_prediction = int(task_spans[-1][0])
        info["intervening_tasks"] = len(curriculum_latents) - 1
        info["prediction_token_delay"] = int(
            history_prediction_tokens[-1] - demo_counts[0]
        )
        info["serialized_token_delay"] = revisit_first_prediction - original_last_prediction
    if include_world:
        info["world"] = pool
    return SequenceSample(
        tokens=tokens, token_type=token_type, targets=targets, loss_mask=loss_mask, info=info
    )


def _retention_control_latent(
    family: HyperTeacher,
    curriculum_latents: np.ndarray,
    revisited: np.ndarray,
    num_demos: int,
    rng: np.random.Generator,
    mode: str,
    max_attempts: int = 1000,
) -> np.ndarray:
    """The task a retention control demonstrates in place of the revisited one.

    "novel" draws an undemonstrated support of the same hotness from the modules
    the curriculum covers; "shared" keeps the revisited task's support and
    redraws its weights, which is only a distinct task under a non-binary
    weighting (binary latents *are* their support)."""
    hotness = int(np.count_nonzero(revisited))
    match mode:
        case "novel":
            final = FinalTaskConfig(mode="composite", hotness=hotness, num_demos=num_demos)
            return _sample_final_latent(family, final, curriculum_latents, rng)
        case "shared":
            if family.cfg.weighting == "binary":
                raise ValueError(
                    "the shared-module retention control is degenerate under "
                    "weighting=binary: a task is its support, so the control would "
                    "duplicate the revisit block"
                )
            pattern = (revisited > 0).astype(np.int8)
            for _ in range(max_attempts):
                latent = family.apply_weighting(rng, pattern)
                if not np.array_equal(latent, revisited):
                    return latent
            raise RuntimeError("no same-support latent distinct from the revisited task found")
        case _:
            raise ValueError(f"unknown retention-control mode: {mode}")


def build_paired_retention_control(
    family: HyperTeacher,
    sequence: SequenceSample,
    rng: np.random.Generator,
    *,
    mode: str,
) -> SequenceSample:
    """Position-matched control for a built retention sequence: the same world,
    curriculum, and tokens, with the final block demonstrating a different task
    over the revisit block's own inputs.

    The pairing is exact rather than distributional — everything up to the final
    block is identical, so the model enters that block in the same state and,
    since the inputs are shared, emits the same prediction at its first
    demonstration. Only the block's targets differ. See
    ``_retention_control_latent`` for what ``mode`` selects.
    """
    if "world" not in sequence.info:
        raise ValueError("paired retention control requires the sequence to carry its world")
    latents = sequence.info["latents"]
    curriculum = int(sequence.info["num_curriculum_tasks"])
    if curriculum == 0 or len(latents) <= curriculum:
        raise ValueError("paired retention control requires a sequence with a revisit block")

    start, _ = sequence.info["task_spans"][-1]
    num_demos = int(sequence.info["demo_counts"][-1])
    replacement = _retention_control_latent(
        family, latents[:curriculum], latents[-1], num_demos, rng, mode
    )

    x_positions = start + 2 * np.arange(num_demos)
    x = sequence.tokens[x_positions, : family.cfg.input_dim]
    y = teacher_forward(sequence.info["world"], replacement, x)

    tokens = sequence.tokens.copy()
    targets = sequence.targets.copy()
    tokens[x_positions + 1, : family.cfg.output_dim] = y
    targets[x_positions] = y

    info = dict(sequence.info)
    info["latents"] = np.concatenate([latents[:-1], replacement[None].astype(np.float32)])
    info["base_mse"] = sequence.info["base_mse"].copy()
    info["base_mse"][-1] = ((y - y.mean(axis=0)) ** 2).mean(axis=0)
    return SequenceSample(
        tokens=tokens,
        token_type=sequence.token_type.copy(),
        targets=targets,
        loss_mask=sequence.loss_mask.copy(),
        info=info,
    )


def build_paired_control(
    family: HyperTeacher,
    cfg: SequenceConfig,
    sequence: SequenceSample,
    final_task: FinalTaskConfig,
    rng: np.random.Generator,
) -> SequenceSample:
    """No-history control for a built eval sequence: the same world and final
    task latent, with the curriculum phases removed. Fresh inputs are drawn for
    the demonstrations; the pairing is on the task function, not the x draws."""
    if "world" not in sequence.info:
        raise ValueError("paired control requires the sequence to carry its world")
    control = build_sequence(
        family,
        cfg,
        rng,
        final_task=final_task,
        history=False,
        include_world=True,
        world=sequence.info["world"],
        fixed_final_latent=sequence.info["latents"][-1],
    )
    return control
