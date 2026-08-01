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
class SequenceConfig:
    """``task_graph`` selects the overlap-graph family of the curriculum's task
    set: "random" draws tasks i.i.d. per phase (rejection-sampled for
    identifiability); "chain", "ring", and "star" construct the co-occurrence
    graph explicitly over a random module permutation, which requires every
    phase to be exactly 2-hot (tasks are graph edges). ``graph_ordered`` keeps
    the structured construction order (e.g. a chain presented as a path);
    otherwise task order is shuffled."""

    phases: tuple[PhaseConfig, ...]
    demos_per_task: int | tuple[int, int]
    signal_boundaries: bool
    require_identifiable: bool
    require_full_rank: bool = False
    task_graph: str = "random"
    graph_ordered: bool = False


@dataclass
class SequenceSample:
    tokens: np.ndarray  # [seq_len, max(input_dim, output_dim)] float32
    token_type: np.ndarray  # [seq_len] int64
    targets: np.ndarray  # [seq_len, output_dim] float32, nonzero at x-positions
    loss_mask: np.ndarray  # [seq_len] float32, 1 at x-positions
    info: dict[str, Any] = field(default_factory=dict)


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


def _sample_curriculum_latents(
    family: HyperTeacher,
    cfg: SequenceConfig,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> np.ndarray:
    """Sample the phase tasks' latents [T, M]: constructively for structured
    graph families, otherwise i.i.d. with rejection-resampling until the
    identifiability checks pass (when required)."""
    if cfg.task_graph != "random":
        return _structured_latents(family, cfg, rng)
    for _ in range(max_attempts):
        latents = []
        for phase in cfg.phases:
            lo, hi = phase.hotness
            for _ in range(phase.num_tasks):
                hotness = int(rng.integers(lo, hi + 1))
                pattern = family.sample_pattern(rng, hotness)
                latents.append(family.apply_weighting(rng, pattern))
        stacked = np.stack(latents)
        if not cfg.require_identifiable:
            return stacked
        supports = stacked > 0
        if not check_compositional(supports, family.cfg.num_modules):
            continue
        if not check_connected(supports):
            continue
        if cfg.require_full_rank and not check_full_rank(stacked):
            continue
        return stacked
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


def _demo_count(cfg: SequenceConfig, rng: np.random.Generator) -> int:
    if isinstance(cfg.demos_per_task, int):
        return cfg.demos_per_task
    lo, hi = cfg.demos_per_task
    return int(rng.integers(lo, hi + 1))


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
        curriculum_latents = _sample_curriculum_latents(family, cfg, rng)
    else:
        if fixed_final_latent is None or world is None:
            raise ValueError("history=False requires fixed_final_latent and world")
        curriculum_latents = np.zeros((0, family.cfg.num_modules), dtype=np.float32)

    latents = [lat for lat in curriculum_latents]
    if final_task is not None:
        if fixed_final_latent is not None:
            latents.append(fixed_final_latent.astype(np.float32))
        else:
            latents.append(_sample_final_latent(family, final_task, curriculum_latents, rng))
    elif fixed_final_latent is not None:
        latents.append(fixed_final_latent.astype(np.float32))
    if revisit_demos > 0:
        if len(curriculum_latents) == 0:
            raise ValueError("revisit requires curriculum history")
        latents.append(curriculum_latents[0])

    demo_counts = []
    for i in range(len(latents)):
        is_revisit = revisit_demos > 0 and i == len(latents) - 1
        if is_revisit:
            demo_counts.append(revisit_demos)
        elif final_task is not None and i == len(curriculum_latents):
            demo_counts.append(final_task.num_demos)
        else:
            demo_counts.append(_demo_count(cfg, rng))

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

    info: dict[str, Any] = {
        "latents": np.stack(latents),
        "demo_counts": np.array(demo_counts, dtype=np.int64),
        "boundaries": np.array(boundaries, dtype=np.int64),
        "task_spans": np.array(task_spans, dtype=np.int64),
        "base_mse": np.stack(base_mse),
        "num_curriculum_tasks": len(curriculum_latents),
    }
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
