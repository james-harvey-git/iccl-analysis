"""Fixed teacher worlds and independently indexed task-vector probe trajectories."""

import hashlib
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np

from iccl.data.curriculum import PhaseConfig, SequenceConfig, check_connected
from iccl.data.sequences import SequenceSample, build_sequence
from iccl.data.teacher import (
    HyperTeacher,
    ModulePool,
    TeacherConfig,
    sample_module_pool,
    teacher_forward,
)

PROTOCOL = "fixed-world-task-vector-v1"
SPLITS = ("train", "validation", "test_familiar", "test_unseen")
CAPTURES = ("final_input", "final_answer", "branch_input")


def stream_seed(seed: int, namespace: str) -> int:
    """Stable independent streams, including distinct development/confirmation worlds."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int.from_bytes(hashlib.sha256(f"{PROTOCOL}/{seed}/{namespace}".encode()).digest()[:8])


@dataclass
class FixedWorld:
    family: HyperTeacher
    pool: ModulePool
    pairs: dict[str, np.ndarray]
    seed: int
    world_id: int
    phase: str

    def rng(self, namespace: str) -> np.random.Generator:
        return np.random.default_rng(stream_seed(self.seed, namespace))

    def metadata(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "phase": self.phase,
            "seed": self.seed,
            "teacher": asdict(self.family.cfg),
            "pairs": {key: value.tolist() for key, value in self.pairs.items()},
        }


def make_world(
    teacher: TeacherConfig,
    *,
    seed: int,
    phase: str,
    world_id: int,
    pair_counts: dict[str, int],
) -> FixedWorld:
    """Partition supports with coverage and a connected, target-excludable train pool."""
    modules = teacher.num_modules
    all_pairs = np.asarray(list(combinations(range(modules), 2)), dtype=np.int64)
    names = ("train", "validation", "test_unseen")
    if set(pair_counts) != set(names) or sum(pair_counts.values()) != len(all_pairs):
        raise ValueError("pair partitions must exhaust all distinct two-module supports")
    if any(pair_counts[key] < (modules + 1) // 2 for key in names):
        raise ValueError("every support partition must have enough pairs to cover every module")
    if pair_counts["train"] < modules:
        raise ValueError("training pairs need degree >= 2 for target-excluded background coverage")
    world_seed = stream_seed(seed, f"{phase}/world/{world_id}")
    family = HyperTeacher(teacher, 2)
    pool = sample_module_pool(teacher, np.random.default_rng(stream_seed(world_seed, "pool")))
    rng = np.random.default_rng(stream_seed(world_seed, "partitions"))
    for _ in range(10000):
        shuffled = all_pairs[rng.permutation(len(all_pairs))]
        offset, pairs = 0, {}
        for name in names:
            pairs[name] = shuffled[offset : offset + pair_counts[name]].copy()
            offset += pair_counts[name]
        if any(len(np.unique(value)) != modules for value in pairs.values()):
            continue
        train = np.zeros((len(pairs["train"]), modules), dtype=np.int8)
        np.put_along_axis(train, pairs["train"], 1, axis=1)
        if train.sum(axis=0).min() >= 2 and check_connected(train):
            return FixedWorld(family, pool, pairs, world_seed, world_id, phase)
    raise ValueError("could not sample covered, connected support partitions; revise pair counts")


@dataclass
class ProbeTrial:
    sample: SequenceSample
    target: np.ndarray
    support: np.ndarray
    prior_exposures: np.ndarray
    branch_input: np.ndarray
    branch_answer: np.ndarray
    task_start: int
    answer_stop: int
    index: int

    def observations(self, capture: str, input_dim: int, output_dim: int) -> np.ndarray:
        tokens = self.sample.tokens[self.task_start : self.answer_stop]
        x, y = tokens[::2, :input_dim], tokens[1::2, :output_dim]
        if capture == "final_input":
            return np.concatenate((np.concatenate((x[:-1], y[:-1]), axis=1).ravel(), x[-1]))
        pairs = np.concatenate((x, y), axis=1).ravel()
        if capture == "final_answer":
            return pairs
        if capture == "branch_input":
            return np.concatenate((pairs, self.branch_input.ravel()))
        raise ValueError(f"unknown capture: {capture}")


def make_trial(
    world: FixedWorld,
    *,
    split: str,
    position: int,
    index: int,
    tasks: int,
    demos: int,
    signal_boundaries: bool = True,
) -> ProbeTrial:
    """Fresh observations/history per trial; balanced target supports in shuffled cycles."""
    if split not in SPLITS or not 1 <= position <= tasks or index < 0 or demos < 2:
        raise ValueError("invalid split, target position, trial index or demonstration count")
    modules = world.family.cfg.num_modules
    if tasks - 1 < (modules + 1) // 2:
        raise ValueError("background schedule is too short to cover the module pool")
    namespace = f"{split}/position/{position}"
    pairs = world.pairs["train" if split == "test_familiar" else split]
    cycle, slot = divmod(index, len(pairs))
    order = world.rng(f"{namespace}/target-cycle/{cycle}").permutation(len(pairs))
    target_pair = pairs[order[slot]]
    rng = world.rng(f"{namespace}/trial/{index}/tasks")
    allowed = world.pairs["train"]
    allowed = allowed[~np.all(allowed == target_pair, axis=1)]
    for _ in range(10000):
        background = allowed[rng.integers(len(allowed), size=tasks - 1)]
        if len(np.unique(background)) == modules:
            break
    else:
        raise ValueError("background coverage rejection budget exhausted")
    background = background[rng.permutation(tasks - 1)]
    supports = np.insert(background, position - 1, target_pair, axis=0)
    patterns = np.zeros((tasks, modules), dtype=np.int8)
    np.put_along_axis(patterns, supports, 1, axis=1)
    latents = np.stack([world.family.apply_weighting(rng, pattern) for pattern in patterns])
    config = SequenceConfig(
        phases=(PhaseConfig(tasks, (2, 2)),),
        demos_per_task=demos,
        signal_boundaries=signal_boundaries,
        require_identifiable=False,
    )
    sample = build_sequence(
        world.family,
        config,
        world.rng(f"{namespace}/trial/{index}/observations"),
        world=world.pool,
        fixed_curriculum_latents=latents,
        fixed_demo_counts=(demos,) * tasks,
    )
    start, stop = map(int, sample.info["task_spans"][position - 1])
    branch_rng = world.rng(f"{namespace}/trial/{index}/branch")
    query = branch_rng.uniform(-1, 1, size=(1, world.family.cfg.input_dim)).astype(np.float32)
    target = latents[position - 1]
    exposures = (latents[: position - 1, target_pair] != 0).sum(axis=0).astype(np.int64)
    return ProbeTrial(
        sample,
        target,
        target_pair.copy(),
        exposures,
        query,
        teacher_forward(world.pool, target, query),
        start,
        stop,
        index,
    )
