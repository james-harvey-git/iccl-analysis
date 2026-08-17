"""Resolve named evaluation slices into homogeneous structural cells."""

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from iccl.data.dataset import module_count_config_from


@dataclass(frozen=True)
class EvalCell:
    slice: str
    status: str
    num_modules: int
    num_tasks: int
    demo_counts: tuple[int, ...]
    variant: str = ""

    @property
    def num_surplus(self) -> int:
        return self.num_tasks - (self.num_modules - 1)

    @property
    def prediction_tokens(self) -> int:
        return sum(self.demo_counts)

    @property
    def serialized_tokens(self) -> int:
        return 2 * self.prediction_tokens + self.num_tasks


def evaluation_module_counts(data_cfg: DictConfig) -> tuple[tuple[int, ...], dict[int, str]]:
    """Resolve selected seen, held-out interpolation, and OOD counts."""
    train = module_count_config_from(data_cfg)
    config = data_cfg.eval_sets.module_counts
    if config.seen_selection != "endpoints_and_canonical":
        raise ValueError(f"unknown seen_selection: {config.seen_selection!r}")
    canonical = int(config.canonical_seen)
    if canonical not in train.allowed:
        raise ValueError(f"canonical_seen={canonical} is absent from {train.allowed}")
    seen = tuple(sorted({train.min, canonical, train.max}))
    ood = tuple(int(value) for value in config.get("ood", []))
    if len(set(ood)) != len(ood) or any(value <= train.max for value in ood):
        raise ValueError(f"OOD counts must be unique and greater than M_max={train.max}: {ood}")
    groups = {"seen": seen, "heldout": train.held_out, "ood": ood}
    values = [value for group in groups.values() for value in group]
    if len(values) != len(set(values)):
        raise ValueError("seen, held-out, and OOD module counts must be disjoint")
    status = {value: name for name, group in groups.items() for value in group}
    return tuple(sorted(status)), status


def even_demo_counts(total: int, tasks: int, seed: int) -> tuple[int, ...]:
    """Allocate a prediction budget as evenly as possible across tasks."""
    if tasks < 1 or total < tasks:
        raise ValueError(f"prediction budget needs B>=T>=1, got B={total}, T={tasks}")
    quotient, remainder = divmod(total, tasks)
    counts = np.full(tasks, quotient, dtype=np.int64)
    if remainder:
        rng = np.random.Generator(np.random.Philox(key=np.array([seed, tasks], dtype=np.uint64)))
        counts[rng.choice(tasks, size=remainder, replace=False)] += 1
    return tuple(int(value) for value in counts)


def _cell(
    name: str,
    status: str,
    modules: int,
    tasks: int,
    demos: tuple[int, ...],
    variant: str = "",
) -> EvalCell:
    cell = EvalCell(name, status, modules, tasks, demos, variant)
    if cell.num_surplus < 0:
        raise ValueError(f"slice={name} cannot cover M={modules} with T={tasks}")
    if len(demos) != tasks or min(demos) < 1:
        raise ValueError(f"slice={name} needs {tasks} positive demo counts, got {demos}")
    return cell


def resolve_eval_cells(data_cfg: DictConfig, seed: int) -> list[EvalCell]:
    """Expand every enabled one-dimensional structural slice."""
    module_values, statuses = evaluation_module_counts(data_cfg)
    slices = data_cfg.eval_sets.structural_slices
    cells: list[EvalCell] = []
    for name in slices.enabled:
        config = slices[name]
        if name in {"fixed_surplus", "matched_task_count", "matched_prediction_tokens"}:
            for modules in module_values:
                tasks = (
                    int(config.task_count)
                    if name == "matched_task_count"
                    else modules - 1 + int(config.surplus_tasks)
                )
                demos = (
                    even_demo_counts(int(config.prediction_tokens), tasks, seed + modules)
                    if name == "matched_prediction_tokens"
                    else (int(config.history_demos_per_task),) * tasks
                )
                cells.append(_cell(name, statuses[modules], modules, tasks, demos))
            continue

        modules = int(config.module_count)
        if modules not in statuses:
            raise ValueError(f"slice={name} uses M={modules}, which is not evaluated")
        status = statuses[modules]
        if name == "task_count":
            demos = int(config.history_demos_per_task)
            cells.extend(
                _cell(name, status, modules, int(tasks), (demos,) * int(tasks))
                for tasks in config["values"]
            )
        elif name == "history_demos":
            tasks = int(config.task_count)
            cells.extend(
                _cell(
                    name,
                    status,
                    modules,
                    tasks,
                    (int(demos),) * tasks,
                    f"d{int(demos):03d}",
                )
                for demos in config["values"]
            )
        elif name == "matched_serialized_prefix":
            length = int(config.serialized_tokens)
            for tasks_raw in config.task_counts:
                tasks = int(tasks_raw)
                remainder = length - tasks
                if remainder < 0 or remainder % 2:
                    raise ValueError(
                        f"matched prefix needs non-negative even L-T: L={length}, T={tasks}"
                    )
                demos = even_demo_counts(remainder // 2, tasks, seed + tasks)
                cells.append(_cell(name, status, modules, tasks, demos))
        elif name == "demo_allocation":
            tasks, budget = int(config.task_count), int(config.prediction_tokens)
            uniform = even_demo_counts(budget, tasks, seed)
            if len(set(uniform)) != 1 or tasks % 2:
                raise ValueError(
                    f"demo allocations need even T and integer B/T: T={tasks}, B={budget}"
                )
            mean = uniform[0]
            low, high = mean // 2, mean + (mean - mean // 2)
            front = (high,) * (tasks // 2) + (low,) * (tasks // 2)
            patterns = {
                "uniform": uniform,
                "alternating": tuple(low if index % 2 == 0 else high for index in range(tasks)),
                "front_loaded": front,
                "back_loaded": tuple(reversed(front)),
            }
            for pattern in config.patterns:
                if pattern not in patterns:
                    raise ValueError(f"unknown demo-allocation pattern: {pattern}")
                cells.append(_cell(name, status, modules, tasks, patterns[pattern], str(pattern)))
        else:
            raise ValueError(f"unknown structural slice: {name}")
    return cells
