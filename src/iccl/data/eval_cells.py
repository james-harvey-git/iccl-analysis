"""Resolve the fixed-demo structural cells used by capability evaluation."""

from dataclasses import dataclass

from omegaconf import DictConfig

from iccl.data.dataset import module_count_config_from

EVAL_FAMILIES = ("canonical", "task_variation", "module_variation")
_EVAL_KEYS = {
    "num_sequences",
    "demos_per_task",
    "out_dir",
    "bootstrap_seed",
    "bootstrap_replicates",
    "best_metric",
    "capabilities",
    "canonical",
    "module_counts",
    "task_variation",
    "composition",
    "retention",
}


@dataclass(frozen=True)
class EvalCell:
    """One physical ``(M,T,D)`` cell with all logical family memberships."""

    family_memberships: tuple[str, ...]
    status: str
    num_modules: int
    num_tasks: int
    demos_per_task: int

    @property
    def num_surplus(self) -> int:
        return self.num_tasks - (self.num_modules - 1)

    @property
    def prediction_tokens(self) -> int:
        return self.num_tasks * self.demos_per_task

    @property
    def serialized_tokens(self) -> int:
        return 2 * self.prediction_tokens + self.num_tasks

    @property
    def demo_counts(self) -> tuple[int, ...]:
        return (self.demos_per_task,) * self.num_tasks

    @property
    def cell_id(self) -> str:
        return f"m{self.num_modules:02d}__t{self.num_tasks:02d}__d{self.demos_per_task:03d}"


def _inclusive_bounds(config: DictConfig, name: str, minimum: int) -> range:
    lo, hi = int(config.min), int(config.max)
    if lo < minimum or hi < lo:
        raise ValueError(f"{name} requires {minimum} <= min <= max, got [{lo}, {hi}]")
    return range(lo, hi + 1)


def evaluation_module_counts(data_cfg: DictConfig) -> tuple[tuple[int, ...], dict[int, str]]:
    """Return the inclusive evaluation range and each count's training status."""
    values = tuple(_inclusive_bounds(data_cfg.eval_sets.module_counts, "module_counts", 2))
    training = module_count_config_from(data_cfg)
    allowed, held_out = set(training.allowed), set(training.held_out)
    statuses = {
        value: "seen" if value in allowed else "heldout" if value in held_out else "ood"
        for value in values
    }
    return values, statuses


def resolve_eval_cells(data_cfg: DictConfig) -> list[EvalCell]:
    """Deduplicate canonical, task-variation and module-variation cells."""
    config = data_cfg.eval_sets
    unknown = set(config.keys()) - _EVAL_KEYS
    if unknown:
        raise ValueError(f"unknown eval_sets fields: {sorted(unknown)}")

    demos = int(config.demos_per_task)
    if demos < 1:
        raise ValueError(f"eval_sets.demos_per_task must be positive, got {demos}")
    modules, statuses = evaluation_module_counts(data_cfg)
    canonical_m = int(config.canonical.module_count)
    canonical_t = int(config.canonical.task_count)
    if canonical_m not in set(modules):
        raise ValueError(
            f"canonical M={canonical_m} is outside evaluation range [{modules[0]}, {modules[-1]}]"
        )
    if canonical_t < canonical_m - 1:
        raise ValueError(f"canonical cell cannot cover M={canonical_m} with T={canonical_t}")

    surplus = _inclusive_bounds(
        config.task_variation.surplus_tasks, "task_variation.surplus_tasks", 0
    )
    memberships: dict[tuple[int, int, int], set[str]] = {}

    def add(family: str, num_modules: int, num_tasks: int) -> None:
        if num_tasks < num_modules - 1:
            raise ValueError(f"{family} cannot cover M={num_modules} with T={num_tasks}")
        memberships.setdefault((num_modules, num_tasks, demos), set()).add(family)

    add("canonical", canonical_m, canonical_t)
    for num_modules in modules:
        for extra in surplus:
            add("task_variation", num_modules, num_modules - 1 + extra)
        add("module_variation", num_modules, modules[-1])

    return [
        EvalCell(
            tuple(family for family in EVAL_FAMILIES if family in families),
            statuses[num_modules],
            num_modules,
            num_tasks,
            demos_per_task,
        )
        for (num_modules, num_tasks, demos_per_task), families in sorted(memberships.items())
    ]
