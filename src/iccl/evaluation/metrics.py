"""Fixed-demo ICL, composition and retention metrics over frozen suites."""

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from jaxtyping import Float

from iccl.data.export import VALIDATION_SUITE, load_suite, load_suite_metadata
from iccl.models.model import GDNModel

Suite = dict[str, Any]
BASE_MSE_FLOOR = 1e-12
METRIC_VERSION = "fixed-d-capabilities-v2"
METRIC_DEFINITIONS = {
    "nmse": "per-demo output MSE divided by that task's mean output variance",
    "validation/token_mse": "raw output MSE pooled over training-distribution prediction tokens",
    "within_task_nmse_mean": "mean nMSE over evaluation sequences, history tasks, and demos",
    "episode_learning": "mean nMSE over sequences and demos at each history task position",
    "benefit_mean": (
        "mean over final-task demos of matched-prefix nMSE minus constituent-history nMSE"
    ),
    "savings_mean": "equal-delay mean over demos of novel nMSE minus exact-repeat nMSE",
    "episodic_savings_mean": (
        "equal-delay mean over demos of shared-support nMSE minus exact-repeat nMSE"
    ),
    "module_savings_mean": "equal-delay mean over demos of novel nMSE minus shared-support nMSE",
}


@dataclass
class EvaluationReport:
    scalars: dict[str, float]
    curves: dict[str, np.ndarray]
    summary_rows: list[dict[str, Any]]
    curve_rows: list[dict[str, Any]]
    raw_errors: dict[str, np.ndarray]


def load_eval_suites(
    out_dir: Path,
    *,
    select: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Suite]:
    """Load selected frozen suites and validate paired-condition identifiers."""
    paths = sorted(out_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"no frozen eval suites found in {out_dir}; run scripts/make_eval_sets.py"
        )
    suites: dict[str, Suite] = {}
    for path in paths:
        metadata_path = path.with_suffix(".meta.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"frozen suite metadata not found: {metadata_path}")
        metadata = load_suite_metadata(metadata_path)
        if select is not None and not select(metadata):
            continue
        suite: Suite = load_suite(path.with_suffix(""))
        suite["__meta__"] = metadata
        suites[path.stem] = suite
    if not suites:
        qualifier = " matching the requested selection" if select is not None else ""
        raise FileNotFoundError(f"no frozen eval suites{qualifier} found in {out_dir}")

    groups: dict[tuple[str, str], dict[str, Suite]] = {}
    for suite in suites.values():
        metadata = suite["__meta__"]
        if pair_group := metadata.get("pair_group"):
            groups.setdefault((metadata["capability"], pair_group), {})[metadata["condition"]] = (
                suite
            )
    required = {
        "composition": {"constituent", "matched_prefix"},
        "retention": {"repeat", "novel"},
    }
    for (capability, pair_group), conditions in groups.items():
        missing = required[capability] - set(conditions)
        if missing:
            raise FileNotFoundError(f"{pair_group} is missing paired conditions {sorted(missing)}")
        pair_ids = [suite.get("pair_id") for suite in conditions.values()]
        if any(ids is None for ids in pair_ids):
            raise ValueError(f"{pair_group} conditions do not share pair identifiers")
        arrays = [cast(np.ndarray, ids) for ids in pair_ids]
        if any(not np.array_equal(arrays[0], ids) for ids in arrays[1:]):
            raise ValueError(f"{pair_group} conditions do not share pair identifiers")
    return suites


@torch.no_grad()
def predict_suite(
    model: GDNModel,
    suite: Suite,
    device: torch.device,
    *,
    batch_size: int = 32,
    autocast_dtype: torch.dtype | None = None,
) -> Float[np.ndarray, "seqs seq_len d_out"]:
    """Run a model over one suite and return fp32 CPU predictions."""
    tokens = torch.from_numpy(suite["tokens"])
    token_type = torch.from_numpy(suite["token_type"])
    predictions = []
    for start in range(0, len(tokens), batch_size):
        context = (
            torch.autocast(device.type, autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with context:
            output = model(
                tokens[start : start + batch_size].to(device),
                token_type[start : start + batch_size].to(device),
            )
        predictions.append(output.preds.float().cpu())
    return torch.cat(predictions).numpy()


def demo_mse(
    preds: Float[np.ndarray, "seqs seq_len d_out"], suite: Suite
) -> Float[np.ndarray, "seqs tasks max_demos"]:
    """Raw MSE per sequence, task and demo, NaN-padded by demo index."""
    squared_error = ((preds - suite["targets"]) ** 2).mean(axis=-1)
    counts, spans = suite["demo_counts"], suite["task_spans"]
    values = np.full((*counts.shape, int(counts.max())), np.nan)
    for sequence, task in np.ndindex(counts.shape):
        count = int(counts[sequence, task])
        positions = spans[sequence, task, 0] + 2 * np.arange(count)
        values[sequence, task, :count] = squared_error[sequence, positions]
    return values


def demo_nmse(
    preds: Float[np.ndarray, "seqs seq_len d_out"], suite: Suite
) -> Float[np.ndarray, "seqs tasks max_demos"]:
    """MSE divided by each task's output variance."""
    denominator = np.maximum(suite["base_mse"].mean(axis=-1), BASE_MSE_FLOOR)
    return demo_mse(preds, suite) / denominator[:, :, None]


def token_mse(preds: np.ndarray, suite: Suite) -> float:
    """Raw MSE pooled over every prediction-bearing token in a suite."""
    mask = np.asarray(suite["loss_mask"], dtype=np.float64)
    if (denominator := float(mask.sum())) <= 0:
        raise ValueError("validation suite contains no prediction-bearing tokens")
    squared_error = ((preds - suite["targets"]) ** 2).mean(axis=-1)
    return float((squared_error * mask).sum() / denominator)


def _aggregate(
    values: np.ndarray,
    *,
    seed: int,
    replicates: int,
    strata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and deterministic bootstrap interval, optionally equal over strata."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or not len(values):
        raise ValueError("aggregation requires a non-empty sequence axis")
    rng = np.random.Generator(np.random.Philox(key=np.array([seed, len(values)], dtype=np.uint64)))
    if strata is None:
        mean = np.nanmean(values, axis=0)
        if replicates < 2 or len(values) == 1:
            return mean, mean, mean
        indices = rng.integers(0, len(values), size=(replicates, len(values)))
        draws = np.nanmean(values[indices], axis=1)
    else:
        strata = np.asarray(strata)
        if strata.shape != (len(values),):
            raise ValueError(f"strata must have shape {(len(values),)}, got {strata.shape}")
        groups = [values[strata == value] for value in np.unique(strata)]
        mean = np.mean([np.nanmean(group, axis=0) for group in groups], axis=0)
        if replicates < 2 or all(len(group) == 1 for group in groups):
            return mean, mean, mean
        draws = np.zeros((replicates, *values.shape[1:]), dtype=np.float64)
        for group in groups:
            indices = rng.integers(0, len(group), size=(replicates, len(group)))
            draws += np.nanmean(group[indices], axis=1) / len(groups)
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    return mean, low, high


def _descriptor(name: str, suite: Suite) -> dict[str, Any]:
    metadata = suite.get("__meta__")
    if not isinstance(metadata, dict) or "capability" not in metadata:
        raise ValueError(f"suite {name} has no capability metadata")
    return {
        "suite": name,
        "cell_id": metadata["cell_id"],
        "family_memberships": "|".join(metadata["family_memberships"]),
        "capability": metadata["capability"],
        "module_count_status": metadata["module_count_status"],
        "sampler": metadata["config"]["sequence"].get("curriculum_sampler", "rejection"),
        "weighting": metadata["config"]["weighting"],
        "M": int(metadata["num_modules"]),
        "T": int(metadata["num_tasks"]),
        "S": int(metadata["num_surplus_tasks"]),
        "D": int(metadata["demos_per_task"]),
        "pair_group": metadata.get("pair_group"),
    }


def _base(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in descriptor.items() if key != "pair_group"}


class _ReportBuilder:
    def __init__(self, seed: int, replicates: int) -> None:
        self.seed = seed
        self.replicates = replicates
        self.scalars: dict[str, float] = {}
        self.curves: dict[str, np.ndarray] = {}
        self.summary_rows: list[dict[str, Any]] = []
        self.curve_rows: list[dict[str, Any]] = []

    def summary(
        self,
        descriptor: dict[str, Any],
        condition: str,
        metric: str,
        values: np.ndarray,
        *,
        seed: int,
        strata: np.ndarray | None = None,
        component: str | None = None,
    ) -> None:
        mean, low, high = _aggregate(
            values, seed=self.seed + seed, replicates=self.replicates, strata=strata
        )
        scalar = float(mean)
        key = f"{descriptor['capability']}/{descriptor['cell_id']}/{metric}"
        self.scalars[key] = scalar
        self.summary_rows.append(
            dict(
                _base(descriptor),
                condition=condition,
                metric=metric,
                retention_component=component,
                original_task_position=None,
                intervening_tasks=None,
                value=scalar,
                ci_low=float(low),
                ci_high=float(high),
                n_sequences=len(values),
            )
        )

    def curve(
        self,
        descriptor: dict[str, Any],
        condition: str,
        curve_type: str,
        mse: np.ndarray,
        nmse: np.ndarray,
        *,
        seed: int,
        x_name: str,
        x_values: np.ndarray | None = None,
        strata: np.ndarray | None = None,
        component: str | None = None,
    ) -> None:
        mse_mean, _, _ = _aggregate(mse, seed=self.seed + seed, replicates=0, strata=strata)
        mean, low, high = _aggregate(
            nmse, seed=self.seed + seed + 1, replicates=self.replicates, strata=strata
        )
        suffix = f"/{component}" if component else ""
        key = f"{descriptor['capability']}/{descriptor['cell_id']}/{curve_type}/{condition}{suffix}"
        self.curves[key] = mean
        coordinates = np.arange(nmse.shape[1]) if x_values is None else x_values
        for index, coordinate in enumerate(coordinates):
            self.curve_rows.append(
                dict(
                    _base(descriptor),
                    condition=condition,
                    curve_type=curve_type,
                    retention_component=component,
                    original_task_position=None,
                    intervening_tasks=(int(coordinate) if x_name == "intervening_tasks" else None),
                    x_name=x_name,
                    x_value=int(coordinate),
                    mse=float(mse_mean[index]),
                    nmse=float(mean[index]),
                    ci_low=float(low[index]),
                    ci_high=float(high[index]),
                    n_sequences=len(nmse),
                )
            )

    def delay_curve(
        self,
        descriptor: dict[str, Any],
        condition: str,
        mse: np.ndarray,
        nmse: np.ndarray,
        strata: np.ndarray,
        *,
        seed: int,
        component: str,
    ) -> None:
        values = np.unique(strata)
        mse_means, means, lows, highs, counts = [], [], [], [], []
        for offset, value in enumerate(values):
            selected = strata == value
            mse_mean, _, _ = _aggregate(
                mse[selected].mean(axis=1), seed=self.seed + seed + offset, replicates=0
            )
            mean, low, high = _aggregate(
                nmse[selected].mean(axis=1),
                seed=self.seed + seed + 100 + offset,
                replicates=self.replicates,
            )
            mse_means.append(float(mse_mean))
            means.append(float(mean))
            lows.append(float(low))
            highs.append(float(high))
            counts.append(int(selected.sum()))
        key = f"{descriptor['capability']}/{descriptor['cell_id']}/retention_delay/{component}"
        self.curves[key] = np.asarray(means)
        for index, value in enumerate(values):
            self.curve_rows.append(
                dict(
                    _base(descriptor),
                    condition=condition,
                    curve_type="retention_delay",
                    retention_component=component,
                    original_task_position=None,
                    intervening_tasks=int(value),
                    x_name="intervening_tasks",
                    x_value=int(value),
                    mse=mse_means[index],
                    nmse=means[index],
                    ci_low=lows[index],
                    ci_high=highs[index],
                    n_sequences=counts[index],
                )
            )

    def build(self, raw_errors: dict[str, np.ndarray]) -> EvaluationReport:
        return EvaluationReport(
            self.scalars, self.curves, self.summary_rows, self.curve_rows, raw_errors
        )


def _last_task(errors: np.ndarray, demos: int) -> np.ndarray:
    return errors[:, -1, :demos]


def _evaluate(
    suites: dict[str, Suite],
    mses: dict[str, np.ndarray],
    nmses: dict[str, np.ndarray],
    raw_errors: dict[str, np.ndarray],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> EvaluationReport:
    report = _ReportBuilder(bootstrap_seed, bootstrap_replicates)
    descriptors = {name: _descriptor(name, suite) for name, suite in suites.items()}
    groups: dict[tuple[str, str], dict[str, str]] = {}

    for index, (name, descriptor) in enumerate(sorted(descriptors.items())):
        if descriptor["capability"] != "icl":
            groups.setdefault((descriptor["capability"], descriptor["pair_group"]), {})[
                suites[name]["__meta__"]["condition"]
            ] = name
            continue
        demos, tasks = descriptor["D"], descriptor["T"]
        mse, nmse = mses[name][:, :tasks, :demos], nmses[name][:, :tasks, :demos]
        within_mse, within_nmse = mse.mean(axis=1), nmse.mean(axis=1)
        report.summary(
            descriptor,
            "ordinary",
            "within_task_nmse_mean",
            nmse.mean(axis=(1, 2)),
            seed=index * 100,
        )
        report.curve(
            descriptor,
            "ordinary",
            "within_task_learning",
            within_mse,
            within_nmse,
            seed=index * 100 + 10,
            x_name="demo_index",
        )
        report.curve(
            descriptor,
            "ordinary",
            "episode_learning",
            mse.mean(axis=2),
            nmse.mean(axis=2),
            seed=index * 100 + 20,
            x_name="task_position",
        )

    for index, ((capability, _), conditions) in enumerate(sorted(groups.items())):
        primary = "constituent" if capability == "composition" else "repeat"
        descriptor = descriptors[conditions[primary]]
        demos = descriptor["D"]
        values = {
            condition: (
                suites[name],
                _last_task(mses[name], demos),
                _last_task(nmses[name], demos),
            )
            for condition, name in conditions.items()
        }
        seed = 100_000 + index * 1000
        if capability == "composition":
            for offset, condition in enumerate(("constituent", "matched_prefix", "no_history")):
                if condition not in values:
                    continue
                suite, mse, nmse = values[condition]
                report.curve(
                    descriptors[conditions[condition]],
                    condition,
                    "composition_learning",
                    mse,
                    nmse,
                    seed=seed + offset * 10,
                    x_name="demo_index",
                )
            constituent, matched = values["constituent"], values["matched_prefix"]
            benefit_mse = matched[1] - constituent[1]
            benefit_nmse = matched[2] - constituent[2]
            report.summary(
                descriptor,
                "benefit",
                "benefit_mean",
                benefit_nmse.mean(axis=1),
                seed=seed + 100,
            )
            report.curve(
                descriptor,
                "benefit",
                "composition_benefit",
                benefit_mse,
                benefit_nmse,
                seed=seed + 110,
                x_name="demo_index",
            )
            continue

        repeat_suite, repeat_mse, repeat_nmse = values["repeat"]
        positions = np.asarray(repeat_suite["original_task_position"], dtype=np.int64)
        strata = np.asarray(repeat_suite["intervening_tasks"], dtype=np.int64)
        rows = np.arange(len(positions))
        original_mse = mses[conditions["repeat"]][rows, positions, :demos]
        original_nmse = nmses[conditions["repeat"]][rows, positions, :demos]
        retention_curves = {"original": (original_mse, original_nmse)} | {
            condition: (pair[1], pair[2]) for condition, pair in values.items()
        }
        for offset, (condition, (mse, nmse)) in enumerate(retention_curves.items()):
            report.curve(
                descriptor,
                condition,
                "retention_learning",
                mse,
                nmse,
                seed=seed + offset * 10,
                x_name="demo_index",
                strata=strata,
            )

        novel = values["novel"]
        components = {
            "total": (novel[1] - repeat_mse, novel[2] - repeat_nmse),
        }
        if "shared" in values:
            shared = values["shared"]
            components |= {
                "episodic": (shared[1] - repeat_mse, shared[2] - repeat_nmse),
                "module": (novel[1] - shared[1], novel[2] - shared[2]),
            }
        for offset, (component, (mse, nmse)) in enumerate(components.items()):
            condition = "savings" if component == "total" else f"{component}_savings"
            metric = f"{condition}_mean"
            report.summary(
                descriptor,
                condition,
                metric,
                nmse.mean(axis=1),
                seed=seed + 200 + offset * 10,
                strata=strata,
                component=component,
            )
            report.curve(
                descriptor,
                condition,
                "retention_savings",
                mse,
                nmse,
                seed=seed + 300 + offset * 10,
                x_name="demo_index",
                strata=strata,
                component=component,
            )
            report.delay_curve(
                descriptor,
                condition,
                mse,
                nmse,
                strata,
                seed=seed + 400 + offset * 20,
                component=component,
            )
            prefix = f"retention/{descriptor['cell_id']}/{component}"
            raw_errors[f"{prefix}_mse"] = mse
            raw_errors[f"{prefix}_nmse"] = nmse
        prefix = f"retention/{descriptor['cell_id']}"
        raw_errors[f"{prefix}/original_task_position"] = positions
        raw_errors[f"{prefix}/intervening_tasks"] = strata

    return report.build(raw_errors)


def evaluate_suites(
    model: GDNModel,
    suites: dict[str, Suite],
    device: torch.device,
    *,
    batch_size: int = 32,
    autocast_dtype: torch.dtype | None = None,
    bootstrap_seed: int = 0,
    bootstrap_replicates: int = 1000,
) -> EvaluationReport:
    """Evaluate the training objective and all configured capability suites."""
    capability_suites: dict[str, Suite] = {}
    mses: dict[str, np.ndarray] = {}
    nmses: dict[str, np.ndarray] = {}
    validation_metric: float | None = None
    for name, suite in suites.items():
        predictions = predict_suite(
            model,
            suite,
            device,
            batch_size=batch_size,
            autocast_dtype=autocast_dtype,
        )
        if name == VALIDATION_SUITE:
            validation_metric = token_mse(predictions, suite)
            continue
        capability_suites[name] = suite
        mses[name] = demo_mse(predictions, suite)
        nmses[name] = demo_nmse(predictions, suite)
    raw_errors = {
        f"{name}/{kind}": values
        for name in capability_suites
        for kind, values in (("mse", mses[name]), ("nmse", nmses[name]))
    }
    report = _evaluate(
        capability_suites,
        mses,
        nmses,
        raw_errors,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    if validation_metric is not None:
        report.scalars[f"{VALIDATION_SUITE}/token_mse"] = validation_metric
    return report
