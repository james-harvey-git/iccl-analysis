"""Capability metrics over homogeneous frozen evaluation cells."""

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from jaxtyping import Float

from iccl.data.export import VALIDATION_SUITE, load_suite, load_suite_metadata
from iccl.models.model import GDNModel

Suite = dict[str, Any]
BASE_MSE_FLOOR = 1e-12
THRESHOLD_NMSE = 0.5


@dataclass
class EvaluationReport:
    scalars: dict[str, float]
    curves: dict[str, np.ndarray]
    summary_rows: list[dict[str, Any]]
    curve_rows: list[dict[str, Any]]


def load_eval_suites(out_dir: Path) -> dict[str, Suite]:
    """Load and validate every frozen suite in a directory."""
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
        suite: Suite = load_suite(path.with_suffix(""))
        suite["__meta__"] = load_suite_metadata(metadata_path)
        suites[path.stem] = suite

    groups: dict[tuple[str, str], set[str]] = {}
    for suite in suites.values():
        metadata = suite["__meta__"]
        pair_group = metadata.get("pair_group")
        if pair_group:
            groups.setdefault((metadata["capability"], pair_group), set()).add(
                metadata["condition"]
            )
    required = {
        "composition": {"constituent", "matched_prefix"},
        "retention": {"repeat", "novel"},
    }
    for (capability, pair_group), conditions in groups.items():
        missing = required[capability] - conditions
        if missing:
            raise FileNotFoundError(f"{pair_group} is missing paired conditions {sorted(missing)}")
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
    """Raw MSE per sequence, task, and demo, NaN-padded by demo index."""
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
    denominator = float(mask.sum())
    if denominator <= 0:
        raise ValueError("validation suite contains no prediction-bearing tokens")
    squared_error = ((preds - suite["targets"]) ** 2).mean(axis=-1)
    return float((squared_error * mask).sum() / denominator)


def _last_task_values(errors: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return errors[:, -1, : int(counts[:, -1].max())]


def _last_values(errors: np.ndarray, counts: np.ndarray) -> np.ndarray:
    sequences, tasks = counts.shape
    return errors[np.arange(sequences)[:, None], np.arange(tasks)[None, :], counts - 1]


def _demos_to_threshold(curve: np.ndarray) -> float:
    reached = np.flatnonzero(curve <= THRESHOLD_NMSE)
    return float(reached[0]) if reached.size else float(len(curve))


def _thresholds(values: np.ndarray) -> np.ndarray:
    return np.array(
        [_demos_to_threshold(curve[np.isfinite(curve)]) for curve in values],
        dtype=np.float64,
    )


def _mean_ci(values: np.ndarray, *, seed: int, replicates: int) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return float("nan"), float("nan"), float("nan")
    mean = float(clean.mean())
    if len(clean) == 1 or replicates < 2:
        return mean, mean, mean
    rng = np.random.Generator(np.random.Philox(key=np.array([seed, len(clean)], dtype=np.uint64)))
    indices = rng.integers(0, len(clean), size=(replicates, len(clean)))
    low, high = np.quantile(clean[indices].mean(axis=1), [0.025, 0.975])
    return mean, float(low), float(high)


def _descriptor(name: str, suite: Suite) -> dict[str, Any]:
    metadata = suite.get("__meta__")
    if not isinstance(metadata, dict) or "capability" not in metadata:
        raise ValueError(f"suite {name} has no capability metadata")
    demos = np.asarray(metadata["demo_counts"], dtype=np.float64)
    return {
        "suite": name,
        "capability": metadata["capability"],
        "condition": metadata["condition"],
        "slice": metadata["structural_slice"],
        "variant": metadata.get("variant", ""),
        "status": metadata["module_count_status"],
        "M": int(metadata["num_modules"]),
        "T": int(metadata["num_tasks"]),
        "S": int(metadata["num_surplus_tasks"]),
        "B_history": int(metadata["history_prediction_tokens"]),
        "L_history": int(metadata["history_serialized_tokens"]),
        "D_min": float(demos.min()),
        "D_max": float(demos.max()),
        "D_mean": float(demos.mean()),
        "D_cv": float(demos.std() / demos.mean()),
        "sampler": metadata["config"]["sequence"].get("curriculum_sampler", "rejection"),
        "weighting": metadata["config"]["weighting"],
        "pair_group": metadata.get("pair_group"),
    }


def _row_base(descriptor: dict[str, Any], condition: str, suite: Suite) -> dict[str, Any]:
    row = {
        **{
            key: descriptor[key]
            for key in (
                "capability",
                "slice",
                "variant",
                "suite",
                "status",
                "sampler",
                "weighting",
                "M",
                "T",
                "S",
                "B_history",
                "L_history",
                "D_min",
                "D_max",
                "D_mean",
                "D_cv",
            )
        },
        "condition": condition,
        "D_target": int(suite["demo_counts"][0, -1]),
        "constituent_exposure_min": None,
        "constituent_exposure_max": None,
        "constituent_demo_exposure_min": None,
        "constituent_demo_exposure_max": None,
        "intervening_tasks": None,
        "prediction_token_delay": None,
        "serialized_token_delay": None,
    }
    for field, prefix in (
        ("constituent_task_exposures", "constituent_exposure"),
        ("constituent_demo_exposures", "constituent_demo_exposure"),
    ):
        if field in suite:
            row[f"{prefix}_min"] = int(suite[field].min())
            row[f"{prefix}_max"] = int(suite[field].max())
    for field in ("intervening_tasks", "prediction_token_delay", "serialized_token_delay"):
        if field in suite:
            row[field] = float(np.mean(suite[field]))
    return row


class _ReportBuilder:
    def __init__(self, bootstrap_seed: int, bootstrap_replicates: int) -> None:
        self.seed = bootstrap_seed
        self.replicates = bootstrap_replicates
        self.scalars: dict[str, float] = {}
        self.curves: dict[str, np.ndarray] = {}
        self.summary_rows: list[dict[str, Any]] = []
        self.curve_rows: list[dict[str, Any]] = []

    def summary(
        self,
        key: str,
        descriptor: dict[str, Any],
        suite: Suite,
        condition: str,
        metric: str,
        values: np.ndarray,
        seed: int,
    ) -> None:
        mean, low, high = _mean_ci(values, seed=self.seed + seed, replicates=self.replicates)
        self.scalars[f"{key}/{metric}"] = mean
        self.summary_rows.append(
            dict(
                _row_base(descriptor, condition, suite),
                metric=metric,
                value=mean,
                ci_low=low,
                ci_high=high,
                n_sequences=int(np.isfinite(values).sum()),
            )
        )

    def curve(
        self,
        key: str,
        descriptor: dict[str, Any],
        suite: Suite,
        condition: str,
        curve_type: str,
        mse: np.ndarray,
        nmse: np.ndarray,
        *,
        x_name: str = "demo_index",
        x_values: np.ndarray | None = None,
        seed: int,
    ) -> None:
        self.curves[f"{key}/{curve_type}"] = np.nanmean(nmse, axis=0)
        base = _row_base(descriptor, condition, suite)
        x_values = np.arange(nmse.shape[1]) if x_values is None else x_values
        for position, x_value in enumerate(x_values):
            mse_mean, _, _ = _mean_ci(
                mse[:, position], seed=self.seed + seed + position, replicates=self.replicates
            )
            mean, low, high = _mean_ci(
                nmse[:, position],
                seed=self.seed + seed + 10_000 + position,
                replicates=self.replicates,
            )
            self.curve_rows.append(
                dict(
                    base,
                    curve_type=curve_type,
                    x_name=x_name,
                    x_value=int(x_value),
                    mse=mse_mean,
                    nmse=mean,
                    ci_low=low,
                    ci_high=high,
                    n_sequences=int(np.isfinite(nmse[:, position]).sum()),
                )
            )

    def condition(
        self,
        name: str,
        descriptor: dict[str, Any],
        suite: Suite,
        mse: np.ndarray,
        nmse: np.ndarray,
        *,
        condition: str,
        curve_type: str,
        seed: int,
        last_mse: np.ndarray | None = None,
        last_nmse: np.ndarray | None = None,
    ) -> None:
        for prefix, values, last in (
            ("mse", mse, last_mse),
            ("nmse", nmse, last_nmse),
        ):
            metrics = {
                f"{prefix}_first_demo": values[:, 0],
                f"{prefix}_last_demo": values[:, -1] if last is None else last,
                f"{prefix}_mean": np.nanmean(values, axis=1),
            }
            if prefix == "nmse":
                metrics["nmse_aulc"] = np.nanmean(values, axis=1)
                metrics["demos_to_threshold"] = _thresholds(values)
            for offset, (metric, sequence_values) in enumerate(metrics.items()):
                self.summary(
                    name,
                    descriptor,
                    suite,
                    condition,
                    metric,
                    sequence_values,
                    seed + offset,
                )
        self.curve(
            name,
            descriptor,
            suite,
            condition,
            curve_type,
            mse,
            nmse,
            seed=seed + 100,
        )

    def difference(
        self,
        key: str,
        descriptor: dict[str, Any],
        suite: Suite,
        mse: np.ndarray,
        nmse: np.ndarray,
        *,
        metric: str,
        seed: int,
    ) -> None:
        self.curve(key, descriptor, suite, metric, f"{metric}_curve", mse, nmse, seed=seed)
        for offset, (suffix, values) in enumerate(
            (
                ("first_demo", nmse[:, 0]),
                ("last_demo", nmse[:, -1]),
                ("mean", np.nanmean(nmse, axis=1)),
            )
        ):
            self.summary(
                key, descriptor, suite, metric, f"{metric}_{suffix}", values, seed + 100 + offset
            )

    def build(self) -> EvaluationReport:
        _append_generalization_gaps(self.summary_rows, self.scalars)
        return EvaluationReport(self.scalars, self.curves, self.summary_rows, self.curve_rows)


def _binned_task_curves(
    report: _ReportBuilder,
    descriptor: dict[str, Any],
    suite: Suite,
    task_mse: np.ndarray,
    task_nmse: np.ndarray,
    coordinates: np.ndarray,
    *,
    curve_type: str,
    x_name: str,
    seed: int,
) -> None:
    values = sorted(int(value) for value in np.unique(coordinates))
    mse = np.full((len(task_mse), len(values)), np.nan)
    nmse = np.full_like(mse, np.nan)
    for sequence in range(len(task_mse)):
        for position, value in enumerate(values):
            mask = coordinates[sequence] == value
            if mask.any():
                mse[sequence, position] = np.nanmean(task_mse[sequence, mask])
                nmse[sequence, position] = np.nanmean(task_nmse[sequence, mask])
    report.curve(
        descriptor["suite"],
        descriptor,
        suite,
        "ordinary",
        curve_type,
        mse,
        nmse,
        x_name=x_name,
        x_values=np.asarray(values),
        seed=seed,
    )


def _append_generalization_gaps(rows: list[dict[str, Any]], scalars: dict[str, float]) -> None:
    """Append held-out interpolation residuals and OOD gaps within a slice."""
    primary = {
        ("icl", "ordinary", "nmse_aulc"),
        ("composition", "benefit", "benefit_mean"),
        ("retention", "savings", "savings_mean"),
    }
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows.copy():
        if (row["capability"], row["condition"], row["metric"]) in primary:
            groups.setdefault((row["capability"], row["slice"], row["metric"]), []).append(row)
    for (capability, slice_name, metric), group in groups.items():
        seen = sorted((row for row in group if row["status"] == "seen"), key=lambda row: row["M"])
        if not seen:
            continue
        for row in group:
            gap: float | None = None
            kind = ""
            if row["status"] == "heldout":
                left = [candidate for candidate in seen if candidate["M"] < row["M"]]
                right = [candidate for candidate in seen if candidate["M"] > row["M"]]
                if left and right:
                    low, high = left[-1], right[0]
                    fraction = (row["M"] - low["M"]) / (high["M"] - low["M"])
                    expected = low["value"] + fraction * (high["value"] - low["value"])
                    gap, kind = float(row["value"] - expected), "interpolation_gap"
            elif row["status"] == "ood":
                gap, kind = float(row["value"] - seen[-1]["value"]), "ood_gap"
            if gap is None:
                continue
            gap_row = dict(
                row,
                condition=kind,
                metric=f"{metric}_{kind}",
                value=gap,
                ci_low=gap,
                ci_high=gap,
            )
            rows.append(gap_row)
            scalars[
                f"generalization/{capability}/{slice_name}/m{row['M']:02d}/{gap_row['metric']}"
            ] = gap


def _cell_key(descriptor: dict[str, Any]) -> str:
    variant = f"__{descriptor['variant']}" if descriptor["variant"] else ""
    return (
        f"{descriptor['slice']}__{descriptor['status']}__m{descriptor['M']:02d}__"
        f"t{descriptor['T']:02d}__b{descriptor['B_history']:04d}{variant}"
    )


def _evaluate(
    suites: dict[str, Suite],
    mses: dict[str, np.ndarray],
    nmses: dict[str, np.ndarray],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> EvaluationReport:
    report = _ReportBuilder(bootstrap_seed, bootstrap_replicates)
    descriptors = {name: _descriptor(name, suite) for name, suite in suites.items()}
    grouped: dict[tuple[str, str], dict[str, str]] = {}

    for index, (name, descriptor) in enumerate(sorted(descriptors.items())):
        if descriptor["capability"] != "icl":
            pair_group = descriptor["pair_group"]
            grouped.setdefault((descriptor["capability"], pair_group), {})[
                descriptor["condition"]
            ] = name
            continue
        suite, mse, nmse = suites[name], mses[name], nmses[name]
        counts = suite["demo_counts"]
        report.condition(
            name,
            descriptor,
            suite,
            np.nanmean(mse, axis=1),
            np.nanmean(nmse, axis=1),
            condition="ordinary",
            curve_type="learning_curve",
            seed=index * 1000,
            last_mse=_last_values(mse, counts).mean(axis=1),
            last_nmse=_last_values(nmse, counts).mean(axis=1),
        )
        task_mse, task_nmse = np.nanmean(mse, axis=2), np.nanmean(nmse, axis=2)
        report.curve(
            name,
            descriptor,
            suite,
            "ordinary",
            "task_position_curve",
            task_mse,
            task_nmse,
            x_name="task_position",
            seed=index * 1000 + 500,
        )
        prefix_curves = {
            "history_prediction_tokens": (
                "nmse_by_prediction_tokens_observed",
                "prediction_tokens_observed",
            ),
            "num_unique_supports_seen": (
                "nmse_by_unique_supports_observed",
                "unique_supports_observed",
            ),
            "num_modules_covered": ("nmse_by_modules_covered", "modules_covered"),
        }
        for offset, (field, (curve_type, x_name)) in enumerate(prefix_curves.items()):
            if field in suite:
                _binned_task_curves(
                    report,
                    descriptor,
                    suite,
                    task_mse,
                    task_nmse,
                    suite[field],
                    curve_type=curve_type,
                    x_name=x_name,
                    seed=index * 1000 + 600 + offset * 100,
                )

    for index, ((capability, _), conditions) in enumerate(sorted(grouped.items())):
        seed = 100_000 + index * 1000
        primary = "constituent" if capability == "composition" else "repeat"
        descriptor = descriptors[conditions[primary]]
        key = f"{capability}/{_cell_key(descriptor)}"
        values: dict[str, tuple[Suite, np.ndarray, np.ndarray]] = {}
        for offset, (condition, name) in enumerate(sorted(conditions.items())):
            suite = suites[name]
            mse = _last_task_values(mses[name], suite["demo_counts"])
            nmse = _last_task_values(nmses[name], suite["demo_counts"])
            values[condition] = (suite, mse, nmse)
            report.condition(
                name,
                descriptors[name],
                suite,
                mse,
                nmse,
                condition=condition,
                curve_type=("relearning_curve" if condition == "repeat" else f"{condition}_curve"),
                seed=seed + offset * 100,
            )

        if capability == "composition":
            source, matched = values["constituent"], values["matched_prefix"]
            report.difference(
                key,
                descriptor,
                source[0],
                matched[1] - source[1],
                matched[2] - source[2],
                metric="benefit",
                seed=seed + 500,
            )
            if "no_history" in values:
                no_history = values["no_history"]
                report.difference(
                    key,
                    descriptor,
                    source[0],
                    no_history[1] - source[1],
                    no_history[2] - source[2],
                    metric="no_history_benefit",
                    seed=seed + 600,
                )
            continue

        repeat, novel = values["repeat"], values["novel"]
        report.difference(
            key,
            descriptor,
            repeat[0],
            novel[1] - repeat[1],
            novel[2] - repeat[2],
            metric="savings",
            seed=seed + 500,
        )
        report.scalars[f"{key}/savings_demo0"] = float(np.nanmean(novel[2][:, 0] - repeat[2][:, 0]))
        report.scalars[f"{key}/savings_one_demo"] = float(
            np.nanmean(novel[2][:, 1] - repeat[2][:, 1])
        )
        if "shared" in values:
            shared = values["shared"]
            for offset, (metric, left, right) in enumerate(
                (
                    ("episodic_savings", shared, repeat),
                    ("module_savings", novel, shared),
                )
            ):
                report.difference(
                    key,
                    descriptor,
                    repeat[0],
                    left[1] - right[1],
                    left[2] - right[2],
                    metric=metric,
                    seed=seed + 600 + offset * 100,
                )
        threshold_delta = _thresholds(novel[2]) - _thresholds(repeat[2])
        report.summary(
            key,
            descriptor,
            repeat[0],
            "savings",
            "demos_to_threshold_delta",
            threshold_delta,
            seed + 900,
        )
        original_mse = mses[conditions["repeat"]][:, 0, : int(repeat[0]["demo_counts"][:, 0].max())]
        original_nmse = nmses[conditions["repeat"]][
            :, 0, : int(repeat[0]["demo_counts"][:, 0].max())
        ]
        report.curve(
            key,
            descriptor,
            repeat[0],
            "original",
            "original_curve",
            original_mse,
            original_nmse,
            seed=seed + 950,
        )
    return report.build()


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
    mses, nmses = {}, {}
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
    report = _evaluate(
        capability_suites,
        mses,
        nmses,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    if validation_metric is not None:
        report.scalars[f"{VALIDATION_SUITE}/token_mse"] = validation_metric
    return report
