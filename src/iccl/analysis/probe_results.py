"""Coordinatewise decoding metrics and equal-world, paired uncertainty estimates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from iccl.analysis.probe_worlds import stream_seed

METRICS = ("r2_macro", "active_pair_accuracy", "active_coefficient_rmse", "fixed_scale_vector_mse")


def finite(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def score_predictions(
    targets: np.ndarray, predictions: np.ndarray, training_variance: np.ndarray
) -> dict[str, Any]:
    if targets.shape != predictions.shape or targets.ndim != 2 or not len(targets):
        raise ValueError("scores require aligned, nonempty target/prediction matrices")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("nonfinite decoding predictions/targets")
    z, predicted = targets.astype(np.float64), predictions.astype(np.float64)
    errors = (predicted - z) ** 2
    sse = errors.sum(axis=0)
    tss = ((z - z.mean(axis=0)) ** 2).sum(axis=0)
    coordinates = np.divide(sse, tss, out=np.full_like(sse, np.nan), where=tss > 0)
    coordinates = 1 - coordinates
    active = z != 0
    if not np.all(active.sum(axis=1) == 2):
        raise ValueError("task-vector scoring requires exactly two active constituents")
    top = np.argsort(-predicted, axis=1, kind="stable")[:, :2]
    predicted_support = np.zeros_like(active)
    np.put_along_axis(predicted_support, top, True, axis=1)
    variance = np.asarray(training_variance, dtype=np.float64)
    fixed = (
        float(np.mean(errors / variance))
        if variance.shape == (z.shape[1],) and np.all(variance > 0)
        else None
    )
    return {
        "r2_macro": finite(coordinates.mean()),
        "r2_coordinates": [finite(x) for x in coordinates],
        "sse": sse.tolist(),
        "tss": tss.tolist(),
        "mse": float(errors.mean()),
        "active_pair_accuracy": float(np.all(predicted_support == active, axis=1).mean()),
        "active_coefficient_rmse": float(np.sqrt(errors[active].mean())),
        "fixed_scale_vector_mse": fixed,
        "trials": len(z),
        "undefined_coordinates": np.flatnonzero(tss == 0).tolist(),
        "target_variances": z.var(axis=0).tolist(),
    }


def world_interval(
    values: dict[int, float], *, seed: int, replicates: int, confidence: float
) -> dict[str, Any]:
    ids = sorted(values)
    array = np.array([values[i] for i in ids], dtype=np.float64)
    if not len(array):
        return {"mean": None, "lower": None, "upper": None, "world_count": 0, "world_ids": []}
    result: dict[str, Any] = {
        "mean": float(array.mean()),
        "world_count": len(ids),
        "world_ids": ids,
    }
    if len(ids) == 1:
        return result | {"lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(ids), size=(replicates, len(ids)))
    bounds = np.quantile(array[draws].mean(axis=1), [(1 - confidence) / 2, (1 + confidence) / 2])
    return result | {"lower": float(bounds[0]), "upper": float(bounds[1])}


CONDITION = (
    "phase",
    "population",
    "position",
    "capture",
    "family",
    "selection",
    "control",
    "train_count",
)


def aggregate_scores(
    rows: list[dict[str, Any]], *, worlds: int, seed: int, replicates: int, confidence: float
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in CONDITION)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        base = dict(zip(CONDITION, key, strict=False))
        for metric in METRICS:
            values = {
                r["world"]: r["scores"][metric] for r in group if r["scores"][metric] is not None
            }
            if len({r["world"] for r in group}) != len(group):
                raise ValueError("duplicate world scores in an aggregation condition")
            estimate = world_interval(
                values,
                seed=stream_seed(seed, f"world-bootstrap/{sorted(values)}"),
                replicates=replicates,
                confidence=confidence,
            )
            output.append(
                base
                | {
                    "metric": metric,
                    **estimate,
                    "expected_world_count": worlds,
                    "missing_world_ids": sorted(set(range(worlds)) - values.keys()),
                    "trial_counts": {str(r["world"]): r["scores"]["trials"] for r in group},
                }
            )
    return output


def exposure_bin(count: int, bins: str | list[int]) -> int:
    return count if bins == "exact" else int(np.searchsorted(bins, count, side="right") - 1)


def exposure_scores(
    targets: np.ndarray,
    predictions: np.ndarray,
    exposures: np.ndarray,
    variance: np.ndarray,
    bins: str | list[int],
) -> list[dict[str, Any]]:
    sorted_counts = np.sort(exposures, axis=1)
    codes = np.asarray([[exposure_bin(int(x), bins) for x in pair] for pair in sorted_counts])
    result = []
    for pair in np.unique(codes, axis=0):
        selected = np.all(codes == pair, axis=1)
        result.append(
            {
                "e_min": int(pair[0]),
                "e_max": int(pair[1]),
                "scores": score_predictions(targets[selected], predictions[selected], variance),
                "target_pairs": np.unique(
                    np.stack([np.flatnonzero(row) for row in targets[selected]]), axis=0
                ).tolist(),
            }
        )
    return result


def aggregate_exposure(
    rows: list[dict[str, Any]],
    *,
    worlds: int,
    seed: int,
    replicates: int,
    confidence: float,
    min_trials: int,
    min_worlds: int,
) -> list[dict[str, Any]]:
    panels: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        panels[tuple(row[k] for k in CONDITION)].append(row)
    result = []
    for key, group in sorted(panels.items()):
        base = dict(zip(CONDITION, key, strict=False))
        cells = sorted({(r["e_min"], r["e_max"]) for r in group})
        for metric in ("r2_macro", "fixed_scale_vector_mse", "active_pair_accuracy"):
            qualified = {
                cell: {
                    r["world"]: r["scores"][metric]
                    for r in group
                    if (r["e_min"], r["e_max"]) == cell
                    and r["scores"]["trials"] >= min_trials
                    and r["scores"][metric] is not None
                }
                for cell in cells
            }
            supported = [set(values) for values in qualified.values() if len(values) >= min_worlds]
            common = set.intersection(*supported) if supported else set()
            for cell in cells:
                values = {w: value for w, value in qualified[cell].items() if w in common}
                valid = len(values) >= min_worlds
                estimate = world_interval(
                    values if valid else {},
                    seed=stream_seed(seed, f"world-bootstrap/{sorted(common)}"),
                    replicates=replicates,
                    confidence=confidence,
                )
                subset = [r for r in group if (r["e_min"], r["e_max"]) == cell]
                result.append(
                    base
                    | {
                        "e_min": cell[0],
                        "e_max": cell[1],
                        "metric": metric,
                        **estimate,
                        "status": "estimated" if valid else "insufficient_coverage",
                        "qualified_world_count": len(qualified[cell]),
                        "common_world_ids": sorted(common),
                        "expected_world_count": worlds,
                        "trial_counts": {str(r["world"]): r["scores"]["trials"] for r in subset},
                    }
                )
    return result


def paired_control_scores(
    rows: list[dict[str, Any]], *, seed: int, replicates: int, confidence: float
) -> list[dict[str, Any]]:
    index = {tuple(r[k] for k in CONDITION) + (r["world"],): r for r in rows}
    groups: dict[tuple, dict[int, float]] = defaultdict(dict)
    for row in rows:
        if row["control"] != "trained":
            continue
        for control in ("shuffled_targets", "untrained", "training_mean", "observations"):
            match = dict(row)
            match["control"] = control
            if control in {"training_mean", "observations"}:
                match.update(family="baseline", selection=control)
            other = index.get(tuple(match[k] for k in CONDITION) + (row["world"],))
            if other is None:
                continue
            for metric in METRICS:
                a, b = row["scores"][metric], other["scores"][metric]
                if a is not None and b is not None:
                    key = tuple(row[k] for k in CONDITION) + (control, metric)
                    groups[key][row["world"]] = a - b
    return [
        dict(zip((*CONDITION, "comparison", "metric"), key, strict=False))
        | world_interval(
            values,
            seed=stream_seed(seed, f"world-bootstrap/{sorted(values)}"),
            replicates=replicates,
            confidence=confidence,
        )
        for key, values in sorted(groups.items())
    ]
