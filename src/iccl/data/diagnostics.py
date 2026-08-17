"""Empirical diagnostics for the configured on-the-fly training distribution."""

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from iccl.data.dataset import sequence_dataset_from_config
from iccl.data.export import _git_commit
from iccl.data.sequences import (
    CURRICULUM_SAMPLER_CODES,
    TASK_CATEGORY_CODES,
    TASK_ORIGIN_CODES,
    check_compositional,
    check_connected,
    check_full_rank,
)

CATEGORY_NAMES = {value: key for key, value in TASK_CATEGORY_CODES.items()}
SAMPLER_NAMES = {value: key for key, value in CURRICULUM_SAMPLER_CODES.items()}


def _distribution(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    counts = Counter(values)
    return {
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "frequencies": {
            str(key): float(value / len(values)) for key, value in sorted(counts.items())
        },
        "mean": float(array.mean()),
        "std": float(array.std()),
        "quantiles": {
            str(q): float(value)
            for q, value in zip(
                (0.0, 0.25, 0.5, 0.75, 0.95, 1.0),
                np.quantile(array, (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)),
                strict=True,
            )
        },
    }


def _ratio_interval(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | None]:
    total_denominator = float(denominators.sum())
    if total_denominator == 0:
        return {"estimate": None, "ci_low": None, "ci_high": None}
    estimate = float(numerators.sum() / total_denominator)
    if len(numerators) < 2 or replicates < 2:
        return {"estimate": estimate, "ci_low": estimate, "ci_high": estimate}
    rng = np.random.Generator(
        np.random.Philox(key=np.array([seed, len(numerators)], dtype=np.uint64))
    )
    indices = rng.integers(0, len(numerators), size=(replicates, len(numerators)))
    numerator = numerators[indices].sum(axis=1)
    denominator = denominators[indices].sum(axis=1)
    bootstrap = np.divide(
        numerator,
        denominator,
        out=np.full(replicates, np.nan),
        where=denominator > 0,
    )
    low, high = np.nanquantile(bootstrap, [0.025, 0.975])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high)}


def _category_estimates(
    records: list[dict[str, Any]],
    field: str,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    surplus = np.array([record["S"] for record in records], dtype=np.float64)
    surplus_demo_tokens = np.array(
        [record["surplus_demo_tokens"] for record in records], dtype=np.float64
    )
    for offset, (code, name) in enumerate(sorted(CATEGORY_NAMES.items())):
        counts = np.array(
            [record[f"{field}_counts"].get(code, 0) for record in records], dtype=np.float64
        )
        demo_tokens = np.array(
            [record[f"{field}_demo_tokens"].get(code, 0) for record in records],
            dtype=np.float64,
        )
        positive = surplus > 0
        fractions = np.divide(
            counts[positive],
            surplus[positive],
            out=np.zeros(int(positive.sum())),
            where=surplus[positive] > 0,
        )
        results[name] = {
            "sequence_uniform": _ratio_interval(
                fractions,
                np.ones(int(positive.sum())),
                seed=bootstrap_seed + offset,
                replicates=bootstrap_replicates,
            ),
            "surplus_task_uniform": _ratio_interval(
                counts,
                surplus,
                seed=bootstrap_seed + 100 + offset,
                replicates=bootstrap_replicates,
            ),
            "loss_token_weighted": _ratio_interval(
                demo_tokens,
                surplus_demo_tokens,
                seed=bootstrap_seed + 200 + offset,
                replicates=bootstrap_replicates,
            ),
            "per_sequence_fraction_summary": {
                "mean": float(fractions.mean()) if fractions.size else None,
                "std": float(fractions.std()) if fractions.size else None,
                "quantiles": {
                    str(q): float(value)
                    for q, value in zip(
                        (0.0, 0.25, 0.5, 0.75, 0.95, 1.0),
                        np.quantile(fractions, (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)),
                        strict=True,
                    )
                }
                if fractions.size
                else {},
            },
        }
    results["p_surplus_zero"] = float(np.mean(surplus == 0))
    return results


def _record(sample: Any, elapsed: float, require_full_rank: bool) -> dict[str, Any]:
    info = sample.info
    curriculum = int(info["num_curriculum_tasks"])
    origins = info["task_origin"][:curriculum]
    surplus_mask = origins == TASK_ORIGIN_CODES["surplus"]
    demo_counts = info["demo_counts"][:curriculum]
    result: dict[str, Any] = {
        "M": int(info["num_modules"]),
        "S": int(info["num_surplus_tasks"]),
        "T": curriculum,
        "B": int(info["num_prediction_tokens"]),
        "L": int(info["serialized_length"]),
        "sampler": SAMPLER_NAMES[int(info["curriculum_sampler"])],
        "attempts": int(info["generation_attempts"]),
        "elapsed": elapsed,
        "surplus_demo_tokens": int(demo_counts[surplus_mask].sum()),
    }
    for field in ("generation_category", "presentation_category"):
        categories = info[field][:curriculum][surplus_mask]
        result[f"{field}_counts"] = Counter(int(value) for value in categories)
        result[f"{field}_demo_tokens"] = Counter()
        for category, demos in zip(categories, demo_counts[surplus_mask], strict=True):
            result[f"{field}_demo_tokens"][int(category)] += int(demos)
    supports = info["latents"][:curriculum] > 0
    result["coverage_ok"] = check_compositional(supports, result["M"])
    result["connected_ok"] = check_connected(supports)
    result["rank_ok"] = (
        check_full_rank(info["latents"][:curriculum]) if require_full_rank else True
    )
    return result


def _stratified(records: list[dict[str, Any]], weighting: str) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["M"], record["S"], record["sampler"])].append(record)
    rows = []
    for (modules, surplus, sampler), group in sorted(groups.items()):
        for relative in ("generation_category", "presentation_category"):
            total = sum(record["S"] for record in group)
            for code, category in sorted(CATEGORY_NAMES.items()):
                count = sum(record[f"{relative}_counts"].get(code, 0) for record in group)
                rows.append(
                    {
                        "M": modules,
                        "S": surplus,
                        "sampler": sampler,
                        "weighting": weighting,
                        "relative_to": relative.removesuffix("_category"),
                        "category": category,
                        "count": int(count),
                        "fraction": float(count / total) if total else None,
                        "num_sequences": len(group),
                    }
                )
    return rows


def analyze_sequence_distribution(
    cfg: DictConfig,
    out_dir: Path,
) -> dict[str, Any]:
    """Sample the real generator and write its empirical distribution report."""
    diagnostic = cfg.data.distribution_diagnostic
    requested = int(diagnostic.num_sequences)
    bootstrap_seed = int(diagnostic.bootstrap_seed)
    bootstrap_replicates = int(diagnostic.bootstrap_replicates)
    dataset = sequence_dataset_from_config(cfg.data, base_seed=int(cfg.seed))
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index in range(requested):
        sequence_started = time.perf_counter()
        try:
            sample = dataset.build(index)
        except (RuntimeError, ValueError) as error:
            failures.append({"index": index, "error": str(error)})
            continue
        records.append(
            _record(
                sample,
                time.perf_counter() - sequence_started,
                bool(cfg.data.sequence.get("require_full_rank", False)),
            )
        )
    elapsed = time.perf_counter() - started
    if not records:
        raise RuntimeError(f"all {requested} diagnostic sequences failed to generate")

    batch_size = int(cfg.training.batch_size)
    padded_tokens = 0
    real_tokens = 0
    for start in range(0, len(records), batch_size):
        lengths = [record["L"] for record in records[start : start + batch_size]]
        real_tokens += sum(lengths)
        padded_tokens += len(lengths) * max(lengths)

    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_pair[(record["M"], record["S"])].append(record)
    rejection = []
    for (modules, surplus), group in sorted(by_pair.items()):
        attempts = sum(record["attempts"] for record in group)
        group_elapsed = sum(record["elapsed"] for record in group)
        rejection.append(
            {
                "M": modules,
                "S": surplus,
                "num_sequences": len(group),
                "attempts": attempts,
                "acceptance_probability": float(len(group) / attempts),
                "sequences_per_second": float(len(group) / group_elapsed),
            }
        )

    report: dict[str, Any] = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "seed": int(cfg.seed),
        "git_commit": _git_commit(),
        "requested_sequences": requested,
        "generated_sequences": len(records),
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_seconds": elapsed,
        "sequences_per_second": len(records) / elapsed,
        "distributions": {
            key: _distribution([record[key] for record in records])
            for key in ("M", "S", "T", "B", "L")
        },
        "held_out_module_occurrences": sum(
            record["M"] in set(dataset.module_counts.held_out) for record in records
        ),
        "categories": {
            "generation_relative": _category_estimates(
                records,
                "generation_category",
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            ),
            "presentation_relative": _category_estimates(
                records,
                "presentation_category",
                bootstrap_seed=bootstrap_seed + 1000,
                bootstrap_replicates=bootstrap_replicates,
            ),
        },
        "stratified_categories": _stratified(records, str(cfg.data.weighting)),
        "rejection_efficiency": rejection,
        "invariant_failures": {
            "coverage": sum(not record["coverage_ok"] for record in records),
            "connectedness": sum(not record["connected_ok"] for record in records),
            "full_rank": sum(not record["rank_ok"] for record in records),
        },
        "batch_padding": {
            "batch_size": batch_size,
            "real_serialized_tokens": real_tokens,
            "padded_serialized_tokens": padded_tokens,
            "expansion": padded_tokens / real_tokens,
            "padding_fraction": 1.0 - real_tokens / padded_tokens,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sequence_distribution.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    rows = report["stratified_categories"]
    with (out_dir / "sequence_distribution.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return report
