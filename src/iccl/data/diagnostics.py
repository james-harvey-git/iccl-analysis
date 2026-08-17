"""Empirical diagnostics for the configured training-sequence distribution."""

import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from iccl.data.curriculum import (
    CURRICULUM_SAMPLER_CODES,
    TASK_CATEGORY_CODES,
    TASK_ORIGIN_CODES,
    check_compositional,
    check_connected,
    check_full_rank,
)
from iccl.data.dataset import sequence_dataset_from_config
from iccl.data.export import _git_commit

CATEGORIES = {value: key for key, value in TASK_CATEGORY_CODES.items()}
SAMPLERS = {value: key for key, value in CURRICULUM_SAMPLER_CODES.items()}
QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)


def _distribution(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    counts = Counter(values)
    return {
        "counts": {str(key): value for key, value in sorted(counts.items())},
        "frequencies": {str(key): value / len(values) for key, value in sorted(counts.items())},
        "mean": float(array.mean()),
        "std": float(array.std()),
        "quantiles": dict(
            zip(map(str, QUANTILES), map(float, np.quantile(array, QUANTILES)), strict=True)
        ),
    }


def _ratio(
    numerators: np.ndarray,
    denominators: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float | None]:
    if not denominators.sum():
        return {"estimate": None, "ci_low": None, "ci_high": None}
    estimate = float(numerators.sum() / denominators.sum())
    if len(numerators) < 2 or replicates < 2:
        return {"estimate": estimate, "ci_low": estimate, "ci_high": estimate}
    rng = np.random.Generator(
        np.random.Philox(key=np.array([seed, len(numerators)], dtype=np.uint64))
    )
    indices = rng.integers(0, len(numerators), size=(replicates, len(numerators)))
    denominator = denominators[indices].sum(axis=1)
    bootstrap = np.divide(
        numerators[indices].sum(axis=1),
        denominator,
        out=np.full(replicates, np.nan),
        where=denominator > 0,
    )
    low, high = np.nanquantile(bootstrap, [0.025, 0.975])
    return {"estimate": estimate, "ci_low": float(low), "ci_high": float(high)}


def _record(sample: Any, elapsed: float, require_full_rank: bool) -> dict[str, Any]:
    info = sample.info
    tasks = int(info["num_curriculum_tasks"])
    origins = info["task_origin"][:tasks]
    surplus = origins == TASK_ORIGIN_CODES["surplus"]
    demos = info["demo_counts"][:tasks][surplus]
    record: dict[str, Any] = {
        "M": int(info["num_modules"]),
        "S": int(info["num_surplus_tasks"]),
        "T": tasks,
        "B": int(info["num_prediction_tokens"]),
        "L": int(info["serialized_length"]),
        "sampler": SAMPLERS[int(info["curriculum_sampler"])],
        "attempts": int(info["generation_attempts"]),
        "elapsed": elapsed,
        "surplus_tokens": int(demos.sum()),
    }
    for relative, field in (
        ("generation", "generation_category"),
        ("presentation", "presentation_category"),
    ):
        categories = info[field][:tasks][surplus]
        record[f"{relative}_counts"] = np.bincount(categories, minlength=len(CATEGORIES))
        record[f"{relative}_tokens"] = np.bincount(
            categories, weights=demos, minlength=len(CATEGORIES)
        )
    supports = info["latents"][:tasks] > 0
    record["coverage_ok"] = check_compositional(supports, record["M"])
    record["connected_ok"] = check_connected(supports)
    record["rank_ok"] = not require_full_rank or check_full_rank(info["latents"][:tasks])
    return record


def _category_report(
    records: list[dict[str, Any]], relative: str, seed: int, replicates: int
) -> dict[str, Any]:
    surplus = np.asarray([record["S"] for record in records], dtype=np.float64)
    token_totals = np.asarray([record["surplus_tokens"] for record in records], dtype=np.float64)
    positive = surplus > 0
    report: dict[str, Any] = {}
    for code, name in CATEGORIES.items():
        counts = np.asarray(
            [record[f"{relative}_counts"][code] for record in records], dtype=np.float64
        )
        tokens = np.asarray(
            [record[f"{relative}_tokens"][code] for record in records], dtype=np.float64
        )
        fractions = counts[positive] / surplus[positive]
        report[name] = {
            "sequence_uniform": _ratio(
                fractions, np.ones(len(fractions)), seed=seed + code, replicates=replicates
            ),
            "surplus_task_uniform": _ratio(
                counts, surplus, seed=seed + 100 + code, replicates=replicates
            ),
            "loss_token_weighted": _ratio(
                tokens, token_totals, seed=seed + 200 + code, replicates=replicates
            ),
            "per_sequence_fraction_summary": {
                "mean": float(fractions.mean()) if len(fractions) else None,
                "std": float(fractions.std()) if len(fractions) else None,
                "quantiles": dict(
                    zip(
                        map(str, QUANTILES),
                        map(float, np.quantile(fractions, QUANTILES)),
                        strict=True,
                    )
                )
                if len(fractions)
                else {},
            },
        }
    report["p_surplus_zero"] = float(np.mean(~positive))
    return report


def _stratified(records: list[dict[str, Any]], weighting: str) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["M"], record["S"], record["sampler"])].append(record)
    rows = []
    for (modules, surplus, sampler), group in sorted(groups.items()):
        total = sum(record["S"] for record in group)
        for relative in ("generation", "presentation"):
            counts = np.sum([record[f"{relative}_counts"] for record in group], axis=0)
            rows.extend(
                {
                    "M": modules,
                    "S": surplus,
                    "sampler": sampler,
                    "weighting": weighting,
                    "relative_to": relative,
                    "category": category,
                    "count": int(counts[code]),
                    "fraction": float(counts[code] / total) if total else None,
                    "num_sequences": len(group),
                }
                for code, category in CATEGORIES.items()
            )
    return rows


def analyze_sequence_distribution(cfg: DictConfig, out_dir: Path) -> dict[str, Any]:
    """Sample the real generator and write JSON and long-form CSV reports."""
    config = cfg.data.distribution_diagnostic
    requested = int(config.num_sequences)
    dataset = sequence_dataset_from_config(cfg.data, base_seed=int(cfg.seed))
    records, failures = [], []
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
        raise RuntimeError(f"all {requested} diagnostic sequences failed")

    batch_size = int(cfg.training.batch_size)
    real_tokens = padded_tokens = 0
    for start in range(0, len(records), batch_size):
        lengths = [record["L"] for record in records[start : start + batch_size]]
        real_tokens += sum(lengths)
        padded_tokens += len(lengths) * max(lengths)

    rejection = []
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["M"], record["S"])].append(record)
    for (modules, surplus), group in sorted(groups.items()):
        attempts = sum(record["attempts"] for record in group)
        group_elapsed = sum(record["elapsed"] for record in group)
        rejection.append(
            {
                "M": modules,
                "S": surplus,
                "num_sequences": len(group),
                "attempts": attempts,
                "acceptance_probability": len(group) / attempts,
                "sequences_per_second": len(group) / group_elapsed,
            }
        )

    bootstrap_seed, replicates = int(config.bootstrap_seed), int(config.bootstrap_replicates)
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
            record["M"] in dataset.module_counts.held_out for record in records
        ),
        "categories": {
            "generation_relative": _category_report(
                records, "generation", bootstrap_seed, replicates
            ),
            "presentation_relative": _category_report(
                records, "presentation", bootstrap_seed + 1000, replicates
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
