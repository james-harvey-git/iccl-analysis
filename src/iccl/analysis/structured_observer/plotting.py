"""Paired comparison figures for GDN and structured GP observers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

METHOD_ORDER = ("gdn", "full_history", "current_task")
METHOD_LABELS = {
    "gdn": "GDN",
    "full_history": "Full-history structured GP",
    "current_task": "Current-task-only structured GP",
}
METHOD_COLORS = {
    "gdn": "#3366cc",
    "full_history": "#109618",
    "current_task": "#dc3912",
}


def paired_bootstrap_methods(
    methods: dict[str, np.ndarray],
    *,
    differences: tuple[tuple[str, str], ...],
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Bootstrap method curves and differences with one shared sequence draw."""
    if not methods:
        raise ValueError("at least one method is required")
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    sequence_counts = {values.shape[0] for values in methods.values()}
    if any(values.ndim != 2 for values in methods.values()) or len(sequence_counts) != 1:
        raise ValueError("method arrays must all have shape [same sequences, demos]")
    num_sequences = sequence_counts.pop()
    if num_sequences == 0:
        raise ValueError("bootstrap needs at least one sequence")
    demos = min(values.shape[1] for values in methods.values())
    trimmed = {name: values[:, :demos].astype(np.float64) for name, values in methods.items()}
    requested = set(sum(([left, right] for left, right in differences), []))
    if not requested <= trimmed.keys():
        raise ValueError("a requested difference names a missing method")

    quantities = list(trimmed)
    for left, right in differences:
        quantities.append(f"{left}_minus_{right}")
    point: dict[str, np.ndarray] = {
        name: np.nanmean(values, axis=0) for name, values in trimmed.items()
    }
    point.update(
        {
            f"{left}_minus_{right}": np.nanmean(trimmed[left] - trimmed[right], axis=0)
            for left, right in differences
        }
    )

    bootstrap = {
        name: np.empty((replicates, demos), dtype=np.float64) for name in quantities
    }
    rng = np.random.default_rng(seed)
    for replicate in range(replicates):
        indices = rng.integers(0, num_sequences, size=num_sequences)
        for name, values in trimmed.items():
            bootstrap[name][replicate] = np.nanmean(values[indices], axis=0)
        for left, right in differences:
            name = f"{left}_minus_{right}"
            bootstrap[name][replicate] = np.nanmean(
                trimmed[left][indices] - trimmed[right][indices],
                axis=0,
            )

    result: dict[str, np.ndarray] = {}
    for name in quantities:
        low, high = np.quantile(bootstrap[name], [0.025, 0.975], axis=0)
        result[f"{name}_mean"] = point[name].astype(np.float32)
        result[f"{name}_ci_low"] = low.astype(np.float32)
        result[f"{name}_ci_high"] = high.astype(np.float32)
    return result


def _save_figure(figure: Figure, prefix: Path) -> list[Path]:
    paths = [prefix.with_suffix(".png"), prefix.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return paths


def _method_figure(
    x: np.ndarray,
    bootstrap: dict[str, np.ndarray],
    *,
    ylabel: str,
    title: str,
) -> Figure:
    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for method in METHOD_ORDER:
        axis.plot(
            x,
            bootstrap[f"{method}_mean"],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linewidth=2,
        )
        axis.fill_between(
            x,
            bootstrap[f"{method}_ci_low"],
            bootstrap[f"{method}_ci_high"],
            color=METHOD_COLORS[method],
            alpha=0.18,
        )
    axis.set(
        xlabel="Current-task demonstrations already observed",
        ylabel=ylabel,
        title=title,
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _difference_figure(
    x: np.ndarray,
    bootstrap: dict[str, np.ndarray],
    *,
    metric_label: str,
) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    panels = (
        (
            "current_task_minus_full_history",
            "Value of sequence history",
            f"Current-task minus full-history {metric_label}",
            "#ff9900",
        ),
        (
            "gdn_minus_full_history",
            "Learned-algorithm reference gap",
            f"GDN minus full-history {metric_label}",
            "#990099",
        ),
    )
    for axis, (name, title, ylabel, color) in zip(axes, panels, strict=True):
        axis.plot(x, bootstrap[f"{name}_mean"], color=color, linewidth=2)
        axis.fill_between(
            x,
            bootstrap[f"{name}_ci_low"],
            bootstrap[f"{name}_ci_high"],
            color=color,
            alpha=0.2,
        )
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        axis.set(title=title, xlabel="Current-task demonstrations already observed", ylabel=ylabel)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _heatmap_figure(
    gdn: np.ndarray,
    full_history: np.ndarray,
    current_task: np.ndarray,
) -> tuple[Figure, dict[str, np.ndarray]]:
    history_benefit = np.nanmean(current_task - full_history, axis=0)
    gdn_gap = np.nanmean(gdn - full_history, axis=0)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, values, title in (
        (axes[0], history_benefit, "Current-task minus full-history NMSE"),
        (axes[1], gdn_gap, "GDN minus full-history NMSE"),
    ):
        finite = np.abs(values[np.isfinite(values)])
        limit = float(finite.max()) if finite.size else 1.0
        limit = max(limit, 1e-12)
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
        )
        axis.set(
            xlabel="Current-task demonstrations already observed",
            title=title,
        )
        axis.set_yticks(np.arange(values.shape[0]), labels=np.arange(1, values.shape[0] + 1))
        figure.colorbar(image, ax=axis, label="NMSE difference")
    axes[0].set_ylabel("Task position")
    figure.tight_layout()
    return figure, {
        "history_benefit_nmse_heatmap": history_benefit.astype(np.float32),
        "gdn_full_history_nmse_gap_heatmap": gdn_gap.astype(np.float32),
    }


def _coverage_figure(
    x: np.ndarray,
    methods: dict[str, np.ndarray],
    coverage: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[Figure, dict[str, np.ndarray]]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    plotted: dict[str, np.ndarray] = {}
    strata = (
        (coverage, "all_seen", "Both current modules seen previously", "#109618"),
        (~coverage, "unseen", "At least one current module unseen", "#990099"),
    )
    for axis, (mask, name, title, color) in zip(axes, strata, strict=True):
        sample_size = int(mask.sum())
        if sample_size:
            result = paired_bootstrap_methods(
                {key: value[mask] for key, value in methods.items()},
                differences=(("current_task", "full_history"),),
                replicates=replicates,
                seed=seed + (0 if name == "all_seen" else 1),
            )
            difference = "current_task_minus_full_history"
            axis.plot(x, result[f"{difference}_mean"], color=color, linewidth=2)
            axis.fill_between(
                x,
                result[f"{difference}_ci_low"],
                result[f"{difference}_ci_high"],
                color=color,
                alpha=0.2,
            )
            plotted[f"coverage_{name}_sample_size"] = np.asarray(sample_size, dtype=np.int32)
            for key, value in result.items():
                plotted[f"coverage_{name}_{key}"] = value
        axis.axhline(0.0, color="black", linewidth=1, alpha=0.7)
        axis.set(
            title=f"{title} (n={sample_size})",
            xlabel="Current-task demonstrations already observed",
        )
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Current-task minus full-history NMSE")
    figure.tight_layout()
    return figure, plotted


def _diagnostic_figure(
    cache: dict[str, np.ndarray],
    task_index: int,
    demo_count: int,
) -> tuple[Figure, dict[str, np.ndarray]]:
    x = np.arange(demo_count)
    full_ess = np.nanmean(
        cache["full_ess_after_by_seed"][:, :, task_index, :demo_count],
        axis=(0, 1),
    )
    full_max_weight = np.nanmean(
        cache["full_max_weight_after_by_seed"][:, :, task_index, :demo_count], axis=(0, 1)
    )
    unique_prefixes = np.nanmean(
        cache["full_unique_prefixes_after_by_seed"][:, :, task_index, :demo_count], axis=(0, 1)
    )
    resampling_fraction = np.mean(
        cache["full_resampled_by_seed"][:, :, task_index, :demo_count], axis=(0, 1)
    )
    full_trace = np.nanmean(
        cache["full_predictive_covariance_trace_by_seed"][:, :, task_index, :demo_count],
        axis=(0, 1),
    )
    current_trace = np.nanmean(
        cache["current_task_predictive_covariance_trace"][:, task_index, :demo_count],
        axis=0,
    )
    task_end_acceptance = np.nanmean(
        cache["full_task_end_rejuvenation_acceptance_by_seed"],
        axis=(0, 1),
    )
    algorithmic_std = np.sqrt(
        np.nanmean(
            cache["full_algorithmic_prediction_std"][:, task_index, :demo_count].astype(
                np.float64
            )
            ** 2,
            axis=(0, -1),
        )
    )
    diagnostics = {
        "full_ess_after_mean": full_ess.astype(np.float32),
        "full_max_weight_after_mean": full_max_weight.astype(np.float32),
        "full_unique_prefixes_after_mean": unique_prefixes.astype(np.float32),
        "full_resampling_fraction": resampling_fraction.astype(np.float32),
        "full_predictive_covariance_trace_mean": full_trace.astype(np.float32),
        "current_predictive_covariance_trace_mean": current_trace.astype(np.float32),
        "task_end_rejuvenation_acceptance_mean": task_end_acceptance.astype(np.float32),
        "full_algorithmic_prediction_rms_std": algorithmic_std.astype(np.float32),
    }

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].plot(x, full_ess)
    axes[0, 0].set_title("Full-history ESS after update")
    axes[0, 1].plot(x, full_max_weight)
    axes[0, 1].set_title("Largest particle weight after update")
    axes[0, 2].plot(x, unique_prefixes)
    axes[0, 2].set_title("Unique causal schedule prefixes")
    axes[1, 0].plot(x, resampling_fraction)
    axes[1, 0].set_title("Fraction of runs resampled")
    axes[1, 1].plot(x, full_trace, label="Full-history")
    axes[1, 1].plot(x, current_trace, label="Current-task")
    axes[1, 1].set_title("Predictive covariance trace")
    axes[1, 1].legend(frameon=False)
    task_positions = np.arange(1, len(task_end_acceptance) + 1)
    axes[1, 2].plot(task_positions, task_end_acceptance)
    axes[1, 2].set_title("Task-end MH acceptance")
    axes[1, 2].set_xlabel("Task position")
    for axis in axes.flat:
        if axis is not axes[1, 2]:
            axis.set_xlabel("Current-task demonstrations already observed")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure, diagnostics


def _algorithmic_stability_figure(
    x: np.ndarray,
    diagnostics: dict[str, np.ndarray],
) -> Figure:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(x, diagnostics["full_algorithmic_prediction_rms_std"], linewidth=2)
    axis.set(
        xlabel="Current-task demonstrations already observed",
        ylabel="RMS prediction standard deviation",
        title="Full-history observer variability across SMC seeds",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_structured_observer_comparison(
    gdn_nmse: np.ndarray,
    gdn_raw_mse: np.ndarray,
    cache: dict[str, np.ndarray],
    *,
    suite_name: str,
    task_position: int,
    checkpoint_step: int,
    output_root: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    include_raw_mse: bool,
    provenance: dict[str, Any],
) -> list[Path]:
    """Write the final-task comparison, history gaps, heatmaps, and diagnostics."""
    required = {
        "current_task_nmse",
        "full_nmse",
        "current_task_raw_mse",
        "full_raw_mse",
        "sequence_indices",
        "demo_counts",
        "all_current_modules_seen",
    }
    missing = sorted(required - cache.keys())
    if missing:
        raise ValueError(f"structured cache is missing arrays: {', '.join(missing)}")
    sequence_indices = cache["sequence_indices"].astype(np.int64)
    gdn_nmse = gdn_nmse[sequence_indices]
    gdn_raw_mse = gdn_raw_mse[sequence_indices]
    if gdn_nmse.shape != cache["full_nmse"].shape:
        raise ValueError("GDN and structured-observer NMSE shapes do not match")
    task_index = task_position - 1
    if not 0 <= task_index < gdn_nmse.shape[1]:
        raise ValueError("task_position is outside the cached task range")
    demo_count = int(cache["demo_counts"][:, task_index].max())
    x = np.arange(demo_count)
    step_label = (
        f"{checkpoint_step // 1000}k" if checkpoint_step % 1000 == 0 else str(checkpoint_step)
    )
    output_dir = output_root / f"pilot-{step_label}-steps"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{suite_name}-task-{task_position}"
    written: list[Path] = []
    plotted: dict[str, np.ndarray] = {
        "demo_index": x,
        "task_positions": cache["task_positions"],
    }

    def comparison(
        gdn: np.ndarray,
        full: np.ndarray,
        current: np.ndarray,
        *,
        metric: str,
        ylabel: str,
        seed_offset: int,
    ) -> dict[str, np.ndarray]:
        methods = {
            "gdn": gdn[:, task_index, :demo_count],
            "full_history": full[:, task_index, :demo_count],
            "current_task": current[:, task_index, :demo_count],
        }
        result = paired_bootstrap_methods(
            methods,
            differences=(
                ("current_task", "full_history"),
                ("gdn", "full_history"),
            ),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + seed_offset,
        )
        title = f"{suite_name}: task position {task_position}"
        written.extend(
            _save_figure(
                _method_figure(x, result, ylabel=ylabel, title=title),
                output_dir / f"{stem}-{metric}-comparison",
            )
        )
        written.extend(
            _save_figure(
                _difference_figure(x, result, metric_label=ylabel),
                output_dir / f"{stem}-{metric}-differences",
            )
        )
        for key, value in result.items():
            plotted[f"{metric}_{key}"] = value
        return methods

    nmse_methods = comparison(
        gdn_nmse,
        cache["full_nmse"],
        cache["current_task_nmse"],
        metric="nmse",
        ylabel="NMSE",
        seed_offset=0,
    )
    if include_raw_mse:
        comparison(
            gdn_raw_mse,
            cache["full_raw_mse"],
            cache["current_task_raw_mse"],
            metric="raw-mse",
            ylabel="raw MSE",
            seed_offset=100,
        )

    heatmap, heatmap_arrays = _heatmap_figure(
        gdn_nmse,
        cache["full_nmse"],
        cache["current_task_nmse"],
    )
    written.extend(_save_figure(heatmap, output_dir / f"{stem}-task-heatmaps"))
    plotted.update(heatmap_arrays)

    coverage_figure, coverage_arrays = _coverage_figure(
        x,
        nmse_methods,
        cache["all_current_modules_seen"][:, task_index],
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 200,
    )
    written.extend(_save_figure(coverage_figure, output_dir / f"{stem}-coverage-strata"))
    plotted.update(coverage_arrays)

    diagnostic_figure, diagnostic_arrays = _diagnostic_figure(
        cache,
        task_index,
        demo_count,
    )
    written.extend(_save_figure(diagnostic_figure, output_dir / f"{stem}-diagnostics"))
    plotted.update(diagnostic_arrays)
    written.extend(
        _save_figure(
            _algorithmic_stability_figure(x, diagnostic_arrays),
            output_dir / f"{stem}-algorithmic-stability",
        )
    )

    arrays_path = output_dir / f"{stem}-plotted-arrays.npz"
    np.savez_compressed(arrays_path, allow_pickle=False, **plotted)
    written.append(arrays_path)
    provenance_path = output_dir / f"{stem}-provenance.json"
    provenance_path.write_text(
        json.dumps(
            provenance
            | {
                "created_at": datetime.now(UTC).isoformat(),
                "suite": suite_name,
                "task_position": task_position,
                "task_index_zero_based": task_index,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_replicates": bootstrap_replicates,
                "confidence_intervals": (
                    "paired sequence-level pointwise percentile 95%; one shared "
                    "resample for all three methods and their differences"
                ),
                "observer_interpretation": (
                    "generator-aware random-feature GP references; approximate and not "
                    "certified lower bounds"
                ),
                "outputs": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )
    written.append(provenance_path)
    return written
