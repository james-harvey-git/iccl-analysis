"""Shared bootstrap and figure helpers for analysis scripts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


def paired_bootstrap_curves(
    left: np.ndarray,
    right: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Point estimates and pointwise percentile intervals for paired curves.

    Whole sequence rows are resampled. One index draw is reused for both
    methods and every demo position, preserving method pairing and within-curve
    dependence.
    """
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("paired bootstrap inputs must be [sequences, demos]")
    if left.shape[0] != right.shape[0]:
        raise ValueError("paired bootstrap inputs must contain the same sequences")
    if left.shape[0] == 0:
        raise ValueError("paired bootstrap needs at least one sequence")
    if replicates <= 0:
        raise ValueError(f"bootstrap replicates must be positive, got {replicates}")
    demos = min(left.shape[1], right.shape[1])
    left = left[:, :demos]
    right = right[:, :demos]
    difference = left - right
    point = np.stack(
        [np.nanmean(left, axis=0), np.nanmean(right, axis=0), np.nanmean(difference, axis=0)]
    )

    rng = np.random.default_rng(seed)
    bootstrap = np.empty((replicates, 3, demos), dtype=np.float64)
    for replicate in range(replicates):
        indices = rng.integers(0, left.shape[0], size=left.shape[0])
        bootstrap[replicate, 0] = np.nanmean(left[indices], axis=0)
        bootstrap[replicate, 1] = np.nanmean(right[indices], axis=0)
        bootstrap[replicate, 2] = np.nanmean(difference[indices], axis=0)
    lower, upper = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    names = ("left", "right", "difference")
    result: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        result[f"{name}_mean"] = point[index].astype(np.float32)
        result[f"{name}_ci_low"] = lower[index].astype(np.float32)
        result[f"{name}_ci_high"] = upper[index].astype(np.float32)
    return result


def _save_figure(figure: Figure, prefix: Path) -> list[Path]:
    paths = [prefix.with_suffix(".png"), prefix.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return paths


def _task_values_with_nan_padding(
    values: np.ndarray,
    counts: np.ndarray,
    task_index: int,
    demo_count: int,
) -> np.ndarray:
    selected = values[:, task_index, :demo_count].astype(np.float64)
    valid = np.arange(demo_count)[None, :] < counts[:, task_index, None]
    return np.where(valid, selected, np.nan)


def _comparison_figure(
    x: np.ndarray,
    pooled: dict[str, np.ndarray],
    strata: dict[str, tuple[int, dict[str, np.ndarray]]],
    *,
    ylabel: str,
    title: str,
) -> Figure:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    left = axes[0]
    for prefix, label, color in (
        ("left", "GDN", "#3366cc"),
        ("right", "Known-world per-task Bayes oracle", "#dc3912"),
    ):
        left.plot(x, pooled[f"{prefix}_mean"], label=label, color=color, linewidth=2)
        left.fill_between(
            x,
            pooled[f"{prefix}_ci_low"],
            pooled[f"{prefix}_ci_high"],
            color=color,
            alpha=0.2,
        )
    left.set(title=title, xlabel="Current-task demonstrations already observed", ylabel=ylabel)
    left.legend(frameon=False)
    left.grid(alpha=0.2)

    right = axes[1]
    colors = {"all_modules_seen": "#109618", "module_unseen": "#990099"}
    labels = {
        "all_modules_seen": "All current modules seen",
        "module_unseen": "At least one current module unseen",
    }
    for name, (sample_size, result) in strata.items():
        label = f"{labels[name]} (n={sample_size})"
        color = colors[name]
        right.plot(x, result["difference_mean"], label=label, color=color, linewidth=2)
        right.fill_between(
            x,
            result["difference_ci_low"],
            result["difference_ci_high"],
            color=color,
            alpha=0.2,
        )
    right.axhline(0.0, color="black", linewidth=1, alpha=0.7)
    right.set(
        title="Paired reference gap by prior module coverage",
        xlabel="Current-task demonstrations already observed",
        ylabel=f"GDN minus oracle {ylabel}",
    )
    right.legend(frameon=False)
    right.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def plot_bayes_oracle_comparison(
    gdn_nmse: np.ndarray,
    gdn_raw_mse: np.ndarray,
    oracle: dict[str, np.ndarray],
    *,
    suite_name: str,
    task_position: int,
    checkpoint_step: int,
    output_dir: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    include_raw_mse: bool,
    provenance: dict[str, Any],
) -> list[Path]:
    """Create comparison, posterior-diagnostic, and all-task gap figures."""
    if gdn_nmse.shape != oracle["nmse"].shape:
        raise ValueError(
            f"GDN/oracle NMSE shapes do not match: {gdn_nmse.shape} vs {oracle['nmse'].shape}"
        )
    if gdn_raw_mse.shape != oracle["raw_mse"].shape:
        raise ValueError("GDN/oracle raw-MSE shapes do not match")
    task_index = task_position - 1
    if not 0 <= task_index < gdn_nmse.shape[1]:
        raise ValueError(
            f"task_position is one-based and must be in [1, {gdn_nmse.shape[1]}], "
            f"got {task_position}"
        )
    demo_count = int(oracle["demo_counts"][:, task_index].max())
    x = np.arange(demo_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{suite_name}-task-{task_position}-step-{checkpoint_step:07d}"
    plotted: dict[str, np.ndarray] = {"demo_index": x, "task_positions": oracle["task_positions"]}
    written: list[Path] = []

    coverage = oracle["all_current_modules_seen"][:, task_index]
    masks = {"all_modules_seen": coverage, "module_unseen": ~coverage}

    def comparison(
        gdn: np.ndarray,
        reference: np.ndarray,
        *,
        metric: str,
        ylabel: str,
        seed_offset: int,
    ) -> None:
        pooled = paired_bootstrap_curves(
            gdn[:, task_index, :demo_count],
            reference[:, task_index, :demo_count],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + seed_offset,
        )
        strata: dict[str, tuple[int, dict[str, np.ndarray]]] = {}
        for offset, (name, mask) in enumerate(masks.items(), start=1):
            if not np.any(mask):
                continue
            strata[name] = (
                int(mask.sum()),
                paired_bootstrap_curves(
                    gdn[mask, task_index, :demo_count],
                    reference[mask, task_index, :demo_count],
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + seed_offset + offset,
                ),
            )
        figure = _comparison_figure(
            x,
            pooled,
            strata,
            ylabel=ylabel,
            title=f"{suite_name}: task position {task_position}",
        )
        written.extend(_save_figure(figure, output_dir / f"{stem}-{metric}"))
        for key, value in pooled.items():
            plotted[f"{metric}_pooled_{key}"] = value
        for name, (sample_size, result) in strata.items():
            plotted[f"{metric}_{name}_sample_size"] = np.array(sample_size, dtype=np.int32)
            for key, value in result.items():
                plotted[f"{metric}_{name}_{key}"] = value

    comparison(gdn_nmse, oracle["nmse"], metric="nmse", ylabel="NMSE", seed_offset=0)
    if include_raw_mse:
        comparison(
            gdn_raw_mse,
            oracle["raw_mse"],
            metric="raw-mse",
            ylabel="raw MSE",
            seed_offset=100,
        )

    counts = oracle["demo_counts"]
    diagnostic = {
        "survivor_count_before_mean": np.nanmean(
            _task_values_with_nan_padding(
                oracle["survivor_count_before"], counts, task_index, demo_count
            ),
            axis=0,
        ),
        "entropy_before_mean": np.nanmean(
            _task_values_with_nan_padding(oracle["entropy_before"], counts, task_index, demo_count),
            axis=0,
        ),
        "unique_identified_fraction_after": np.nanmean(
            _task_values_with_nan_padding(
                oracle["unique_identified_after"], counts, task_index, demo_count
            ),
            axis=0,
        ),
        "true_posterior_mass_after_mean": np.nanmean(
            _task_values_with_nan_padding(
                oracle["true_posterior_mass_after"], counts, task_index, demo_count
            ),
            axis=0,
        ),
        "predictive_covariance_trace_mean": np.nanmean(
            _task_values_with_nan_padding(
                oracle["predictive_covariance_trace"], counts, task_index, demo_count
            ),
            axis=0,
        ),
    }
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(x, diagnostic["survivor_count_before_mean"])
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Mean surviving hypotheses before prediction")
    axes[0, 1].plot(x, diagnostic["entropy_before_mean"])
    axes[0, 1].set_title("Mean posterior entropy before prediction")
    axes[1, 0].plot(x, diagnostic["unique_identified_fraction_after"], label="Unique")
    axes[1, 0].plot(x, diagnostic["true_posterior_mass_after_mean"], label="True-task mass")
    axes[1, 0].set_ylim(-0.02, 1.02)
    axes[1, 0].legend(frameon=False)
    axes[1, 0].set_title("Identification after observing the target")
    axes[1, 1].plot(x, diagnostic["predictive_covariance_trace_mean"])
    axes[1, 1].set_title("Mean predictive covariance trace")
    for axis in axes.flat:
        axis.set_xlabel("Current-task demonstrations already observed")
        axis.grid(alpha=0.2)
    figure.suptitle(f"Oracle diagnostics: {suite_name}, task position {task_position}")
    figure.tight_layout()
    written.extend(_save_figure(figure, output_dir / f"{stem}-diagnostics"))
    plotted.update({key: value.astype(np.float32) for key, value in diagnostic.items()})

    gap_heatmap = np.nanmean(gdn_nmse - oracle["nmse"], axis=0)
    figure, axis = plt.subplots(figsize=(10, 5))
    finite = np.abs(gap_heatmap[np.isfinite(gap_heatmap)])
    limit = float(finite.max()) if finite.size else 1.0
    if limit == 0.0:
        limit = 1.0
    image = axis.imshow(
        gap_heatmap,
        aspect="auto",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
        origin="lower",
    )
    axis.set(
        xlabel="Current-task demonstrations already observed",
        ylabel="Task position",
        title="Mean GDN minus oracle NMSE",
    )
    axis.set_yticks(np.arange(gap_heatmap.shape[0]), labels=np.arange(1, gap_heatmap.shape[0] + 1))
    figure.colorbar(image, ax=axis, label="NMSE difference")
    figure.tight_layout()
    written.extend(_save_figure(figure, output_dir / f"{stem}-gap-heatmap"))
    plotted["nmse_gap_heatmap"] = gap_heatmap.astype(np.float32)

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
                "confidence_intervals": "paired sequence-level pointwise percentile 95%",
                "outputs": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )
    written.append(provenance_path)
    return written
