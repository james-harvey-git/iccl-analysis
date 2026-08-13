"""Resumable feature/particle convergence calibration for the full observer."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from iccl.analysis.structured_observer.cache import (
    StructuredObserverSettings,
    generate_or_reuse_cache,
    load_cache,
    schedule_config_from_spec,
)
from iccl.analysis.structured_observer.kernel import FeatureBank
from iccl.analysis.structured_observer.runner import make_feature_bank_from_spec
from iccl.analysis.structured_observer.schedule import (
    StructuredSuiteSpec,
    canonical_task_classes,
)


def kernel_frobenius_convergence(
    suite: dict[str, np.ndarray],
    spec: StructuredSuiteSpec,
    settings: StructuredObserverSettings,
    feature_counts: tuple[int, ...],
    *,
    num_points: int = 32,
) -> np.ndarray:
    """Compare nested random-feature Gram matrices with the largest bank."""
    if tuple(sorted(feature_counts)) != feature_counts or len(set(feature_counts)) != len(
        feature_counts
    ):
        raise ValueError("feature_counts must be unique and increasing")
    maximum = feature_counts[-1]
    bank = make_feature_bank_from_spec(
        spec,
        num_features=maximum,
        seed=settings.kernel.seed,
        device=settings.device,
        dtype=settings.kernel.dtype,
    )
    count = min(num_points, int(suite["demo_counts"][0].sum()))
    positions: list[int] = []
    for task_index, demos in enumerate(suite["demo_counts"][0]):
        start = int(suite["task_spans"][0, task_index, 0])
        positions.extend((start + 2 * np.arange(int(demos))).tolist())
    selected = np.linspace(0, len(positions) - 1, count, dtype=np.int64)
    inputs = suite["tokens"][0, np.asarray(positions)[selected], : bank.input_dim]
    classes, _ = canonical_task_classes(schedule_config_from_spec(spec))
    latents = classes[np.arange(count) % len(classes)]

    grams: list[np.ndarray] = []
    for num_features in feature_counts:
        prefix = FeatureBank(
            module_weights=bank.module_weights[:num_features],
            module_biases=bank.module_biases[:num_features],
            scale=bank.scale,
            seed=bank.seed,
        )
        features = torch.stack(
            [prefix.features(x, z)[0] for x, z in zip(inputs, latents, strict=True)]
        )
        grams.append((features @ features.T).detach().cpu().numpy())
    reference = grams[-1]
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    return np.asarray(
        [np.linalg.norm(gram - reference) / denominator for gram in grams],
        dtype=np.float64,
    )


def convergence_statistics(
    curves_by_seed: np.ndarray,
    kernel_relative_frobenius: np.ndarray,
    maximum_relative_jitter: np.ndarray,
    *,
    kernel_threshold: float,
    curve_threshold: float,
    seed_standard_error_threshold: float,
    jitter_threshold: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Summarize convergence against the largest feature/particle setting."""
    if curves_by_seed.ndim != 5:
        raise ValueError(
            "curves_by_seed must have shape [feature counts, particle counts, seeds, "
            "sequences, demos]"
        )
    feature_levels, particle_levels, num_seeds, _, _ = curves_by_seed.shape
    if kernel_relative_frobenius.shape != (feature_levels,):
        raise ValueError("kernel convergence values do not match feature levels")
    if maximum_relative_jitter.shape != (feature_levels, particle_levels):
        raise ValueError("jitter values do not match the convergence grid")
    mean_curves = np.nanmean(curves_by_seed, axis=(2, 3))
    reference = mean_curves[-1, -1]
    curve_change = np.nanmax(np.abs(mean_curves - reference), axis=-1)
    seed_curves = np.nanmean(curves_by_seed[-1, -1], axis=1)
    seed_standard_error = np.std(
        seed_curves,
        axis=0,
        ddof=1 if num_seeds > 1 else 0,
    ) / np.sqrt(num_seeds)
    last_feature_change = (
        float(np.nanmax(np.abs(mean_curves[-2, -1] - reference)))
        if feature_levels > 1
        else 0.0
    )
    last_particle_change = (
        float(np.nanmax(np.abs(mean_curves[-1, -2] - reference)))
        if particle_levels > 1
        else 0.0
    )
    penultimate_kernel_error = (
        float(kernel_relative_frobenius[-2]) if feature_levels > 1 else 0.0
    )
    maximum_seed_standard_error = float(np.nanmax(seed_standard_error))
    observed_maximum_jitter = float(np.nanmax(maximum_relative_jitter))
    summary = {
        "kernel_relative_frobenius_penultimate_to_max": penultimate_kernel_error,
        "last_feature_curve_max_abs_change": last_feature_change,
        "last_particle_curve_max_abs_change": last_particle_change,
        "maximum_seed_standard_error": maximum_seed_standard_error,
        "maximum_relative_jitter": observed_maximum_jitter,
        "thresholds": {
            "kernel_relative_frobenius": kernel_threshold,
            "curve_max_abs_change": curve_threshold,
            "seed_standard_error": seed_standard_error_threshold,
            "relative_jitter": jitter_threshold,
        },
        "passes": {
            "kernel": penultimate_kernel_error < kernel_threshold,
            "feature_curve": last_feature_change < curve_threshold,
            "particle_curve": last_particle_change < curve_threshold,
            "seed_standard_error": (
                maximum_seed_standard_error < seed_standard_error_threshold
            ),
            "jitter": observed_maximum_jitter <= jitter_threshold,
        },
    }
    summary["passes"]["all"] = bool(all(summary["passes"].values()))
    arrays = {
        "mean_curves": mean_curves.astype(np.float32),
        "curve_max_abs_change_to_reference": curve_change.astype(np.float32),
        "reference_seed_mean_curves": seed_curves.astype(np.float32),
        "reference_seed_standard_error": seed_standard_error.astype(np.float32),
        "kernel_relative_frobenius_to_max": kernel_relative_frobenius.astype(np.float32),
        "maximum_relative_jitter": maximum_relative_jitter.astype(np.float32),
    }
    return arrays, summary


def _plot_convergence(
    arrays: dict[str, np.ndarray],
    feature_counts: tuple[int, ...],
    particle_counts: tuple[int, ...],
) -> Figure:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(
        feature_counts,
        arrays["kernel_relative_frobenius_to_max"],
        marker="o",
    )
    axes[0, 0].set(
        xlabel="Random features",
        ylabel="Relative Frobenius error",
        title="Kernel Gram convergence to largest bank",
    )
    image = axes[0, 1].imshow(
        arrays["curve_max_abs_change_to_reference"],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    axes[0, 1].set(
        xticks=np.arange(len(particle_counts)),
        xticklabels=particle_counts,
        yticks=np.arange(len(feature_counts)),
        yticklabels=feature_counts,
        xlabel="Particles",
        ylabel="Random features",
        title="Maximum curve change to largest setting",
    )
    figure.colorbar(image, ax=axes[0, 1], label="absolute NMSE change")
    axes[1, 0].plot(arrays["reference_seed_standard_error"])
    axes[1, 0].set(
        xlabel="Current-task demonstrations already observed",
        ylabel="Standard error",
        title="Reference-setting variation across SMC seeds",
    )
    jitter_image = axes[1, 1].imshow(
        arrays["maximum_relative_jitter"],
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axes[1, 1].set(
        xticks=np.arange(len(particle_counts)),
        xticklabels=particle_counts,
        yticks=np.arange(len(feature_counts)),
        yticklabels=feature_counts,
        xlabel="Particles",
        ylabel="Random features",
        title="Maximum relative jitter",
    )
    figure.colorbar(jitter_image, ax=axes[1, 1], label="relative jitter")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def run_convergence_calibration(
    *,
    suite: dict[str, np.ndarray],
    suite_path: Path,
    metadata_path: Path,
    spec: StructuredSuiteSpec,
    base_settings: StructuredObserverSettings,
    feature_counts: tuple[int, ...],
    particle_counts: tuple[int, ...],
    sequence_limit: int,
    cache_dir: Path,
    output_dir: Path,
    thresholds: dict[str, float],
) -> list[Path]:
    """Run or reuse every calibration cell and write aggregate evidence."""
    kernel_errors = kernel_frobenius_convergence(
        suite,
        spec,
        base_settings,
        feature_counts,
    )
    grid_curves: list[list[np.ndarray]] = []
    grid_jitter = np.empty((len(feature_counts), len(particle_counts)), dtype=np.float64)
    cache_paths: list[str] = []
    for feature_index, num_features in enumerate(feature_counts):
        feature_row: list[np.ndarray] = []
        for particle_index, num_particles in enumerate(particle_counts):
            settings = replace(
                base_settings,
                modes=("full_history",),
                sequence_limit=sequence_limit,
                kernel=replace(base_settings.kernel, num_features=num_features),
                smc=replace(base_settings.smc, num_particles=num_particles),
            )
            cache_path, _ = generate_or_reuse_cache(
                suite_path,
                metadata_path,
                cache_dir,
                settings,
            )
            cache, _ = load_cache(cache_path)
            feature_row.append(cache["full_nmse_by_seed"][:, :, -1])
            grid_jitter[feature_index, particle_index] = float(
                np.nanmax(cache["full_relative_jitter_by_seed"])
            )
            cache_paths.append(str(cache_path.resolve()))
        grid_curves.append(feature_row)
    curves = np.stack([np.stack(row) for row in grid_curves])
    arrays, summary = convergence_statistics(
        curves,
        kernel_errors,
        grid_jitter,
        kernel_threshold=thresholds["kernel_relative_frobenius"],
        curve_threshold=thresholds["curve_max_abs_change"],
        seed_standard_error_threshold=thresholds["seed_standard_error"],
        jitter_threshold=thresholds["relative_jitter"],
    )
    arrays["feature_counts"] = np.asarray(feature_counts, dtype=np.int32)
    arrays["particle_counts"] = np.asarray(particle_counts, dtype=np.int32)
    arrays["curves_by_seed"] = curves.astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "convergence-arrays.npz"
    np.savez_compressed(arrays_path, allow_pickle=False, **arrays)
    figure = _plot_convergence(arrays, feature_counts, particle_counts)
    png_path = output_dir / "convergence.png"
    pdf_path = output_dir / "convergence.pdf"
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    summary_path = output_dir / "convergence-provenance.json"
    summary_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "summary": summary,
                "feature_counts": feature_counts,
                "particle_counts": particle_counts,
                "sequence_limit": sequence_limit,
                "smc_seeds": base_settings.smc_seeds,
                "base_settings": asdict(base_settings),
                "cell_cache_paths": cache_paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return [arrays_path, png_path, pdf_path, summary_path]
