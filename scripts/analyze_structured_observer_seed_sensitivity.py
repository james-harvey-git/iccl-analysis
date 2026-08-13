"""Analyze Monte Carlo sensitivity of a merged structured-observer cache."""

from __future__ import annotations

import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig

from iccl.analysis.bayes_oracle import file_sha256, load_suite_metadata, suite_paths
from iccl.analysis.checkpoints import resolve_checkpoint_path
from iccl.analysis.structured_observer.cache import load_cache
from iccl.analysis.structured_observer.plotting import paired_bootstrap_methods
from iccl.data.export import load_suite
from iccl.models.model import model_from_config
from iccl.training.logger import source_from_checkpoint
from iccl.training.metrics import BASE_MSE_FLOOR, demo_nmse, predict_suite
from iccl.training.trainer import resolve_autocast_dtype
from iccl.utils import resolve_device, seed_everything


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _prediction_nmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    base_mse: np.ndarray,
    demo_counts: np.ndarray,
) -> np.ndarray:
    """Evaluate prediction arrays using the frozen-suite NMSE convention."""
    raw_mse = np.mean((predictions - targets) ** 2, axis=-1)
    denominator = np.maximum(base_mse.mean(axis=-1), BASE_MSE_FLOOR)
    nmse = raw_mse / denominator[:, :, None]
    valid = np.arange(raw_mse.shape[-1])[None, None, :] < demo_counts[:, :, None]
    return np.where(valid, nmse, np.nan)


def seed_sensitivity_statistics(
    gdn_nmse: np.ndarray,
    cache: dict[str, np.ndarray],
    *,
    task_index: int,
    demo_count: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, np.ndarray]:
    """Compute individual, cumulative, and subset-ensemble SMC comparisons."""
    required = {
        "base_mse",
        "demo_counts",
        "full_nmse",
        "full_nmse_by_seed",
        "full_predictions_by_seed",
        "sequence_indices",
        "smc_seeds",
        "targets",
    }
    missing = sorted(required - cache.keys())
    if missing:
        raise ValueError(f"structured cache is missing arrays: {', '.join(missing)}")
    seed_predictions = cache["full_predictions_by_seed"].astype(np.float64)
    num_seeds = seed_predictions.shape[0]
    if num_seeds < 2:
        raise ValueError("seed-sensitivity analysis requires at least two SMC seeds")
    sequence_indices = cache["sequence_indices"].astype(np.int64)
    gdn_task = gdn_nmse[sequence_indices, task_index, :demo_count].astype(np.float64)
    individual_nmse = cache["full_nmse_by_seed"][
        :, :, task_index, :demo_count
    ].astype(np.float64)
    targets = cache["targets"].astype(np.float64)
    base_mse = cache["base_mse"].astype(np.float64)
    demo_counts = cache["demo_counts"].astype(np.int64)

    cumulative_nmse: list[np.ndarray] = []
    cumulative_predictions: list[np.ndarray] = []
    gap_mean: list[np.ndarray] = []
    gap_low: list[np.ndarray] = []
    gap_high: list[np.ndarray] = []
    subset_min: list[np.ndarray] = []
    subset_max: list[np.ndarray] = []
    subset_mean: list[np.ndarray] = []
    subset_counts: list[int] = []

    for seed_count in range(1, num_seeds + 1):
        cumulative_prediction = np.mean(seed_predictions[:seed_count], axis=0)
        cumulative_predictions.append(cumulative_prediction)
        cumulative_metric = _prediction_nmse(
            cumulative_prediction,
            targets,
            base_mse,
            demo_counts,
        )[:, task_index, :demo_count]
        cumulative_nmse.append(cumulative_metric)
        bootstrap = paired_bootstrap_methods(
            {"gdn": gdn_task, "ensemble": cumulative_metric},
            differences=(("gdn", "ensemble"),),
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + seed_count,
        )
        gap_mean.append(bootstrap["gdn_minus_ensemble_mean"])
        gap_low.append(bootstrap["gdn_minus_ensemble_ci_low"])
        gap_high.append(bootstrap["gdn_minus_ensemble_ci_high"])

        subset_curves: list[np.ndarray] = []
        for subset in combinations(range(num_seeds), seed_count):
            prediction = np.mean(seed_predictions[np.asarray(subset)], axis=0)
            metric = _prediction_nmse(
                prediction,
                targets,
                base_mse,
                demo_counts,
            )[:, task_index, :demo_count]
            subset_curves.append(np.nanmean(metric, axis=0))
        stacked_subsets = np.stack(subset_curves)
        subset_min.append(np.min(stacked_subsets, axis=0))
        subset_max.append(np.max(stacked_subsets, axis=0))
        subset_mean.append(np.mean(stacked_subsets, axis=0))
        subset_counts.append(len(subset_curves))

    cumulative_stack = np.stack(cumulative_predictions)
    prediction_change = np.full(
        (num_seeds, demo_count),
        np.nan,
        dtype=np.float64,
    )
    for seed_index in range(1, num_seeds):
        delta = cumulative_stack[seed_index] - cumulative_stack[seed_index - 1]
        prediction_change[seed_index] = np.sqrt(
            np.mean(delta[:, task_index, :demo_count] ** 2, axis=(0, -1))
        )

    reconstructed_full = cumulative_nmse[-1]
    cached_full = cache["full_nmse"][:, task_index, :demo_count]
    if not np.allclose(reconstructed_full, cached_full, rtol=1e-5, atol=1e-6):
        raise ValueError("reconstructed all-seed ensemble does not match cached full NMSE")

    return {
        "demo_index": np.arange(demo_count, dtype=np.int64),
        "smc_seeds": cache["smc_seeds"].astype(np.int64),
        "seed_count": np.arange(1, num_seeds + 1, dtype=np.int64),
        "subset_count": np.asarray(subset_counts, dtype=np.int64),
        "gdn_nmse_mean": np.nanmean(gdn_task, axis=0).astype(np.float32),
        "individual_seed_nmse_mean": np.nanmean(individual_nmse, axis=1).astype(
            np.float32
        ),
        "cumulative_ensemble_nmse_mean": np.stack(
            [np.nanmean(metric, axis=0) for metric in cumulative_nmse]
        ).astype(np.float32),
        "subset_ensemble_nmse_mean": np.stack(subset_mean).astype(np.float32),
        "subset_ensemble_nmse_min": np.stack(subset_min).astype(np.float32),
        "subset_ensemble_nmse_max": np.stack(subset_max).astype(np.float32),
        "gdn_minus_cumulative_mean": np.stack(gap_mean).astype(np.float32),
        "gdn_minus_cumulative_ci_low": np.stack(gap_low).astype(np.float32),
        "gdn_minus_cumulative_ci_high": np.stack(gap_high).astype(np.float32),
        "cumulative_prediction_rms_change": prediction_change.astype(np.float32),
    }


def _seed_sensitivity_figure(statistics: dict[str, np.ndarray]) -> Figure:
    x = statistics["demo_index"]
    seeds = statistics["smc_seeds"]
    seed_counts = statistics["seed_count"]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(seed_counts)))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    for seed_index, seed in enumerate(seeds):
        axes[0].plot(
            x,
            statistics["individual_seed_nmse_mean"][seed_index],
            linewidth=1.5,
            alpha=0.75,
            label=f"SMC seed {seed}",
        )
    axes[0].plot(
        x,
        statistics["cumulative_ensemble_nmse_mean"][-1],
        color="#109618",
        linewidth=2.5,
        label=f"{len(seeds)}-seed ensemble",
    )
    axes[0].plot(
        x,
        statistics["gdn_nmse_mean"],
        color="#3366cc",
        linewidth=2.5,
        label="GDN",
    )
    axes[0].set(title="Individual SMC seeds", ylabel="NMSE")
    axes[0].legend(frameon=False, fontsize=9)

    for index, seed_count in enumerate(seed_counts):
        axes[1].fill_between(
            x,
            statistics["subset_ensemble_nmse_min"][index],
            statistics["subset_ensemble_nmse_max"][index],
            color=colors[index],
            alpha=0.12,
        )
        axes[1].plot(
            x,
            statistics["cumulative_ensemble_nmse_mean"][index],
            color=colors[index],
            linewidth=2,
            label=f"First {seed_count} seed{'s' if seed_count > 1 else ''}",
        )
    axes[1].plot(
        x,
        statistics["gdn_nmse_mean"],
        color="#3366cc",
        linewidth=2.5,
        label="GDN",
    )
    axes[1].set(title="Cumulative seed ensembles", ylabel="NMSE")
    axes[1].legend(frameon=False, fontsize=9)

    for index, seed_count in enumerate(seed_counts):
        axes[2].plot(
            x,
            statistics["gdn_minus_cumulative_mean"][index],
            color=colors[index],
            linewidth=2,
            label=f"GDN minus {seed_count}-seed ensemble",
        )
        axes[2].fill_between(
            x,
            statistics["gdn_minus_cumulative_ci_low"][index],
            statistics["gdn_minus_cumulative_ci_high"][index],
            color=colors[index],
            alpha=0.12,
        )
    axes[2].axhline(0.0, color="black", linewidth=1, alpha=0.7)
    axes[2].set(title="Paired learned-algorithm gap", ylabel="GDN minus ensemble NMSE")
    axes[2].legend(frameon=False, fontsize=9)

    for axis in axes:
        axis.set_xlabel("Current-task demonstrations already observed")
        axis.grid(alpha=0.2)
    figure.suptitle("Final-task sensitivity to structured-observer SMC seeds")
    figure.tight_layout()
    return figure


def _convergence_figure(statistics: dict[str, np.ndarray]) -> Figure:
    x = statistics["demo_index"]
    seed_counts = statistics["seed_count"]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for index in range(1, len(seed_counts)):
        axis.plot(
            x,
            statistics["cumulative_prediction_rms_change"][index],
            linewidth=2,
            label=f"{seed_counts[index - 1]} to {seed_counts[index]} seeds",
        )
    axis.set(
        title="Change in posterior prediction as SMC seeds are added",
        xlabel="Current-task demonstrations already observed",
        ylabel="RMS prediction change",
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def _save_figure(figure: Figure, prefix: Path) -> list[Path]:
    paths = [prefix.with_suffix(".png"), prefix.with_suffix(".pdf")]
    for path in paths:
        figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return paths


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.training.resume is None:
        raise ValueError("pass an explicit checkpoint with training.resume=<reference>")
    explicit_cache = cfg.structured_observer.get("cache_path")
    if explicit_cache is None:
        raise ValueError("pass the merged cache with structured_observer.cache_path=<path>")
    seed_everything(cfg.seed)
    suite_name = str(cfg.structured_observer.suite)
    suite_path, metadata_path = suite_paths(
        Path(cfg.data.eval_sets.out_dir),
        suite_name,
        cfg.structured_observer.get("suite_path"),
    )
    cache_path = Path(str(explicit_cache))
    cache, cache_metadata = load_cache(cache_path)
    identity = cache_metadata["identity"]
    if identity["suite_sha256"] != file_sha256(suite_path):
        raise ValueError("structured cache and frozen suite arrays do not match")
    if identity["metadata_sha256"] != file_sha256(metadata_path):
        raise ValueError("structured cache and frozen suite metadata do not match")

    suite = load_suite(suite_path.with_suffix(""))
    checkpoint_reference = str(cfg.training.resume)
    checkpoint_path, is_artifact = resolve_checkpoint_path(checkpoint_reference)
    device = resolve_device(cfg.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions = predict_suite(
        model,
        suite,
        device,
        autocast_dtype=resolve_autocast_dtype(cfg.training.precision, device),
    )
    gdn_nmse = demo_nmse(predictions, suite)
    task_position = int(cfg.structured_observer.plotting.task_position)
    task_index = task_position - 1
    demo_count = int(cache["demo_counts"][:, task_index].max())
    statistics = seed_sensitivity_statistics(
        gdn_nmse,
        cache,
        task_index=task_index,
        demo_count=demo_count,
        bootstrap_replicates=int(
            cfg.structured_observer.plotting.bootstrap_replicates
        ),
        bootstrap_seed=int(cfg.structured_observer.plotting.bootstrap_seed),
    )

    step = int(checkpoint["step"])
    step_label = f"{step // 1000}k" if step % 1000 == 0 else str(step)
    output_dir = (
        Path(cfg.structured_observer.plotting.output_dir)
        / f"pilot-{step_label}-steps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{suite_name}-task-{task_position}-seed-sensitivity"
    written = _save_figure(
        _seed_sensitivity_figure(statistics),
        output_dir / stem,
    )
    written.extend(
        _save_figure(
            _convergence_figure(statistics),
            output_dir / f"{stem}-prediction-convergence",
        )
    )
    arrays_path = output_dir / f"{stem}-arrays.npz"
    np.savez_compressed(arrays_path, allow_pickle=False, **statistics)
    written.append(arrays_path)

    source = source_from_checkpoint(checkpoint)
    source_record: dict[str, Any] | None = None
    if source is not None:
        source_record = {
            "entity": source.entity,
            "project": source.project,
            "run_id": source.run_id,
            "name": source.name,
            "step": source.step,
            "url": source.url,
        }
    provenance_path = output_dir / f"{stem}-provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "checkpoint_reference": checkpoint_reference,
                "resolved_checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_from_wandb_artifact": is_artifact,
                "checkpoint_step": step,
                "source_run": source_record,
                "suite_path": str(suite_path.resolve()),
                "suite_metadata": load_suite_metadata(metadata_path),
                "structured_observer_cache_path": str(cache_path.resolve()),
                "structured_observer_cache_identity": identity,
                "smc_seeds": statistics["smc_seeds"].tolist(),
                "task_position": task_position,
                "bootstrap_replicates": int(
                    cfg.structured_observer.plotting.bootstrap_replicates
                ),
                "bootstrap_seed": int(cfg.structured_observer.plotting.bootstrap_seed),
                "confidence_intervals": (
                    "paired sequence-level pointwise percentile 95%, conditional on "
                    "each fixed cumulative SMC seed ensemble"
                ),
                "subset_bands": (
                    "minimum and maximum sequence-mean NMSE over every seed subset "
                    "of the stated size; not confidence intervals"
                ),
                "ensemble_semantics": (
                    "average posterior predictions across the first k ordered SMC "
                    "seeds, then evaluate squared error"
                ),
                "analysis_git_commit": _git_commit(),
                "outputs": [path.name for path in written],
            },
            indent=2,
            sort_keys=True,
        )
    )
    written.append(provenance_path)
    print("wrote structured-observer seed-sensitivity outputs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
