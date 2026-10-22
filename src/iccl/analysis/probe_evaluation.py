"""Held-out scoring and equal-world reports from immutable captures and selected probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

from iccl.analysis.probe_config import configure_runtime, resolved_config, validate_probe_config
from iccl.analysis.probe_execution import run_stage
from iccl.analysis.probe_features import FeatureSelection
from iccl.analysis.probe_results import (
    aggregate_exposure,
    aggregate_scores,
    exposure_scores,
    paired_control_scores,
    score_predictions,
    world_interval,
)
from iccl.analysis.probe_storage import (
    CapturedSplit,
    complete_artifact,
    file_digest,
    initialize_artifact,
    lock,
    read_artifact,
    read_json,
    read_manifest,
    read_world_artifact,
    save_npz,
    source_hashes,
    write_json,
)
from iccl.analysis.probe_tracking import WorkerEvents
from iccl.analysis.probe_training import check_dataset_config, checked_bundle, feature_matrix
from iccl.analysis.probe_worlds import stream_seed
from iccl.analysis.probes import predict_ridge


def behavior_scores(split: CapturedSplit) -> list[dict[str, Any]]:
    variance = split.metadata("target_variance").astype(np.float64).mean(axis=1)
    rows = []
    for model in split.manifest["identity"]["capture"]["models"]:
        for point, label in (("final_input", "ordinary_answer"), ("branch_input", "branch_answer")):
            predicted = split.metadata(f"{model}__{point}__prediction").astype(np.float64)
            truth = split.metadata(label).astype(np.float64)
            mse = ((predicted - truth) ** 2).mean(axis=1)
            valid = variance > 0
            rows.append(
                {
                    "world": split.world,
                    "position": split.position,
                    "population": split.split,
                    "model": model,
                    "capture": point,
                    "mse": float(mse.mean()),
                    "nmse": float((mse[valid] / variance[valid]).mean()) if valid.any() else None,
                    "undefined_normalizers": int((~valid).sum()),
                    "trials": len(mse),
                }
            )
    return rows


def aggregate_behavior(
    rows: list[dict[str, Any]], seed: int, replicates: int, confidence: float
) -> list[dict[str, Any]]:
    keys = ("population", "position", "model", "capture")
    conditions = sorted({tuple(r[k] for k in keys) for r in rows})
    result = []
    for key in conditions:
        subset = [r for r in rows if tuple(r[k] for k in keys) == key]
        for metric in ("mse", "nmse"):
            values = {r["world"]: r[metric] for r in subset if r[metric] is not None}
            estimate = world_interval(
                values,
                seed=stream_seed(seed, f"world-bootstrap/{sorted(values)}"),
                replicates=replicates,
                confidence=confidence,
            )
            result.append(dict(zip(keys, key, strict=False)) | {"metric": metric, **estimate})
    return result


def write_report(root: Path, summary: dict[str, Any]) -> None:
    primary = [
        r
        for r in summary["aggregate"]
        if r["metric"] == "r2_macro" and r["control"] == "trained" and r["selection"] == "all"
    ]
    lines = [
        "# Within-world task-vector decoding",
        "",
        "The question is whether a frozen GDN representation allows an affine linear "
        "readout of the current "
        "task's weighted constituent vector within a fixed teacher world. Each world has "
        "its own probes; "
        "world coordinate systems are never pooled. Decodability demonstrates accessible "
        "information, "
        "not that the model explicitly uses that representation to solve the task.",
        "",
        "Teacher modules and readout stay fixed within a world. Fresh streams vary the "
        "target, demonstrations "
        "and preceding tasks. Target supports are partitioned into training, validation "
        "and unseen-test sets; "
        "familiar-test supports use new streams from the training support set. All "
        "representations and capture "
        "points for a trial stay together. Background tasks use training supports and "
        "exclude the target pair.",
        "",
        "**Capture points.** Final input follows D−1 complete demonstrations plus the "
        "final query. Final answer "
        "includes all D demonstrations. Branch input is a fresh query from a copy of the "
        "complete post-answer "
        "cache, before the next boundary; the branch is discarded. Positions are one-"
        "based task indices.",
        "",
        "**Metrics.** R² is computed separately for every constituent coordinate and "
        "averaged within a world. "
        "A constant test coordinate makes that world's macro R² undefined; coordinates "
        "are not silently dropped. "
        "Exact-pair accuracy requires both top-two predicted coordinates to match the "
        "active pair (ties use "
        "constituent index order). Active-coefficient RMSE scores only the two nonzero "
        "target coefficients. "
        "Fixed-scale vector MSE divides each coordinate's squared error by its variance "
        "in the original full "
        "training set, then averages coordinates and trials; lower is better and its "
        "scale stays fixed across "
        "exposure cells. Behavioral nMSE divides each query's output MSE by the mean "
        "coordinate variance of "
        "that task's D demonstration answers, then averages trials.",
        "",
        "**Aggregation.** Figures average world-level scores with equal weight. Intervals"
        " resample whole worlds. "
        "Control differences use paired worlds. Coverage counts and undefined scores are "
        "retained in summary.json. "
        "No interval is shown for a single world.",
        "",
        "**Exposure.** Each active constituent's exposure counts preceding tasks "
        "containing that constituent. "
        "The current task, later tasks and discarded query do not count. Heatmap axes are"
        " the sorted counts "
        "(e_min, e_max), conditional on target position. Each panel uses the same "
        "qualifying world cohort "
        "across its supported cells; gray cells lack coverage and hatched cells are "
        "structurally impossible. "
        "These comparisons are observational: task composition and history differ between"
        " exposure cells.",
        "",
        f"Phase: **{summary['phase']}**. Worlds: **{summary['worlds']}**. "
        "Development results guide protocol choices; scientific confirmation requires "
        "fresh confirmation worlds.",
        "",
        "| Test population | Position | Capture | Representation | Mean R² | Worlds |",
        "|---|---:|---|---|---:|---:|",
    ]
    for row in primary:
        value = "undefined" if row["mean"] is None else f"{row['mean']:.3f}"
        lines.append(
            f"| {row['population']} | {row['position']} | {row['capture']} | "
            f"{row['family']} | {value} | {row['world_count']} |"
        )
    lines.extend(
        [
            "",
            "Interactive figures are in plots/index.html. Every figure can be regenerated"
            " from summary.json "
            "without loading the model or fitting probes. Per-world scores and held-out "
            "predictions remain available "
            "for numerical audit.",
            "",
            "<details><summary>Fitting, controls and provenance</summary>",
            "",
            "Features are standardized using training moments only. The intercept is "
            "unpenalized and lambda "
            "penalizes the standardized feature weights: ||XW + b − Z||²/n + lambda ||W||². "
            "Only converged candidates with defined full-validation macro R² can be "
            "selected. Validation ties "
            "choose the larger lambda. Test populations are read only after selection.",
            "",
            "Controls are the training target mean, training targets shuffled within "
            "world and position with the "
            "same permutation across representations/captures, an independently "
            "initialized GDN on the same full "
            "prefix, and a linear probe on the current task's available raw "
            "demonstration/query values. The raw "
            "observation control does not include nonlinear products of x and y.",
            "",
            "Monitor curves use fixed train/validation subsets and do not choose lambda; "
            "final candidate selection "
            "uses the full validation split. Nested training-budget curves also use "
            "validation, never test scores.",
            "",
            f"Dataset identity: `{summary['dataset_id']}`. "
            f"Probe identity: `{summary['probes_id']}`.",
            "",
            "Resolved settings, checkpoint digest, source-run identity, teacher arrays, "
            "support partitions, seeds, "
            "feature shapes and checksums are stored in the corresponding manifests. "
            "Bootstrap draws share seeds "
            "for identical world cohorts. Exposure thresholds and bins are recorded in "
            "this result's manifest.",
            "",
            "</details>",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines))


def evaluate_probe(cfg: DictConfig, out_dir: Path | str) -> Path:
    validate_probe_config(cfg, "eval")
    device = configure_runtime(cfg)
    e = cfg.probe.evaluation
    dataset = read_manifest(e.dataset)
    check_dataset_config(cfg, dataset)
    probes_root, root = Path(e.probes), Path(e.output)
    probes = read_artifact(probes_root, "probes")
    if probes["identity"]["dataset_id"] != dataset["dataset_id"]:
        raise ValueError("selected probes belong to another dataset")
    settings = resolved_config(cfg)["probe"]["evaluation"]
    settings = {k: v for k, v in settings.items() if k not in {"dataset", "probes", "output"}}
    artifact = initialize_artifact(
        root,
        "results",
        {
            "dataset_id": dataset["dataset_id"],
            "probes_id": probes["artifact_id"],
            "settings": settings,
            "seed": cfg.experiment.seed,
            "implementation": source_hashes("probe_evaluation.py", "probe_results.py"),
        },
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def worker(world: int, events: WorkerEvents) -> None:
        folder = root / f"world-{world:04d}"
        with lock(folder / ".evaluation.lock", blocking=False):
            path = folder / "complete.json"
            if path.exists():
                saved = read_json(path)
                if saved["artifact_id"] != artifact["artifact_id"]:
                    raise ValueError("evaluation world belongs to another result identity")
                for name, checksum in saved["predictions"].items():
                    if file_digest(root / name) != checksum:
                        raise ValueError("saved held-out prediction checksum mismatch")
                return
            trained = read_world_artifact(probes_root, probes, world)
            rows, exposures, behaviors, files, validation_rows = [], [], [], {}, []
            for item in trained["selected"]:
                validation_rows.append(
                    item["condition"] | {"population": "validation", "scores": item["validation"]}
                )
            for position in cfg.probe.dataset.target_positions:
                conditions = [
                    item
                    for item in trained["selected"]
                    if item["condition"]["position"] == position
                    and item["condition"]["train_count"] == cfg.probe.dataset.trajectories.train
                ]
                for population in e.splits:
                    split = CapturedSplit(e.dataset, world, position, population)
                    try:
                        targets, prior = split.targets, split.metadata("exposure")
                        behaviors.extend(behavior_scores(split))
                        for item in conditions:
                            condition = item["condition"] | {"population": population}
                            bundle = checked_bundle(
                                probes_root / item["path"] / "selected.pt", item["sha256"]
                            )
                            selection = (
                                None
                                if item["selection"] is None
                                else FeatureSelection(
                                    item["selection"]["family"],
                                    item["selection"]["name"],
                                    tuple(tuple(pair) for pair in item["selection"]["ranges"]),
                                )
                            )
                            matrix = feature_matrix(split, condition, selection)
                            predictions = predict_ridge(
                                bundle, matrix, device, cfg.probe.training.solver.row_batch_size
                            )
                            variance = np.asarray(item["training_variance"])
                            rows.append(
                                condition
                                | {"scores": score_predictions(targets, predictions, variance)}
                            )
                            if e.exposure.enabled:
                                bins = settings["exposure"]["count_bins"]
                                exposures.extend(
                                    condition | cell
                                    for cell in exposure_scores(
                                        targets, predictions, prior, variance, bins
                                    )
                                )
                            prediction_path = root / item["path"] / f"{population}.npz"
                            save_npz(
                                prediction_path,
                                {
                                    "predictions": predictions,
                                    "targets": targets,
                                    "prior_exposures": prior,
                                    "training_variance": variance,
                                },
                            )
                            files[str(prediction_path.relative_to(root))] = file_digest(
                                prediction_path
                            )
                    finally:
                        split.close()
            write_json(
                path,
                {
                    "artifact_id": artifact["artifact_id"],
                    "scores": rows,
                    "exposure": exposures,
                    "behavior": behaviors,
                    "validation": validation_rows,
                    "predictions": files,
                },
            )
            events.emit(
                "evaluation_complete", phase=cfg.experiment.phase, world=world, conditions=len(rows)
            )

    def finalize() -> dict[str, Any]:
        completed = artifact | {
            "worlds": {
                str(world): file_digest(root / f"world-{world:04d}" / "complete.json")
                for world in range(cfg.probe.dataset.num_worlds)
            }
        }
        worlds = [
            read_world_artifact(root, completed, world)
            for world in range(cfg.probe.dataset.num_worlds)
        ]
        rows = [row for world in worlds for row in world["scores"]]
        exposure_rows = [row for world in worlds for row in world["exposure"]]
        behavior = [row for world in worlds for row in world["behavior"]]
        validation_rows = [row for world in worlds for row in world["validation"]]
        uncertainty = {
            "seed": cfg.experiment.seed,
            "replicates": e.bootstrap_replicates,
            "confidence": e.confidence_level,
        }
        world_count = cfg.probe.dataset.num_worlds
        summary = {
            "phase": cfg.experiment.phase,
            "worlds": world_count,
            "dataset_id": dataset["dataset_id"],
            "probes_id": probes["artifact_id"],
            "dataset": dataset["identity"]["dataset"],
            "capture": dataset["identity"]["capture"],
            "settings": settings,
            "aggregate": aggregate_scores(rows, worlds=world_count, **uncertainty),
            "paired_controls": paired_control_scores(rows, **uncertainty),
            "exposure": aggregate_exposure(
                exposure_rows,
                worlds=world_count,
                **uncertainty,
                min_trials=e.exposure.min_trials_per_world_cell,
                min_worlds=e.exposure.min_worlds_per_cell,
            )
            if e.exposure.enabled
            else [],
            "behavior": aggregate_behavior(behavior, **uncertainty),
            "validation": aggregate_scores(validation_rows, worlds=world_count, **uncertainty),
            "world_files": [
                {
                    "world": int(world),
                    "path": f"world-{int(world):04d}/complete.json",
                    "sha256": checksum,
                }
                for world, checksum in completed["worlds"].items()
            ],
        }
        write_json(root / "summary.json", summary)
        write_report(root, summary)
        from iccl.analysis.plotting import plot_probe_results

        plot_probe_results(root / "summary.json", root / "plots")
        complete_artifact(root, "results", cfg.probe.dataset.num_worlds)
        return {"results_path": str(root.resolve()), "results_id": completed["artifact_id"]}

    run_stage(cfg, "eval", worker, finalize)
    return root / "summary.json"
