"""Scientific contract validation and runtime settings for within-world task-vector probes."""

import math
import os
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from iccl.analysis.probe_worlds import CAPTURES, SPLITS, stream_seed
from iccl.utils import resolve_device, seed_everything


def resolved_config(cfg: DictConfig) -> dict[str, Any]:
    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


def positive(value: Any, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_probe_config(cfg: DictConfig, stage: str) -> None:
    if stage not in {"capture", "train", "eval", "benchmark"}:
        raise ValueError(f"unknown probe stage {stage}")
    p, experiment = cfg.probe, cfg.experiment
    d, t, e = p.dataset, p.training, p.evaluation
    if experiment.phase not in {"development", "confirmation"}:
        raise ValueError("phase must be development or confirmation")
    if experiment.phase == "confirmation" and not experiment.protocol_frozen:
        raise ValueError("freeze the development protocol before confirmation")
    stream_seed(experiment.seed, "validation")
    for name in (
        "num_worlds",
        "modules_per_world",
        "tasks_per_episode",
        "demos_per_task",
        "shard_size",
    ):
        positive(d[name], f"dataset.{name}")
    if d.active_modules_per_task != 2 or d.weighting != "discrete" or d.demos_per_task < 2:
        raise ValueError(
            "task-vector protocol requires two active constituents and discrete weights"
        )
    if d.tasks_per_episode - 1 < math.ceil(d.modules_per_world / 2):
        raise ValueError("background schedule cannot cover the module pool")
    if not d.target_positions or len(set(d.target_positions)) != len(d.target_positions):
        raise ValueError("target positions must be nonempty and unique")
    for position in d.target_positions:
        positive(position, "target position")
        if position > d.tasks_per_episode:
            raise ValueError("target position exceeds the episode length")
    if set(d.trajectories) != set(SPLITS):
        raise ValueError("expected train, validation and two held-out test populations")
    for name in SPLITS:
        positive(d.trajectories[name], f"trajectories.{name}")
    counts = d.constituent_pair_split
    if set(counts) != {"train", "validation", "test_unseen"}:
        raise ValueError("invalid constituent-pair partitions")
    if sum(counts.values()) != math.comb(d.modules_per_world, 2):
        raise ValueError("pair partitions must exhaust the two-module supports")
    for split in SPLITS:
        pool = "train" if split == "test_familiar" else split
        if d.trajectories[split] < counts[pool]:
            raise ValueError(f"{split} needs at least one complete target-pair cycle")
    if tuple(d.capture.points) != CAPTURES:
        raise ValueError("capture exactly final_input, final_answer and branch_input")
    if not set(d.capture.models) <= {"trained", "untrained"} or "trained" not in d.capture.models:
        raise ValueError("capture models must include trained and optionally untrained")
    if len(set(d.capture.models)) != len(d.capture.models):
        raise ValueError("duplicate capture models")
    if d.capture.backend not in {"auto", "reference", "fla"} or d.capture.precision not in {
        "auto",
        "fp32",
        "bf16",
    }:
        raise ValueError("unsupported capture backend/precision")
    if d.capture.storage_dtype != "float32":
        raise ValueError("task-vector feature storage must be float32")
    positive(d.capture.batch_size, "capture.batch_size")
    positive(d.capture.untrained_seed, "capture.untrained_seed", 0)
    if stage in {"capture", "benchmark"} and not d.checkpoint:
        raise ValueError("dataset.checkpoint is required for capture")
    if not t.fit_intercept or not t.standardize_features:
        raise ValueError(
            "the agreed ridge protocol requires an intercept and training-only standardization"
        )
    if (
        not t.l2_grid
        or len(set(t.l2_grid)) != len(t.l2_grid)
        or any(not math.isfinite(x) or x <= 0 for x in t.l2_grid)
    ):
        raise ValueError("ridge regularizers must be distinct finite positive values")
    if dict(t.selection) != {"metric": "r2_macro", "mode": "max", "split": "validation"}:
        raise ValueError("select regularization using full-validation macro R2 only")
    if not set(t.controls) <= {"training_mean", "shuffled_targets", "untrained", "observations"}:
        raise ValueError("unknown probe control")
    if "untrained" in t.controls and "untrained" not in d.capture.models:
        raise ValueError("untrained control requires untrained captures")
    if "observations" in t.controls and not d.capture.store_observations:
        raise ValueError("observation control requires stored observations")
    if experiment.phase != "development" and t.sample_sizes:
        raise ValueError("sample-size diagnostics belong to development worlds")
    for count in t.sample_sizes:
        positive(count, "training sample size", d.modules_per_world + 1)
        if count > d.trajectories.train:
            raise ValueError("sample-size diagnostic exceeds the captured training population")
    if t.solver.method not in {"auto", "direct", "cg"} or t.solver.dtype not in {
        "auto",
        "float32",
        "float64",
    }:
        raise ValueError("unsupported ridge solver/dtype")
    for name in ("direct_max_dimension", "max_working_mb", "max_iterations", "row_batch_size"):
        positive(t.solver[name], f"solver.{name}")
    if not 0 < t.solver.relative_tolerance < 1:
        raise ValueError("relative tolerance must lie in (0, 1)")
    for name, value in t.monitoring.items():
        positive(value, f"monitoring.{name}")
    if not e.splits or not set(e.splits) <= {"test_familiar", "test_unseen"}:
        raise ValueError("evaluation reads only the declared held-out test populations")
    if e.bootstrap_unit != "world" or not 0 < e.confidence_level < 1:
        raise ValueError("confidence intervals must resample whole worlds")
    positive(e.bootstrap_replicates, "bootstrap_replicates")
    if e.exposure.enabled:
        bins = e.exposure.count_bins
        if bins != "exact" and (
            not bins
            or list(bins) != sorted(set(bins))
            or bins[0] != 0
            or any(type(x) is not int for x in bins)
        ):
            raise ValueError(
                "exposure bins must be exact or sorted integer lower edges starting at zero"
            )
        for name in ("min_trials_per_world_cell", "min_worlds_per_cell"):
            positive(e.exposure[name], f"exposure.{name}")
    positive(cfg.execution.max_gpus, "max_gpus")
    if cfg.execution.max_gpus > 8:
        raise ValueError("this study is capped at eight concurrent GPUs")
    positive(cfg.execution.stage_timeout_seconds, "stage_timeout_seconds")
    positive(cfg.torch_num_threads, "torch_num_threads")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and cfg.torch_num_threads > int(allocated):
        raise ValueError("PyTorch threads exceed the Slurm CPU allocation")
    if cfg.wandb.mode not in {"online", "offline", "disabled"}:
        raise ValueError("invalid W&B mode")
    if cfg.wandb.upload_activations:
        raise ValueError(
            "bulk activation upload is outside this pipeline; keep feature shards on scratch"
        )
    for value in (d.path, t.output, e.output):
        if not value or str(Path(value)) == ".":
            raise ValueError("explicit dataset, probe and result directories are required")
    for name in ("max_worlds", "max_trials_per_split", "measured_repeats"):
        positive(p.benchmark[name], f"benchmark.{name}")
    positive(p.benchmark.warmup_repeats, "benchmark.warmup_repeats", 0)


def configure_runtime(cfg: DictConfig) -> torch.device:
    torch.set_num_threads(cfg.torch_num_threads)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    seed_everything(cfg.experiment.seed)
    device = resolve_device(cfg.device)
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1 and (
        device.type != "cuda" or torch.cuda.device_count() != 1
    ):
        raise ValueError(
            "multi-worker stages require srun to bind exactly one visible GPU per rank"
        )
    return device
