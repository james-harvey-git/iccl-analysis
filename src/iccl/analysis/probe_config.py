"""Configuration checks and execution policy for the module-decoder pipeline."""

import hashlib
import math
import os
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from iccl.analysis.probe_targets import PROTOCOL
from iccl.utils import resolve_device, seed_everything

SPLITS = ("train", "validation", "test")


def stream_seed(seed: int, namespace: str) -> int:
    """Stable upper-half uint64 namespace, independent of counts, workers and checkpoints.

    Current production training/evaluation seeds occupy the lower half of the
    uint64 space (32-bit root seeds plus offsets and 32-bit cell hashes).
    """
    if not 0 <= seed < 2**64:
        raise ValueError("stream seeds must fit uint64")
    key = f"{PROTOCOL}:{namespace}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") | (1 << 63)


def resolved_config(cfg: DictConfig) -> dict[str, Any]:
    result = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(result, dict):
        raise ValueError("expected a configuration mapping")
    return cast(dict[str, Any], dict(result))


def _integer(value: Any, name: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def validate_probe_config(cfg: DictConfig, stage: str) -> None:
    """Validate only the launch inputs needed for this stage, plus shared invariants."""
    if stage not in {"capture", "train", "eval", "benchmark"}:
        raise ValueError(f"unknown probe stage: {stage}")
    data, p = cfg.data, cfg.probe
    if (data.input_dim, data.output_dim, list(data.hidden_dims)) != (16, 16, [16]):
        raise ValueError("module decoding requires teacher dimensions 16 -> 16 -> 16")
    expected = {
        "curriculum_sampler": "constructive",
        "hotness": 2,
        "surplus_tasks": 1,
        "demos_per_task": 32,
        "signal_boundaries": True,
        "require_identifiable": True,
        "require_full_rank": False,
    }
    if data.num_modules != 8 or not data.use_bias or data.weighting != "discrete":
        raise ValueError("module decoding requires eight biased modules and discrete weights")
    if not math.isclose(float(data.scale), math.sqrt(3), rel_tol=1e-12):
        raise ValueError("teacher scale must be sqrt(3)")
    if any(data.sequence.get(key) != value for key, value in expected.items()):
        raise ValueError("probe episodes require the constructive M=T=8, D=32 contract")
    if data.sequence.get("phases") is not None:
        raise ValueError("fixed-phase curricula are not supported by this probe protocol")
    if not p.dataset.path:
        raise ValueError("probe.dataset.path is required")
    capture = stage == "capture" or (stage == "benchmark" and p.benchmark.capture_first)
    if capture and not p.capture.checkpoint:
        raise ValueError("probe.capture.checkpoint is required for capture")
    if stage == "eval" and not p.evaluation.checkpoint:
        raise ValueError("probe.evaluation.checkpoint is required for evaluation")
    for name, minimum in (("seed", 0), ("torch_num_threads", 1)):
        _integer(cfg[name], name, minimum)
    if cfg.seed >= 2**32:
        raise ValueError("seed must fit uint32 for the existing Python/NumPy/Torch seeding helper")
    if cfg.float32_matmul_precision not in {"highest", "high", "medium"}:
        raise ValueError("float32_matmul_precision must be highest, high or medium")
    if p.capture.backend not in {"auto", "fla", "reference"}:
        raise ValueError("capture backend must be auto, fla or reference")
    for group in ("capture", "training", "evaluation"):
        if p[group].precision not in {"auto", "fp32", "bf16"}:
            raise ValueError(f"invalid {group} precision")
        _integer(p[group].batch_size, f"{group}.batch_size")
    for group in ("capture", "training"):
        _integer(p[group].num_workers, f"{group}.num_workers", 0)
    if set(p.dataset.counts) != set(SPLITS):
        raise ValueError("dataset counts must specify train, validation and test")
    for split in SPLITS:
        _integer(p.dataset.counts[split], f"dataset.counts.{split}")
    _integer(p.dataset.seed, "dataset.seed", 0)
    stream_seed(p.dataset.seed, "validation")
    _integer(p.dataset.shard_size, "dataset.shard_size")
    _integer(p.solver.num_threads, "solver.num_threads")
    if p.solver.num_threads > 256:
        raise ValueError("at most 256 native solver threads are supported")
    weight = float(p.loss.readout_weight)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("readout_weight must be positive and finite")
    if p.training.control not in {"none", "constant", "shuffled_targets"}:
        raise ValueError("invalid training control")
    if stage == "benchmark" and p.training.control != "none":
        raise ValueError("the full-decoder benchmark requires training.control=none")
    for name in ("num_steps", "log_every", "validation_every", "checkpoint_every"):
        _integer(p.training[name], f"training.{name}")
    _integer(p.training.warmup_steps, "training.warmup_steps", 0)
    if p.training.optimizer != "adamw" or p.training.schedule not in {"constant", "cosine"}:
        raise ValueError("use AdamW with a constant or cosine schedule")
    if not math.isfinite(p.training.lr) or p.training.lr <= 0:
        raise ValueError("training.lr must be positive and finite")
    if not math.isfinite(p.training.weight_decay) or p.training.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if p.training.grad_clip is not None and (
        not math.isfinite(p.training.grad_clip) or p.training.grad_clip <= 0
    ):
        raise ValueError("grad_clip must be null or positive and finite")
    if p.evaluation.split not in {"validation", "test"}:
        raise ValueError("standalone evaluation must select validation or test")
    _integer(p.evaluation.functional_inputs_per_task, "evaluation.functional_inputs_per_task")
    _integer(p.evaluation.bootstrap_replicates, "evaluation.bootstrap_replicates", 0)
    for name in ("functional_seed", "bootstrap_seed"):
        _integer(p.evaluation[name], f"evaluation.{name}", 0)
        stream_seed(p.evaluation[name], name)
    for name in ("warmup_steps", "profile_steps"):
        _integer(p.benchmark[name], f"benchmark.{name}", 0)
    _integer(p.benchmark.measured_steps, "benchmark.measured_steps")
    if cfg.wandb.mode not in {"disabled", "offline", "online"}:
        raise ValueError("invalid W&B mode")
    workers = p.capture.num_workers if stage == "capture" else p.training.num_workers
    required = cfg.torch_num_threads + workers + (0 if stage == "capture" else p.solver.num_threads)
    allocation = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocation and required > int(allocation):
        raise ValueError(
            f"probe CPU settings request {required} cores but SLURM allocated {allocation}; "
            "reduce solver/loader/PyTorch threads or request more cores"
        )


def configure_runtime(cfg: DictConfig) -> torch.device:
    torch.set_num_threads(cfg.torch_num_threads)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    seed_everything(cfg.seed)
    return resolve_device(cfg.device)
