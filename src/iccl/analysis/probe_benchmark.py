"""Bounded resource measurements around real production decoder optimizer updates."""

from __future__ import annotations

import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from iccl.analysis.capture import capture_dataset
from iccl.analysis.probe_config import resolved_config, validate_probe_config
from iccl.analysis.probe_dataset import write_json
from iccl.analysis.probe_training import ProbeTrainer, synchronize


def _distribution(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "max": max(values),
    }


def resource_report(trainer: ProbeTrainer) -> dict[str, Any]:
    device = trainer.device
    usage = resource.getrusage(resource.RUSAGE_SELF)
    root = trainer.train.root
    storage = os.statvfs(root)
    affinity_reader = getattr(os, "sched_getaffinity", None)
    gpu: dict[str, Any] = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    if device.type == "cuda":
        gpu.update(
            name=torch.cuda.get_device_name(device),
            total_bytes=torch.cuda.get_device_properties(device).total_memory,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    elif device.type == "mps":
        gpu.update(
            name="Apple MPS",
            current_allocated_bytes=torch.mps.current_allocated_memory(),
            current_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
        )
    return {
        "device": str(device),
        "gpu": gpu,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(affinity_reader(0)) if affinity_reader is not None else None,
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "solver_threads": trainer.solver.num_threads,
        "loader_workers": int(trainer.p.training.num_workers),
        "host_peak_rss_bytes": int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
        "host_memory_scope": "parent-process high-water RSS; excludes separate loader workers",
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "cuda_runtime": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "dataset_path": str(root),
        "storage_cache": (
            "shards were checksum-verified before timing; OS page cache is not evicted"
        ),
        "dataset_device_id": root.stat().st_dev,
        "filesystem_block_bytes": storage.f_frsize,
        "filesystem_free_bytes": storage.f_bavail * storage.f_frsize,
        "dataset_file_bytes": sum(
            file.stat().st_size for file in root.rglob("*") if file.is_file()
        ),
    }


def _window(trainer: ProbeTrainer, steps: int, *, profile: bool) -> dict[str, Any]:
    start_step = trainer.step
    records = []
    synchronize(trainer.device)
    started = time.perf_counter()
    for _ in range(steps):
        tick = time.perf_counter()
        batch = trainer.next_batch()
        data_wait = time.perf_counter() - tick
        result = trainer.update(batch, profile=profile)
        synchronize(trainer.device)
        seconds = time.perf_counter() - tick
        record = {
            "step": trainer.step,
            "episodes": result.episodes,
            "seconds": seconds,
            "loss": result.loss,
            "gradient_norm": result.grad_norm,
            "loss_and_gradient_finite": True,
            "matching_nodes_mean": float(result.assignment.counters[:, 1].mean()),
            "matching_nodes_max": int(result.assignment.counters[:, 1].max()),
            "hungarian_solves_mean": float(result.assignment.counters[:, 0].mean()),
        }
        if profile:
            record["stages_seconds"] = {"data_wait": data_wait, **result.stages}
            record["native_worker_seconds"] = {
                "cost_construction_sum": float(result.assignment.timings[:, 0].sum()),
                "search_sum": float(result.assignment.timings[:, 1].sum()),
            }
        records.append(record)
    synchronize(trainer.device)
    elapsed = time.perf_counter() - started
    episodes = sum(record["episodes"] for record in records)
    stage_names = records[0]["stages_seconds"].keys() if records and profile else []
    return {
        "start_step_exclusive": start_step,
        "end_step_inclusive": trainer.step,
        "updates": steps,
        "episodes": episodes,
        "wall_seconds": elapsed,
        "episodes_per_second": episodes / elapsed if steps else None,
        "seconds_per_update": elapsed / steps if steps else None,
        "update_seconds_distribution": _distribution([row["seconds"] for row in records]),
        "stage_seconds_distributions": {
            name: _distribution([row["stages_seconds"][name] for row in records])
            for name in stage_names
        },
        "records": records,
    }


def benchmark_probe(cfg: DictConfig, out_dir: Path | str) -> dict[str, Any]:
    validate_probe_config(cfg, "benchmark")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    captured = (
        capture_dataset(cfg, out_dir / "capture") if cfg.probe.benchmark.capture_first else None
    )
    with ProbeTrainer(cfg, out_dir, stage="benchmark") as trainer:
        trainer.logger.start()
        b = cfg.probe.benchmark
        warmup = _window(trainer, b.warmup_steps, profile=False)
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)
        measured = _window(trainer, b.measured_steps, profile=False)
        measured_resources = resource_report(trainer)
        profiled = _window(trainer, b.profile_steps, profile=True)
        parameters = sum(p.numel() for p in trainer.model.parameters())
        report = {
            "protocol": "module-decoder-v1/benchmark-v1",
            "config": resolved_config(cfg),
            "dataset_id": trainer.train.manifest["dataset_id"],
            "train_split_signature": trainer.train.signature,
            "source_model_digest": trainer.train.manifest["identity"]["source_model_digest"],
            "state_layout": trainer.train.manifest["identity"]["state_layout"],
            "input_features": trainer.train.manifest["identity"]["input_features"],
            "parameters": parameters,
            "reference_parameter_count": parameters == 603984384,
            "decoder": "full_affine",
            "batch_size": int(cfg.probe.training.batch_size),
            "precision": str(trainer.dtype or torch.float32),
            "solver_build": trainer.solver.build,
            "capture": captured,
            "setup_seconds": trainer.timings,
            "warmup": warmup,
            "measured": measured,
            "profiled": profiled,
            "resources_after_measured": measured_resources,
            "resources_after_profiled": resource_report(trainer),
            "timing_scope": {
                "measured": (
                    "complete optimizer updates, including data wait, transfers, matching "
                    "and finite-gradient checks"
                ),
                "update_boundary": "device synchronized at each completed update",
                "profiled": (
                    "separate later updates with per-stage synchronization; "
                    "instrumentation changes throughput"
                ),
                "native_worker_times": (
                    "summed CPU worker times overlap; do not add them to wall-time stages"
                ),
                "excluded": [
                    "capture",
                    "compiler/build/load",
                    "checkpoint loading",
                    "validation",
                    "checkpoint writes",
                    "control fitting",
                ],
                "checkpoint_write_seconds": 0,
            },
            "planning_bytes": {
                "fp32_weights": parameters * 4,
                "weights_gradients_adam_moments": parameters * 16,
            },
        }
        write_json(out_dir / "benchmark.json", report)
        trainer.logger.log(
            {
                "probe/benchmark/episodes_per_second": measured["episodes_per_second"],
                "probe/benchmark/seconds_per_update": measured["seconds_per_update"],
            },
            trainer.step,
        )
        trainer.logger.upload_probe_artifact(out_dir / "benchmark.json", kind="results")
        return report
