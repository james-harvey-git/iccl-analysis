"""Explicitly bounded end-to-end pilot and repeated full-feature ridge timing."""

from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from iccl.analysis.capture import capture_dataset
from iccl.analysis.probe_config import configure_runtime, resolved_config, validate_probe_config
from iccl.analysis.probe_evaluation import evaluate_probe
from iccl.analysis.probe_storage import CapturedSplit, read_manifest, write_json
from iccl.analysis.probe_training import selected_features, train_probe
from iccl.analysis.probes import fit_ridge


def benchmark_probe(cfg: DictConfig, out_dir: Path | str) -> dict[str, Any]:
    validate_probe_config(cfg, "benchmark")
    if int(os.environ.get("SLURM_NTASKS", 1)) != 1:
        raise ValueError("the throughput pilot is bounded to one GPU worker")
    d, b = cfg.probe.dataset, cfg.probe.benchmark
    if d.num_worlds > b.max_worlds or max(d.trajectories.values()) > b.max_trials_per_split:
        raise ValueError(
            "pilot budget exceeds benchmark bounds; use probe=pilot or a smaller configuration"
        )
    if Path(d.path, "manifest.json").exists():
        raise ValueError("timing requires a fresh pilot root; existing shards would measure resume")
    device = configure_runtime(cfg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    times = {}
    for stage, action in (
        ("capture", capture_dataset),
        ("train", train_probe),
        ("eval", evaluate_probe),
    ):
        start = time.perf_counter()
        action(cfg, root / stage)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times[stage] = time.perf_counter() - start
    manifest = read_manifest(d.path)
    split = CapturedSplit(d.path, 0, d.target_positions[-1], "train")
    options = resolved_config(cfg)["probe"]["training"]["solver"]
    timings = []
    try:
        for feature in selected_features(cfg, manifest):
            if feature.name != "all" or feature.family not in {"state", "mlp_hidden"}:
                continue
            matrix = split.matrix(f"trained__final_answer__{feature.family}", feature)
            for method in ("direct", "cg"):
                measurements = []
                diagnostic: dict[str, Any] = {}
                for repeat in range(b.warmup_repeats + b.measured_repeats):
                    start = time.perf_counter()
                    _, diagnostic = fit_ridge(
                        matrix,
                        split.targets,
                        cfg.probe.training.l2_grid[-1],
                        device=device,
                        options=options | {"method": method},
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    if not diagnostic["converged"]:
                        raise RuntimeError(
                            "pilot ridge solve failed its numerical convergence check"
                        )
                    if repeat >= b.warmup_repeats:
                        measurements.append(elapsed)
                timings.append(
                    {
                        "family": feature.family,
                        "features": matrix.shape[1],
                        "trials": matrix.shape[0],
                        "method": method,
                        "median_seconds": float(np.median(measurements)),
                        "measured_seconds": measurements,
                        "diagnostics": diagnostic,
                    }
                )
    finally:
        split.close()
    usage = resource.getrusage(resource.RUSAGE_SELF)
    report = {
        "stage_seconds": times,
        "warmed_solves": timings,
        "host_peak_bytes": int(usage.ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "feature_bytes": manifest["estimated_feature_bytes"],
        "device": str(device),
        "config": resolved_config(cfg),
        "interpretation": "Pipeline timings include cold kernel compilation, I/O and reporting. "
        "Repeated solves exclude capture and report rendering. "
        "Small-n timings do not establish convergence costs at larger training budgets.",
    }
    write_json(root / "benchmark.json", report)
    return report
