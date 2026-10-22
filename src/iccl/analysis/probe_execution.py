"""World-sharded stage execution with one writer and explicit completion barriers."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from iccl.analysis.probe_config import resolved_config
from iccl.analysis.probe_storage import digest, lock, read_json, write_json
from iccl.analysis.probe_tracking import StudyReporter, WorkerEvents


def run_stage(
    cfg: DictConfig,
    stage: str,
    worker: Callable[[int, WorkerEvents], Any],
    finalize: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Slurm ranks operate on disjoint worlds; rank zero alone owns the study/W&B lock."""
    rank, size = int(os.environ.get("SLURM_PROCID", 0)), int(os.environ.get("SLURM_NTASKS", 1))
    if (
        not 1 <= size <= min(cfg.execution.max_gpus, cfg.probe.dataset.num_worlds)
        or not 0 <= rank < size
    ):
        raise ValueError("worker allocation exceeds the declared GPU cap or has invalid ranks")
    root = Path(cfg.experiment.root) / cfg.experiment.phase
    root.mkdir(parents=True, exist_ok=True)
    attempt = os.environ.get("SLURM_JOB_ID", uuid.uuid4().hex)
    attempt += "-" + os.environ.get("SLURM_RESTART_COUNT", "0")
    folder = root / "stages" / f"{stage}-{attempt}"
    folder.mkdir(parents=True, exist_ok=True)
    contract = digest(resolved_config(cfg))
    coordinator = lock(Path(cfg.experiment.root) / ".study-writer.lock", blocking=False)
    reporter = None
    success = False
    if rank == 0:
        coordinator.__enter__()
    events = WorkerEvents(root, stage, attempt, rank)
    started = time.monotonic()
    try:
        if rank == 0:
            reporter = StudyReporter(cfg, root)
            reporter.start()
            write_json(folder / "ready.json", {"contract": contract, "workers": size})
        while not (folder / "ready.json").exists():
            if time.monotonic() - started > cfg.execution.stage_timeout_seconds:
                raise TimeoutError("coordinator did not initialize the stage")
            time.sleep(1)
        if read_json(folder / "ready.json") != {"contract": contract, "workers": size}:
            raise ValueError("worker configuration differs from the coordinator")
        for world in range(rank, cfg.probe.dataset.num_worlds, size):
            events.emit("world_start", phase=cfg.experiment.phase, world=world)
            worker(world, events)
            events.emit("world_complete", phase=cfg.experiment.phase, world=world)
        write_json(folder / f"rank-{rank}.done.json", {"contract": contract})
        result = {"stage": stage, "rank": rank, "workers": size}
        if rank == 0:
            waiting = time.monotonic()
            while len(list(folder.glob("rank-*.done.json"))) != size:
                if list(folder.glob("rank-*.failed.json")):
                    raise RuntimeError("a stage worker failed; see its event log")
                if time.monotonic() - waiting > cfg.execution.stage_timeout_seconds:
                    raise TimeoutError("stage completion barrier exceeded its bound")
                time.sleep(1)
            result.update(finalize())
            events.emit("stage_complete", phase=cfg.experiment.phase, **result)
            if stage == "eval" and reporter is not None:
                # Stop the background writer before publishing final tables and figures.
                reporter.stop.set()
                if reporter.thread is not None:
                    reporter.thread.join()
                reporter.flush()
                try:
                    reporter.publish_results(Path(cfg.probe.evaluation.output))
                except Exception as error:
                    print(
                        f"W&B publication deferred; saved results can be replayed: {error}",
                        flush=True,
                    )
            write_json(folder / "complete.json", result)
        success = True
        return result
    except BaseException as error:
        write_json(folder / f"rank-{rank}.failed.json", {"error": str(error)})
        events.emit("failed", phase=cfg.experiment.phase, error=str(error))
        raise
    finally:
        events.close()
        if reporter is not None:
            reporter.finish(success=success)
        if rank == 0:
            coordinator.__exit__(None, None, None)
