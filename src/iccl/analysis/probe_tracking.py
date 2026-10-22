"""Durable worker events and one resumable W&B writer for the complete study."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from omegaconf import DictConfig
from plotly.subplots import make_subplots

from iccl.analysis.probe_config import resolved_config
from iccl.analysis.probe_storage import read_json, write_json


class WorkerEvents:
    """One append-only stream per worker attempt; incomplete trailing records are ignored."""

    def __init__(self, root: Path, stage: str, attempt: str, rank: int) -> None:
        self.stage, self.rank, self.counter = stage, rank, 0
        self.phase = root.name
        self.attempt = attempt
        folder = root / "events"
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{stage}-{attempt}-rank{rank}.jsonl"
        self.stream = self.path.open("a", buffering=1)

    def emit(self, kind: str, **values: Any) -> None:
        event = {
            "event_id": f"{self.stage}/{self.attempt}/{self.rank}/{self.counter}",
            "time": time.time(),
            "stage": self.stage,
            "rank": self.rank,
            "kind": kind,
            "phase": self.phase,
            **values,
        }
        self.stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.counter += 1
        if kind in {"world_complete", "failed", "stage_complete"}:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            print(json.dumps(event), flush=True)

    def close(self) -> None:
        self.stream.close()


class StudyReporter:
    """A coordinator consumes committed lines while workers fit independent probes."""

    def __init__(self, cfg: DictConfig, phase_root: Path) -> None:
        self.cfg, self.phase_root = cfg, phase_root
        self.study_root = Path(cfg.experiment.root)
        self.run = None
        self.module: Any = None
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.state_path = self.study_root / "reporting-state.json"
        self.state = (
            read_json(self.state_path) if self.state_path.exists() else {"offsets": {}, "step": 0}
        )
        self.active: dict[str, list[dict[str, Any]]] = {}
        self.finished = self.state.get("finished", 0)
        self.failed = self.state.get("failed", 0)
        self.last_plot = 0.0
        identity_path = Path(cfg.wandb.run_id_file)
        if identity_path.exists():
            self.identity = read_json(identity_path)
            if any(self.identity[key] != cfg.wandb[key] for key in ("entity", "project")):
                raise ValueError("study W&B identity differs from the configured entity/project")
        else:
            self.identity = {
                "id": uuid.uuid4().hex[:8],
                "entity": cfg.wandb.entity,
                "project": cfg.wandb.project,
                "online_created": False,
            }
            write_json(identity_path, self.identity)
        if cfg.wandb.mode == "online":
            import wandb

            self.module = wandb
            try:
                self.run = wandb.init(
                    entity=cfg.wandb.entity,
                    project=cfg.wandb.project,
                    id=self.identity["id"],
                    name=cfg.wandb.name,
                    resume="allow" if self.identity["online_created"] else "never",
                    config={
                        "study": str(self.study_root.resolve()),
                        "protocol": "fixed-world-task-vector-v1",
                    },
                    dir=str(phase_root),
                    job_type="task-vector-study",
                )
                self.identity["online_created"] = True
                write_json(identity_path, self.identity)
                self.state["step"] = max(self.state["step"], int(self.run.step))
                self.run.config.update(
                    {f"configuration/{cfg.experiment.phase}": resolved_config(cfg)},
                    allow_val_change=True,
                )
            except Exception as error:
                print(f"W&B unavailable; preserving local events for replay: {error}", flush=True)
                self.run = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._loop, name="study-reporter", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.wait(2):
            try:
                self.flush()
            except Exception as error:
                print(f"reporter retained events after logging failure: {error}", flush=True)

    def flush(self) -> None:
        if self.run is None:
            return
        for file in sorted(self.study_root.glob("*/events/*.jsonl")):
            key = str(file.relative_to(self.study_root))
            offset = self.state["offsets"].get(key, 0)
            with file.open() as stream:
                stream.seek(offset)
                while True:
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        break
                    event = json.loads(line)
                    self._publish(event)
                    self.state["offsets"][key] = stream.tell()
                    write_json(self.state_path, self.state)

    def _publish(self, event: dict[str, Any]) -> None:
        if self.run is None:
            return
        payload: dict[str, Any] = {
            "phase": event.get("phase", self.phase_root.name),
            "stage": event["stage"],
            "event_id": event["event_id"],
            "worker/rank": event["rank"],
            "event/kind": event["kind"],
        }
        if event["kind"] == "fit_trace":
            worker = f"{payload['phase']}/{event['rank']}"
            trace = self.active.setdefault(worker, [])
            if trace and trace[-1]["fit_id"] != event["fit_id"]:
                trace.clear()
            if not trace or trace[-1]["event_id"] != event["event_id"]:
                trace.append(event)
            payload.update({"fit/id": event["fit_id"], "fit/iteration": event["iteration"]})
            # Per-fit curves use their own iteration axes in the explicit figure.
            if time.monotonic() - self.last_plot >= 15:
                self._trace_figure(payload)
                self.last_plot = time.monotonic()
        finished = self.finished + int(event["kind"] == "fit_complete")
        failed = self.failed + int(event["kind"] == "failed")
        payload["progress/completed_fits"] = finished
        payload["progress/failed_units"] = failed
        for key in (
            "world",
            "seconds",
            "gpu_memory_bytes",
            "host_peak_bytes",
            "trials",
            "features",
        ):
            if key in event:
                payload[f"worker/{key}"] = event[key]
        self.run.log(payload, step=self.state["step"])
        self.finished, self.failed = finished, failed
        self.state["finished"] = self.finished
        self.state["failed"] = self.failed
        self.state["step"] += 1

    def _trace_figure(self, payload: dict[str, Any]) -> None:
        if self.module is None or not self.active:
            return
        traces = list(self.active.values())
        figure = make_subplots(
            rows=len(traces),
            cols=2,
            subplot_titles=[
                title for trace in traces for title in (trace[-1]["fit_id"] + " · MSE", "R²")
            ],
        )
        for row, trace in enumerate(traces, 1):
            for metric, column in (
                ("train_mse", 1),
                ("validation_mse", 1),
                ("train_r2", 2),
                ("validation_r2", 2),
            ):
                figure.add_trace(
                    go.Scatter(
                        x=[x["iteration"] for x in trace],
                        y=[x.get(metric) for x in trace],
                        name=metric,
                        legendgroup=metric,
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=column,
                )
        figure.update_xaxes(title_text="Solver iteration")
        figure.update_layout(
            height=max(340, 260 * len(traces)), title="Current / most recent fit per worker"
        )
        payload[f"{self.phase_root.name}/training/live_fits"] = self.module.Plotly(figure)

    def publish_results(self, results: Path) -> None:
        if self.run is None or self.module is None:
            return
        from iccl.analysis.plotting import probe_evaluation_figures

        report = read_json(results / "summary.json")
        payload: dict[str, Any] = {
            f"{report['phase']}/evaluation/{name}": self.module.Plotly(figure)
            for name, figure in probe_evaluation_figures(report).items()
        }
        columns: list[Any] = [
            "phase",
            "population",
            "position",
            "capture",
            "family",
            "selection",
            "control",
            "metric",
            "mean",
            "lower",
            "upper",
            "world_count",
        ]
        payload[f"{report['phase']}/evaluation/summary"] = self.module.Table(
            columns=list(columns),
            data=[[row.get(key) for key in columns] for row in report["aggregate"]],
        )
        self.run.log(payload, step=self.state["step"])
        self.state["step"] += 1
        write_json(self.state_path, self.state)
        if self.cfg.wandb.upload_results:
            artifact = self.module.Artifact(
                f"task-vector-results-{self.identity['id']}", type="evaluation"
            )
            for name in ("summary.json", "report.md", "manifest.json"):
                path = results / name
                if path.exists():
                    artifact.add_file(str(path))
            self.run.log_artifact(artifact)

    def finish(self, *, success: bool) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        try:
            self.flush()
            if self.run is not None:
                payload: dict[str, Any] = {}
                self._trace_figure(payload)
                if payload:
                    self.run.log(payload, step=self.state["step"])
                    self.state["step"] += 1
                write_json(self.state_path, self.state)
                self.run.finish(exit_code=0 if success else 1)
        except Exception as error:
            print(f"W&B publishing incomplete; local events remain available: {error}", flush=True)
