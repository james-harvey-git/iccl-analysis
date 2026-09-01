"""The W&B run for one training or evaluation job, mirrored to stdout.

Training reports a small validation and canonical-capability monitor. Full
frozen-suite evaluations write portable numerical artifacts for later plotting.

A checkpoint records the identity of the run that wrote it, so an evaluation of
that checkpoint can name its source run and link back to it. Weights can also be
uploaded as W&B artifacts, which survive the run directory and let W&B record
which weights produced which numbers.
"""

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from iccl.checkpoints import SourceRun
from iccl.evaluation.metrics import EvaluationReport
from iccl.evaluation.results import SUMMARY_COLUMNS
from iccl.reporting.monitor import canonical_monitor_figures, canonical_monitor_scalars

# W&B artifact names admit only these characters.
_ARTIFACT_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def weights_artifact_name(label: str) -> str:
    """Artifact name for one run's weights. One artifact per run, so each run
    owns its own version chain and the lineage graph ties versions to the run
    that produced them."""
    return f"weights-{_ARTIFACT_UNSAFE.sub('-', label)}"


def evaluation_artifact_name(label: str) -> str:
    """Artifact name for one evaluation run's complete numerical results."""
    return f"evaluation-{_ARTIFACT_UNSAFE.sub('-', label)}"


def repo_relative(path: Path | str) -> str:
    """``path`` relative to the repo root when it sits inside it, else absolute.

    The root is the package's grandparent under the editable install ``uv sync``
    produces; the ``pyproject.toml`` check stops a non-editable install from
    reporting a path inside site-packages.
    """
    import iccl

    root = Path(iccl.__file__).resolve().parents[2]
    resolved = Path(path).resolve()
    if (root / "pyproject.toml").exists() and resolved.is_relative_to(root):
        return str(resolved.relative_to(root))
    return str(resolved)


class RunLogger:
    """Metric sink for one job: a W&B run when ``cfg.wandb.mode`` allows one,
    always stdout, and canonical curve ``.npz`` files in ``out_dir/monitor/``.

    ``start`` opens the W&B run. It is separate from construction so a caller
    can build its model and fail before a run is created. ``source`` names the
    training run an evaluated checkpoint came from: it goes into this run's
    config for filtering and into its notes as a link.
    """

    def __init__(
        self,
        cfg: DictConfig,
        out_dir: Path | str,
        *,
        job_type: str,
        source: SourceRun | None = None,
    ) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.job_type = job_type
        self.source = source
        self.run = None
        self._pending_step: int | None = None
        self._pending_payload: dict[str, Any] = {}

    def start(self) -> None:
        if self.cfg.wandb.mode == "disabled":
            return
        import wandb

        config = cast(dict[str, Any], OmegaConf.to_container(self.cfg, resolve=True))
        # Where this run wrote its weights, so the overview page leads back to
        # them. Repo-relative, since `outputs/` is gitignored and lives only on
        # the machine that trained the model.
        config["paths"] = {
            "run_dir": repo_relative(self.out_dir),
            "checkpoints": repo_relative(self.out_dir / "checkpoints"),
            "snapshots": repo_relative(self.out_dir / "snapshots"),
        }
        tags = [self.job_type]
        notes = None
        if self.source is not None:
            config["source_run"] = asdict(self.source) | {"url": self.source.url}
            tags.append(f"source:{self.source.label}")
            link = (
                f"[{self.source.label}]({self.source.url})"
                if self.source.url
                else f"`{self.source.label}`"
            )
            notes = (
                f"Checkpoint trajectory from {link}."
                if self.job_type == "eval-trajectory"
                else f"Checkpoint from {link} at step {self.source.step}."
            )
        self.run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            mode=self.cfg.wandb.mode,
            # None is W&B's "generate one", so an unset name needs no branch.
            name=self.cfg.wandb.get("name"),
            job_type=self.job_type,
            tags=tags,
            notes=notes,
            config=config,
            dir=str(self.out_dir),
        )
        if self.source is not None and self.source.url:
            print(f"evaluating a checkpoint from {self.source.url}")

    def run_reference(self) -> dict[str, Any] | None:
        """This run's identity, for a checkpoint to record so a later evaluation
        can link back to it."""
        if self.run is None:
            return None
        return {
            "entity": self.run.entity,
            "project": self.run.project,
            "run_id": self.run.id,
            "name": self.run.name,
        }

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._queue(metrics, step)
        self._print(metrics, step)

    def _queue(self, payload: dict[str, Any], step: int) -> None:
        """Merge all values produced at one optimizer step into one W&B record."""
        if self.run is None:
            return
        if self._pending_step is not None and self._pending_step != step:
            self.flush()
        self._pending_step = step
        self._pending_payload.update(payload)

    def flush(self) -> None:
        """Commit the pending W&B record, if any."""
        if self.run is not None and self._pending_step is not None:
            self.run.log(self._pending_payload, step=self._pending_step)
        self._pending_step = None
        self._pending_payload = {}

    @staticmethod
    def _print(metrics: dict[str, float], step: int) -> None:
        rendered = ", ".join(f"{key}={value:.4g}" for key, value in metrics.items())
        print(f"step {step}: {rendered}")

    def log_validation(self, token_mse: float, step: int) -> None:
        """Report the training-distribution objective used for model selection."""
        self.log({"validation/token_mse": token_mse}, step)

    def log_monitor(self, report: EvaluationReport, step: int) -> Path:
        """Report one version of the compact canonical capability monitor."""
        metrics = canonical_monitor_scalars(report.summary_rows)
        path = self._log_curves(report.curves, step, directory="monitor")
        if self.run is not None:
            import wandb

            payload: dict[str, Any] = dict(metrics)
            payload.update(
                {
                    key: wandb.Plotly(figure)
                    for key, figure in canonical_monitor_figures(report.curve_rows, step).items()
                }
            )
            self._queue(payload, step)
        self._print(metrics, step)
        return path

    def log_full_evaluation(
        self,
        report: EvaluationReport,
        step: int,
        figures: dict[str, Any],
        checkpoint_reference: str,
    ) -> None:
        """Log one summary table and the compact explicit evaluation figures."""
        rows = [
            dict(row, step=step, checkpoint_reference=checkpoint_reference)
            for row in report.summary_rows
        ]
        metrics = (
            {"evaluation/validation_token_mse": report.scalars["validation/token_mse"]}
            if "validation/token_mse" in report.scalars
            else {}
        )
        if self.run is not None:
            import wandb

            payload: dict[str, Any] = dict(metrics)
            payload["evaluation/summary"] = wandb.Table(
                columns=cast(Any, SUMMARY_COLUMNS),
                data=[[row.get(column) for column in SUMMARY_COLUMNS] for row in rows],
            )
            payload.update({key: wandb.Plotly(figure) for key, figure in figures.items()})
            self._queue(payload, step)
        self._print(metrics | {"evaluation/primary_rows": float(len(rows))}, step)

    def upload_weights(self, path: Path) -> None:
        """The final snapshot as a versioned W&B artifact, so weights outlive the
        run directory and a reported number can be traced to the weights behind
        it. Gates on ``cfg.wandb.upload_weights``, which covers only this
        automatic upload — ``scripts/upload_snapshots.py`` promotes a full series
        independently, an explicit invocation being its own consent.
        """
        if self.run is None or not self.cfg.wandb.get("upload_weights"):
            return
        import wandb

        artifact = wandb.Artifact(weights_artifact_name(self.run.name or self.run.id), type="model")
        artifact.add_file(str(path))
        self.run.log_artifact(artifact, aliases=["final"])
        size_mb = path.stat().st_size / 1e6
        print(f"uploading {path.name} ({size_mb:.0f} MB) as {artifact.name}:final")

    def upload_evaluation_results(self, path: Path) -> None:
        """Upload complete evaluation rows and per-demo errors when configured."""
        enabled = bool(self.cfg.get("evaluation", {}).get("upload_results", True))
        if self.run is None or not enabled:
            return
        import wandb

        artifact = wandb.Artifact(
            evaluation_artifact_name(self.run.name or self.run.id), type="evaluation"
        )
        artifact.add_dir(str(path))
        self.run.log_artifact(artifact, aliases=["latest"])
        print(f"uploading evaluation results as {artifact.name}:latest")

    def use_artifact(self, reference: str) -> None:
        """Records this run as a consumer of an artifact, drawing the lineage
        edge from weights to the numbers computed from them."""
        if self.run is not None:
            self.run.use_artifact(reference)

    def finish(self) -> None:
        if self.run is not None:
            self.flush()
            self.run.finish()

    def _log_curves(self, curves: dict[str, np.ndarray], step: int, *, directory: str) -> Path:
        """Write one local curve archive per monitor step."""
        eval_dir = self.out_dir / directory
        eval_dir.mkdir(parents=True, exist_ok=True)
        path = eval_dir / f"step_{step:07d}.npz"
        np.savez(path, **{key.replace("/", "."): curve for key, curve in curves.items()})  # pyright: ignore[reportArgumentType]
        return path
