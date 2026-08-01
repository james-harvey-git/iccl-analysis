"""The W&B run for one training or evaluation job, mirrored to stdout.

Two entry points report the same evaluation: the trainer periodically during a
run, and ``scripts/eval.py`` once against a saved checkpoint. Metric namespacing
(``eval-scalars/``, ``eval-curves/``), the accumulating Plotly curve panels, and
the run's setup live here so the two cannot drift apart.

A checkpoint records the identity of the run that wrote it, so an evaluation of
that checkpoint can name its source run and link back to it.
"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import plotly.graph_objects as go


@dataclass(frozen=True)
class SourceRun:
    """The W&B run that produced an evaluated checkpoint."""

    entity: str | None
    project: str | None
    run_id: str
    name: str | None
    step: int

    @property
    def label(self) -> str:
        """Display name, falling back to the id: an offline run is not named
        until it syncs, and cluster jobs train offline."""
        return self.name or self.run_id

    @property
    def url(self) -> str | None:
        """Canonical run URL. Built from the ids rather than taken from the live
        run, whose own ``url`` is unset offline; the id survives ``wandb sync``,
        so the link resolves once the run is uploaded. Assumes wandb.ai."""
        if self.entity is None or self.project is None:
            return None
        return f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.run_id}"


def source_from_checkpoint(checkpoint: dict[str, Any]) -> SourceRun | None:
    """The run a checkpoint came from, or None when it carries no reference —
    it was trained with W&B disabled, or written before the reference existed.
    An evaluation of such a checkpoint simply stands alone."""
    reference = checkpoint.get("wandb_run")
    if reference is None:
        return None
    return SourceRun(step=int(checkpoint["step"]), **reference)


def _nan_to_none(curve: np.ndarray) -> list[float | None]:
    """NaN-padded curve tail to a plottable list (NaN renders as a gap)."""
    return [None if math.isnan(v) else float(v) for v in curve]


def _curve_figure(history: list[tuple[int, np.ndarray]], title: str, xname: str) -> "go.Figure":
    """A line-per-eval-step figure, each line colored by training progress so
    later steps read as the curve moving over the earlier ones."""
    import plotly.colors as pc
    import plotly.graph_objects as go

    figure = go.Figure()
    for i, (step, curve) in enumerate(history):
        fraction = i / (len(history) - 1) if len(history) > 1 else 1.0
        color = pc.sample_colorscale("Viridis", fraction)[0]
        figure.add_trace(
            go.Scatter(
                x=list(range(len(curve))),
                y=_nan_to_none(curve),
                mode="lines+markers",
                name=f"step {step}",
                line={"color": color},
            )
        )
    figure.update_layout(
        title=title, xaxis_title=xname, yaxis_title="normalized MSE", template="plotly_white"
    )
    return figure


class RunLogger:
    """Metric sink for one job: a W&B run when ``cfg.wandb.mode`` allows one,
    always stdout, and always the curve ``.npz`` files in ``out_dir/eval/``.

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
        # Per-curve list of (step, curve) across calls, so each curve is logged
        # to W&B as one Plotly panel with a line per evaluation.
        self.curve_history: dict[str, list[tuple[int, np.ndarray]]] = {}

    def start(self) -> None:
        if self.cfg.wandb.mode == "disabled":
            return
        import wandb

        config = cast(dict[str, Any], OmegaConf.to_container(self.cfg, resolve=True))
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
            notes = f"Checkpoint from {link} at step {self.source.step}."
        self.run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            mode=self.cfg.wandb.mode,
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
        if self.run is not None:
            self.run.log(metrics, step=step)
        rendered = ", ".join(f"{key}={value:.4g}" for key, value in metrics.items())
        print(f"step {step}: {rendered}")

    def log_eval(self, scalars: dict[str, float], curves: dict[str, np.ndarray], step: int) -> Path:
        """Report one evaluation, returning the path the curves were written to."""
        self.log({f"eval-scalars/{key}": value for key, value in scalars.items()}, step)
        return self._log_curves(curves, step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    def _log_curves(self, curves: dict[str, np.ndarray], step: int) -> Path:
        """Curves to the run dir as one ``.npz`` per step, and with a live run to
        an ``eval-curves/<suite>/<name>`` Plotly panel per curve, accumulating a
        line per evaluation so curves can be watched sharpening over training.
        Plotly panels carry no backing summary table, unlike
        ``wandb.plot.line_series``."""
        eval_dir = self.out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        path = eval_dir / f"step_{step:07d}.npz"
        np.savez(path, **{key.replace("/", "."): curve for key, curve in curves.items()})  # pyright: ignore[reportArgumentType]
        if self.run is None:
            return path
        import wandb

        for key, curve in curves.items():
            history = self.curve_history.setdefault(key, [])
            history.append((step, curve))
            xname = "task position" if key.endswith("task_position_curve") else "demo index"
            figure = _curve_figure(history, key, xname)
            self.run.log({f"eval-curves/{key}": wandb.Plotly(figure)}, step=step)
        return path
