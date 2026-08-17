"""The W&B run for one training or evaluation job, mirrored to stdout.

Two entry points report the same evaluation: the trainer periodically during a
run, and ``scripts/eval.py`` once against a saved checkpoint. Metric namespacing
(``eval-scalars/``, ``eval-curves/``), the accumulating Plotly curve panels, and
the run's setup live here so the two cannot drift apart.

A checkpoint records the identity of the run that wrote it, so an evaluation of
that checkpoint can name its source run and link back to it. Weights can also be
uploaded as W&B artifacts, which survive the run directory and let W&B record
which weights produced which numbers.
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    import plotly.graph_objects as go

# W&B artifact names admit only these characters.
_ARTIFACT_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")

SUMMARY_COLUMNS = [
    "step",
    "capability",
    "condition",
    "slice",
    "variant",
    "suite",
    "status",
    "sampler",
    "weighting",
    "M",
    "T",
    "S",
    "B_history",
    "L_history",
    "D_target",
    "D_min",
    "D_max",
    "D_mean",
    "D_cv",
    "constituent_exposure_min",
    "constituent_exposure_max",
    "constituent_demo_exposure_min",
    "constituent_demo_exposure_max",
    "intervening_tasks",
    "prediction_token_delay",
    "serialized_token_delay",
    "metric",
    "value",
    "ci_low",
    "ci_high",
    "n_sequences",
]

CURVE_COLUMNS = [
    "step",
    "capability",
    "condition",
    "slice",
    "variant",
    "status",
    "M",
    "T",
    "S",
    "B_history",
    "L_history",
    "curve_type",
    "x_name",
    "x_value",
    "mse",
    "nmse",
    "ci_low",
    "ci_high",
    "n_sequences",
]


def weights_artifact_name(label: str) -> str:
    """Artifact name for one run's weights. One artifact per run, so each run
    owns its own version chain and the lineage graph ties versions to the run
    that produced them."""
    return f"weights-{_ARTIFACT_UNSAFE.sub('-', label)}"


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


def _grouped_row_figure(
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_field: str,
    y_field: str,
    x_title: str,
    y_title: str,
    group_fields: tuple[str, ...],
) -> "go.Figure":
    """Plot structural rows as one trace per declared control group."""
    import plotly.graph_objects as go

    figure = go.Figure()
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    for group, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        ordered = sorted(group_rows, key=lambda row: (str(row.get(x_field)), row.get(x_field)))
        label = ", ".join(
            f"{field}={value}" for field, value in zip(group_fields, group, strict=True)
        )
        error = [max(0.0, float(row["ci_high"]) - float(row[y_field])) for row in ordered]
        figure.add_trace(
            go.Scatter(
                x=[row.get(x_field) for row in ordered],
                y=[row[y_field] for row in ordered],
                mode="lines+markers",
                name=label,
                error_y={"type": "data", "array": error, "visible": True},
                customdata=[
                    [row.get("M"), row.get("T"), row.get("B_history"), row.get("L_history")]
                    for row in ordered
                ],
                hovertemplate=(
                    "M=%{customdata[0]}<br>T=%{customdata[1]}<br>"
                    "B=%{customdata[2]}<br>L=%{customdata[3]}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
    )
    return figure


def _summary_figures(summary_rows: list[dict[str, Any]]) -> dict[str, "go.Figure"]:
    """Capability summaries projected onto the configured structural axes."""
    capability_metrics = {
        "icl": ("ordinary", "nmse_aulc", "aulc"),
        "composition": ("benefit", "benefit_mean", "benefit_mean"),
        "retention": ("savings", "savings_mean", "savings_mean"),
    }
    axes = {
        "M": ("M", {"fixed_surplus", "matched_task_count", "matched_prediction_tokens"}),
        "task_count": ("T", {"task_count", "matched_serialized_prefix"}),
        "history_tokens": (
            "L_history",
            {"task_count", "history_demos", "matched_prediction_tokens"},
        ),
        "history_demos": ("D_mean", {"history_demos", "demo_allocation"}),
        "demo_allocation": ("variant", {"demo_allocation"}),
    }
    specifications = []
    for capability, (condition, metric, label) in capability_metrics.items():
        for axis, (field, slices) in axes.items():
            specifications.append(
                (
                    capability,
                    condition,
                    metric,
                    field,
                    slices,
                    f"capability/{capability}/{label}_vs_{axis}",
                )
            )
    specifications.extend(
        (
            "composition",
            "benefit",
            "benefit_mean",
            field,
            None,
            key,
        )
        for field, key in {
            "constituent_exposure_min": "composition/benefit_vs_constituent_task_exposure",
            "constituent_demo_exposure_min": "composition/benefit_vs_constituent_demo_exposure",
        }.items()
    )
    specifications.extend(
        ("retention", "savings", "savings_mean", field, None, key)
        for field, key in {
            "intervening_tasks": "retention/savings_vs_intervening_tasks",
            "prediction_token_delay": "retention/savings_vs_prediction_token_delay",
            "serialized_token_delay": "retention/savings_vs_serialized_token_delay",
        }.items()
    )
    figures: dict[str, go.Figure] = {}
    for capability, condition, metric, field, slices, key in specifications:
        rows = [
            row
            for row in summary_rows
            if row["capability"] == capability
            and row["condition"] == condition
            and row["metric"] == metric
            and row.get(field) is not None
            and (slices is None or row["slice"] in slices)
        ]
        if rows:
            figures[key] = _grouped_row_figure(
                rows,
                title=key,
                x_field=field,
                y_field="value",
                x_title=field,
                y_title=metric,
                group_fields=("slice", "status", "M"),
            )
    return figures


def _capability_curve_figures(curve_rows: list[dict[str, Any]]) -> dict[str, "go.Figure"]:
    """Demo-resolved capability panels with one trace per structural cell."""
    mapping = {
        ("icl", "learning_curve"): "capability/icl/learning_curve",
        ("icl", "task_position_curve"): "icl/nmse_by_task_position",
        ("icl", "nmse_by_prediction_tokens_observed"): ("icl/nmse_by_prediction_tokens_observed"),
        ("icl", "nmse_by_unique_supports_observed"): ("icl/nmse_by_unique_supports_observed"),
        ("icl", "nmse_by_modules_covered"): "icl/nmse_by_modules_covered",
        ("composition", "constituent_curve"): "capability/composition/constituent_curve",
        ("composition", "matched_prefix_curve"): "capability/composition/matched_prefix_curve",
        ("composition", "benefit_curve"): "capability/composition/benefit_curve",
        ("composition", "no_history_curve"): "capability/composition/no_history_curve",
        ("retention", "original_curve"): "capability/retention/original_curve",
        ("retention", "relearning_curve"): "capability/retention/relearning_curve",
        ("retention", "novel_curve"): "capability/retention/novel_control_curve",
        ("retention", "shared_curve"): "capability/retention/shared_control_curve",
        ("retention", "savings_curve"): "capability/retention/savings_curve",
        ("retention", "episodic_savings_curve"): ("capability/retention/episodic_savings_curve"),
        ("retention", "module_savings_curve"): "capability/retention/module_savings_curve",
    }
    figures: dict[str, go.Figure] = {}
    for (capability, curve_type), key in mapping.items():
        rows = [
            row
            for row in curve_rows
            if row["capability"] == capability and row["curve_type"] == curve_type
        ]
        if rows:
            figures[key] = _grouped_row_figure(
                rows,
                title=key,
                x_field="x_value",
                y_field="nmse",
                x_title="demo index",
                y_title="normalized MSE",
                group_fields=("slice", "status", "M", "T", "condition"),
            )
    return figures


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
        self.summary_history: list[dict[str, Any]] = []
        self.curve_row_history: list[dict[str, Any]] = []

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
            notes = f"Checkpoint from {link} at step {self.source.step}."
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
        if self.run is not None:
            self.run.log(metrics, step=step)
        rendered = ", ".join(f"{key}={value:.4g}" for key, value in metrics.items())
        print(f"step {step}: {rendered}")

    def log_eval(
        self,
        scalars: dict[str, float],
        curves: dict[str, np.ndarray],
        step: int,
        *,
        summary_rows: list[dict[str, Any]] | None = None,
        curve_rows: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Report one evaluation, returning the path the curves were written to."""
        self.log({f"eval-scalars/{key}": value for key, value in scalars.items()}, step)
        path = self._log_curves(curves, step)
        if summary_rows or curve_rows:
            self._log_capability_tables(summary_rows or [], curve_rows or [], step)
        return path

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

    def use_artifact(self, reference: str) -> None:
        """Records this run as a consumer of an artifact, drawing the lineage
        edge from weights to the numbers computed from them."""
        if self.run is not None:
            self.run.use_artifact(reference)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()

    def _log_curves(self, curves: dict[str, np.ndarray], step: int) -> Path:
        """Write one local curve archive per evaluation step."""
        eval_dir = self.out_dir / "eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        path = eval_dir / f"step_{step:07d}.npz"
        np.savez(path, **{key.replace("/", "."): curve for key, curve in curves.items()})  # pyright: ignore[reportArgumentType]
        return path

    def _log_capability_tables(
        self,
        summary_rows: list[dict[str, Any]],
        curve_rows: list[dict[str, Any]],
        step: int,
    ) -> None:
        """Persist filterable capability rows and render their W&B panels."""
        summary = [dict(row, step=step) for row in summary_rows]
        curves = [dict(row, step=step) for row in curve_rows]
        self.summary_history.extend(summary)
        self.curve_row_history.extend(curves)

        if self.run is None:
            return
        import wandb

        payload: dict[str, Any] = {
            "eval-tables/capability_summary": wandb.Table(
                columns=cast(Any, SUMMARY_COLUMNS),
                data=[
                    [row.get(column) for column in SUMMARY_COLUMNS] for row in self.summary_history
                ],
            ),
            "eval-tables/capability_curves": wandb.Table(
                columns=cast(Any, CURVE_COLUMNS),
                data=[
                    [row.get(column) for column in CURVE_COLUMNS] for row in self.curve_row_history
                ],
            ),
        }
        figures = _summary_figures(summary_rows)
        figures.update(_capability_curve_figures(curve_rows))
        payload.update({key: wandb.Plotly(figure) for key, figure in figures.items()})
        self.run.log(payload, step=step)
