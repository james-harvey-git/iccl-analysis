"""Saved-result plots for module reconstruction and independently fitted controls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iccl.analysis.probe_results import probe_summary_rows, read_results, require_comparable
from iccl.analysis.probe_targets import PROTOCOL

_PREDICTION_NAMES = {
    "none": "Full decoder",
    "constant": "Constant control",
    "shuffled_targets": "Shuffled targets",
}
_COLORS = {"decoder": "#2563eb", "zero": "#64748b"}


def _estimate_trace(
    figure: go.Figure,
    x: list[Any],
    rows: list[dict[str, Any]],
    *,
    prediction: str,
    name: str,
    col: int = 1,
    lines: bool = True,
    coordinate_labels: list[str] | None = None,
) -> None:
    """Plot saved means and exact interval endpoints, including absent or asymmetric CIs."""
    color = _COLORS[prediction]
    selected = [
        (coordinate, row)
        for coordinate, row in zip(x, rows, strict=True)
        if row["ci_low"] is not None and row["ci_high"] is not None
    ]
    if selected:
        # Centre the interval independently of the mean: a small bootstrap sample
        # need not produce an interval that contains the observed mean.
        figure.add_trace(
            go.Scatter(
                x=[coordinate for coordinate, _ in selected],
                y=[(row["ci_low"] + row["ci_high"]) / 2 for _, row in selected],
                error_y={
                    "type": "data",
                    "array": [(row["ci_high"] - row["ci_low"]) / 2 for _, row in selected],
                    "color": color,
                    "thickness": 1.5,
                    "width": 4,
                },
                mode="markers",
                marker={"size": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=prediction,
            ),
            row=1,
            col=col,
        )
    coordinate_labels = coordinate_labels or [str(coordinate) for coordinate in x]
    custom = [
        [
            row["n_episodes"],
            "unavailable"
            if row["ci_low"] is None
            else f"[{row['ci_low']:.5g}, {row['ci_high']:.5g}]",
            label,
            row.get("n_modules"),
        ]
        for row, label in zip(rows, coordinate_labels, strict=True)
    ]
    figure.add_trace(
        go.Scatter(
            x=x,
            y=[row["mean"] for row in rows],
            name=name,
            legendgroup=prediction,
            showlegend=col == 1,
            mode="lines+markers" if lines else "markers",
            line={"color": color, "dash": "dash" if prediction == "zero" else "solid"},
            marker={
                "color": color,
                "size": 8,
                "symbol": "diamond" if prediction == "zero" else "circle",
            },
            customdata=custom,
            hovertemplate=(
                "%{customdata[2]}<br>Mean: %{y:.5g}<br>Episodes: %{customdata[0]}"
                "<br>CI: %{customdata[1]}"
                + (
                    "<br>Modules: %{customdata[3]}"
                    if any(row.get("n_modules") is not None for row in rows)
                    else ""
                )
                + "<extra>%{fullData.name}</extra>"
            ),
        ),
        row=1,
        col=col,
    )


def probe_evaluation_figures(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray], summary: dict[str, Any]
) -> dict[str, go.Figure]:
    """Interactive dashboard panels derived solely from the saved evaluation measurements."""
    rows = probe_summary_rows(summary)
    labels = {"decoder": _PREDICTION_NAMES[metadata["control"]], "zero": "Zero output"}
    figures: dict[str, go.Figure] = {}
    count = summary["decoder"]["all"]["n_episodes"]
    confidence = metadata["confidence"] * 100
    ci_note = (
        f"Intervals: {confidence:g}% whole-episode bootstrap. Unavailable intervals are omitted."
    )
    floor_count = summary["decoder"]["all"]["variance_floored_tasks"]
    functional_note = (
        "Oracle task coefficients; predicted readout. "
        f"{floor_count:,}/{metadata['task_count'] * count:,} task variances "
        f"floored at {metadata['variance_floor']:g}."
    )

    def panel(
        key: str, title: str, subtitles: tuple[str, ...] = (), note: str = ci_note
    ) -> go.Figure:
        figure = make_subplots(
            rows=1, cols=len(subtitles) or 1, subplot_titles=subtitles, horizontal_spacing=0.14
        )
        figure.update_layout(
            title={
                "text": (
                    f"{title}<br><sup>{metadata['split'].title()} · {labels['decoder']} · "
                    f"step {metadata['step']:,} · {count:,} episodes</sup>"
                ),
                "font": {"size": 19},
                "x": 0.04,
            },
            template="plotly_white",
            height=510,
            margin={"l": 75, "r": 25, "t": 120, "b": 125},
            legend={"orientation": "h", "y": -0.26, "x": 0},
            hovermode="closest",
        )
        figure.add_annotation(
            text=note,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.47,
            xanchor="left",
            align="left",
            showarrow=False,
            font={"size": 11, "color": "#475569"},
        )
        figure.update_yaxes(rangemode="tozero")
        figure.update_xaxes(automargin=True)
        figures[f"probe/{metadata['split']}/figures/{key}"] = figure
        return figure

    module = panel(
        "module_by_exposure",
        "Module reconstruction by exposure count",
        note=f"Each distinct module contributes once within its exposure group.<br>{ci_note}",
    )
    for prediction, name in labels.items():
        selected = sorted(
            (
                row
                for row in rows
                if row["prediction"] == prediction
                and row["population"] == "all"
                and row["exposure_count"] is not None
            ),
            key=lambda row: row["exposure_count"],
        )
        _estimate_trace(
            module,
            [row["exposure_count"] for row in selected],
            selected,
            prediction=prediction,
            name=name,
        )
    module.update_xaxes(
        title_text="Tasks containing the module", tickmode="linear", tick0=1, dtick=1
    )
    module.update_yaxes(title_text="Aligned module weight and bias MSE")
    functional = panel(
        "functional_by_task",
        "Oracle-coefficient functional reconstruction",
        ("Raw error", "Variance-normalized error"),
        note=f"{functional_note}<br>{ci_note}",
    )
    for figure, metric, y_title, col in (
        (functional, "functional_mse_by_task", "Output MSE", 1),
        (functional, "functional_nmse_by_task", "Output nMSE", 2),
    ):
        for prediction, name in labels.items():
            selected = [
                row
                for row in rows
                if row["prediction"] == prediction
                and row["population"] == "all"
                and row["metric"] == metric
            ]
            _estimate_trace(
                figure,
                [row["task_position"] for row in selected],
                selected,
                prediction=prediction,
                name=name,
                col=col,
            )
        figure.update_xaxes(
            title_text="Task position", tickmode="linear", tick0=1, dtick=1, row=1, col=col
        )
        figure.update_yaxes(title_text=y_title, row=1, col=col)

    components = panel(
        "parameter_components",
        "Parameter reconstruction by component",
        note=f"Each component is averaged over its own scalars.<br>{ci_note}",
    )
    component_keys = ("weight_mse", "bias_mse", "readout_mse")
    component_labels = ["Module weights", "Module biases", "Shared readout"]
    for prediction, name in labels.items():
        selected = [
            next(
                row
                for row in rows
                if row["prediction"] == prediction
                and row["population"] == "all"
                and row["metric"] == metric
            )
            for metric in component_keys
        ]
        offset = -0.08 if prediction == "decoder" else 0.08
        _estimate_trace(
            components,
            [i + offset for i in range(3)],
            selected,
            prediction=prediction,
            name=name,
            lines=False,
            coordinate_labels=component_labels,
        )
    components.update_xaxes(
        tickvals=[0, 1, 2],
        ticktext=component_labels,
        range=[-0.5, 2.5],
    )
    components.update_yaxes(title_text="Aligned MSE per scalar")

    distributions = panel(
        "episode_error_distribution",
        "Distribution of episode errors",
        ("Joint parameter error", "Functional error"),
        note=(
            "Each point is one whole episode; functional errors are averaged "
            f"over its {metadata['task_count']} tasks."
        ),
    )
    for prediction, name in labels.items():
        for col, values, x_title in (
            (1, arrays[f"{prediction}_joint_mse"], "Joint aligned MSE"),
            (
                2,
                arrays[f"{prediction}_functional_nmse_by_task"].mean(axis=1),
                "Oracle-coefficient nMSE",
            ),
        ):
            order = np.argsort(values, kind="stable")
            distributions.add_trace(
                go.Scatter(
                    x=values[order].tolist(),
                    y=(np.arange(1, count + 1) / count).tolist(),
                    mode="lines+markers",
                    marker={"size": 4},
                    name=name,
                    legendgroup=prediction,
                    showlegend=col == 1,
                    line={
                        "color": _COLORS[prediction],
                        "shape": "hv",
                        "dash": "dash" if prediction == "zero" else "solid",
                    },
                    customdata=np.stack(
                        (arrays["episode_index"][order], arrays["latent_rank"][order]), axis=1
                    ).tolist(),
                    hovertemplate=(
                        "Error: %{x:.5g}<br>Cumulative fraction: %{y:.1%}"
                        "<br>Episode: %{customdata[0]}<br>Latent rank: %{customdata[1]}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                ),
                row=1,
                col=col,
            )
            distributions.update_xaxes(title_text=x_title, row=1, col=col)
    distributions.update_yaxes(
        title_text="Fraction of episodes ≤ error", range=[0, 1.02], tickformat=".0%"
    )

    ranks = panel(
        "latent_rank",
        "Reconstruction by observed latent rank",
        ("Joint parameter error", "Functional error"),
    )
    rank_values = sorted(
        int(group.removeprefix("rank_")) for group in summary["decoder"] if group != "all"
    )
    for prediction, name in labels.items():
        for col, metric, y_title in (
            (1, "joint_mse", "Joint aligned MSE"),
            (2, "functional_nmse", "Oracle-coefficient nMSE"),
        ):
            selected = [
                next(
                    row
                    for row in rows
                    if row["prediction"] == prediction
                    and row["population"] == f"rank_{rank}"
                    and row["metric"] == metric
                )
                for rank in rank_values
            ]
            _estimate_trace(ranks, rank_values, selected, prediction=prediction, name=name, col=col)
            ranks.update_yaxes(title_text=y_title, row=1, col=col)
    ranks.update_xaxes(
        title_text="Observed task-latent rank",
        tickvals=rank_values,
        ticktext=[
            f"{rank}<br>n={summary['decoder'][f'rank_{rank}']['n_episodes']:,}"
            for rank in rank_values
        ],
        range=[min(rank_values) - 0.5, max(rank_values) + 0.5],
    )

    return figures


def plot_probe_results(
    results: list[Path | str], out_dir: Path | str, *, histories: list[Path | str] | None = None
) -> list[Path]:
    """Draw exposure, functional and control comparisons from checked numerical artifacts."""
    if not results:
        raise ValueError("provide at least one probe results directory")
    reports = [read_results(path) for path in results]
    require_comparable([report[0] for report in reports])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{_PREDICTION_NAMES[meta['control']]} · step {meta['step']} · {Path(path).name}"
        for (meta, _, _), path in zip(reports, results, strict=True)
    ]
    saved = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    populations = [
        (label, report[2]["decoder"]["all"]) for label, report in zip(labels, reports, strict=True)
    ]
    populations.append(("Zero output", reports[0][2]["zero"]["all"]))
    for label, report in populations:
        exposures = sorted(map(int, report["exposure"]))
        module = [report["exposure"][str(n)] for n in exposures]
        function = report["metrics"]["functional_nmse_by_task"]
        positions = np.arange(1, len(function["mean"]) + 1)
        style = {"linestyle": "--", "color": "0.4"} if label == "Zero output" else {}
        (line,) = axes[0].plot(
            exposures, [value["mean"] for value in module], marker="o", label=label, **style
        )
        for exposure, value in zip(exposures, module, strict=True):
            if value["ci_low"] is not None:
                axes[0].vlines(exposure, value["ci_low"], value["ci_high"], color=line.get_color())
        axes[1].plot(
            positions,
            function["mean"],
            marker="o",
            label=label,
            color=line.get_color(),
            linestyle=style.get("linestyle", "-"),
        )
        if function["ci_low"] is not None:
            axes[1].fill_between(
                positions,
                function["ci_low"],
                function["ci_high"],
                alpha=0.15,
                color=line.get_color(),
            )
    axes[0].set(
        xlabel="Tasks containing the module",
        ylabel="Module parameters (MSE)",
        xticks=np.unique(reports[0][1]["occurrence_count"]),
    )
    axes[1].set(
        xlabel="Task position",
        ylabel="Oracle-coefficient function (nMSE)",
        xticks=np.arange(1, reports[0][0]["task_count"] + 1),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    path = out_dir / "reconstruction-by-exposure.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    estimates = [report[2]["decoder"]["all"]["metrics"]["joint_mse"] for report in reports]
    estimates.append(reports[0][2]["zero"]["all"]["metrics"]["joint_mse"])
    for i, value in enumerate(estimates):
        ax.scatter(i, value["mean"])
        if value["ci_low"] is not None:
            ax.vlines(i, value["ci_low"], value["ci_high"])
    ax.set(
        xticks=np.arange(len(estimates)),
        xticklabels=[*labels, "zero output"],
        ylabel="Joint aligned parameter MSE",
    )
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    path = out_dir / "control-comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)
    for index, (metadata, arrays, summary) in enumerate(reports):
        directory = out_dir if len(reports) == 1 else out_dir / f"report-{index + 1}"
        directory.mkdir(exist_ok=True)
        for key, figure in probe_evaluation_figures(metadata, arrays, summary).items():
            path = directory / f"{key.rsplit('/', 1)[-1]}.html"
            figure.write_html(path, include_plotlyjs="directory")
            saved.append(path)
    if histories:
        saved.append(plot_probe_history(histories, out_dir))
    return saved


def plot_probe_history(histories: list[Path | str], out_dir: Path | str) -> Path:
    loaded = [json.loads(Path(source).read_text()) for source in histories]
    if not loaded or any(history.get("protocol") != PROTOCOL for history in loaded):
        raise ValueError("learning curves require module-set-decoder histories; recapture/retrain")
    keys = ("dataset_id", "split_signatures", "target_layout", "readout_weight")
    if any(any(history[key] != loaded[0][key] for key in keys) for history in loaded[1:]):
        raise ValueError("learning curves require identical captured populations and objectives")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for source, history in zip(histories, loaded, strict=True):
        source = Path(source)
        training, validation = history["training"], history["validation"]
        (line,) = ax.plot(
            [r["step"] for r in training],
            [r["loss"] for r in training],
            alpha=0.5,
            label=f"{source.parent.name}: training",
        )
        ax.plot(
            [r["step"] for r in validation],
            [r["joint_mse"] for r in validation],
            marker="o",
            color=line.get_color(),
            label=f"{source.parent.name}: validation",
        )
    ax.set(xlabel="Optimizer update", ylabel="Joint aligned parameter MSE")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    path = out_dir / "learning-curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
