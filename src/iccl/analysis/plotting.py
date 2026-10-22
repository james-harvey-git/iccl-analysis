"""World-averaged scientific figures regenerated solely from saved numerical results."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Rectangle
from plotly.subplots import make_subplots

from iccl.analysis.probe_storage import read_json
from iccl.analysis.probe_worlds import CAPTURES

LABELS = {
    "residual_after_mixer": "Residual after mixer",
    "residual_after_mlp": "Residual after MLP",
    "mlp_hidden": "MLP hidden",
    "state": "Fast-weight state",
    "head_read": "Native head read",
    "observations": "Raw observations",
    "training_mean": "Training mean",
    "shuffled_targets": "Shuffled targets",
    "untrained": "Untrained GDN",
    "final_input": "Final input",
    "final_answer": "Final answer",
    "branch_input": "Discarded query",
    "test_familiar": "Fresh streams · familiar pairs",
    "test_unseen": "Fresh streams · unseen pairs",
    "r2_macro": "Macro constituent R²",
    "active_pair_accuracy": "Exact-pair accuracy",
    "active_coefficient_rmse": "Active-coefficient RMSE",
    "fixed_scale_vector_mse": "Fixed-scale vector MSE",
}
COLORS = ["#2166ac", "#ef8a62", "#1b9e77", "#7b3294", "#c99b19", "#737373", "#222222"]


def _curve(
    figure: go.Figure,
    rows: list[dict[str, Any]],
    name: str,
    color: str,
    *,
    column: int,
    row: int = 1,
    x: str = "position",
    legend: bool = True,
    dashed: bool = False,
) -> None:
    rows = sorted(rows, key=lambda r: r[x])
    ys = [r["mean"] for r in rows]
    plus = [None if r["upper"] is None else r["upper"] - r["mean"] for r in rows]
    minus = [None if r["lower"] is None else r["mean"] - r["lower"] for r in rows]
    figure.add_trace(
        go.Scatter(
            x=[r[x] for r in rows],
            y=ys,
            name=name,
            legendgroup=name,
            showlegend=legend,
            mode="lines+markers",
            line={"color": color, "dash": "dash" if dashed else "solid"},
            error_y={"type": "data", "array": plus, "arrayminus": minus, "symmetric": False},
            customdata=[r["world_count"] for r in rows],
            hovertemplate="%{x}: %{y:.3f}<br>worlds=%{customdata}<extra>%{fullData.name}</extra>",
        ),
        row=row,
        col=column,
    )


def exposure_grid(
    summary: dict[str, Any], population: str, family: str, metric: str
) -> dict[str, Any]:
    positions = summary["dataset"]["target_positions"]
    task_count = summary["dataset"]["tasks_per_episode"]
    bins = summary["settings"]["exposure"]["count_bins"]
    count = max(positions) if bins == "exact" else len(bins)
    labels = (
        [str(i) for i in range(count)]
        if bins == "exact"
        else [
            str(edge)
            if i + 1 < len(bins) and bins[i + 1] == edge + 1
            else f"{edge}–{bins[i + 1] - 1}"
            if i + 1 < len(bins)
            else f"{edge}+"
            for i, edge in enumerate(bins)
        ]
    )
    rows = [
        r
        for r in summary["exposure"]
        if r["population"] == population
        and r["family"] == family
        and r["selection"] == "all"
        and r["control"] == "trained"
        and r["metric"] == metric
    ]
    values = [r["mean"] for r in rows if r["mean"] is not None]
    limits = (min([0.0, *values]), max([1.0, *values]))
    panels = []
    for position in positions:
        for capture in CAPTURES:
            z, possible = np.full((count, count), np.nan), np.zeros((count, count), dtype=bool)
            hover = np.full((count, count), "Insufficient coverage", dtype=object)
            for a in range(count):
                for b in range(count):
                    a_values = (
                        [a]
                        if bins == "exact"
                        else range(bins[a], bins[a + 1] if a + 1 < count else position)
                    )
                    b_values = (
                        [b]
                        if bins == "exact"
                        else range(bins[b], bins[b + 1] if b + 1 < count else position)
                    )
                    possible[b, a] = any(
                        x <= y and x + y <= position - 1 and (position != task_count or x > 0)
                        for x in a_values
                        for y in b_values
                    )
                    if not possible[b, a]:
                        hover[b, a] = "Structurally impossible"
            for item in rows:
                if item["position"] != position or item["capture"] != capture:
                    continue
                a, b = item["e_min"], item["e_max"]
                if item["mean"] is not None:
                    z[b, a] = item["mean"]
                interval = (
                    f"[{item['lower']:.3f}, {item['upper']:.3f}]"
                    if item["lower"] is not None
                    else "unavailable"
                )
                hover[b, a] = (
                    f"{item['status']}<br>worlds={item['world_count']}<br>"
                    f"world-bootstrap CI={interval}<br>trials/world={item['trial_counts']}"
                )
            panels.append(
                {
                    "position": position,
                    "capture": capture,
                    "z": z,
                    "possible": possible,
                    "hover": hover,
                }
            )
    return {"labels": labels, "limits": limits, "panels": panels, "positions": positions}


def exposure_figure(
    summary: dict[str, Any], population: str, family: str, metric: str
) -> go.Figure:
    data = exposure_grid(summary, population, family, metric)
    labels = data["labels"]
    figure = make_subplots(
        rows=len(data["positions"]),
        cols=3,
        subplot_titles=[f"Task {p} · {LABELS[c]}" for p in data["positions"] for c in CAPTURES],
        horizontal_spacing=0.09,
        vertical_spacing=0.12,
    )
    for i, panel in enumerate(data["panels"]):
        row, col = i // 3 + 1, i % 3 + 1
        figure.add_trace(
            go.Heatmap(
                x=labels,
                y=labels,
                z=np.where(panel["possible"], 1, 0),
                colorscale=[[0, "#ffffff"], [1, "#dddddd"]],
                zmin=0,
                zmax=1,
                showscale=False,
                customdata=panel["hover"],
                hovertemplate="e_min=%{x}, e_max=%{y}<br>%{customdata}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        figure.add_trace(
            go.Heatmap(
                x=labels,
                y=labels,
                z=panel["z"],
                coloraxis="coloraxis",
                text=panel["z"],
                texttemplate="%{text:.2f}",
                customdata=panel["hover"],
                hoverongaps=False,
                hovertemplate=(
                    "e_min=%{x}, e_max=%{y}<br>score=%{z:.3f}<br>%{customdata}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
        ys, xs = np.where(~panel["possible"])
        figure.add_trace(
            go.Scatter(
                x=[labels[x] for x in xs],
                y=[labels[y] for y in ys],
                mode="text",
                text=["×"] * len(xs),
                textfont={"color": "#aaaaaa"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        figure.update_xaxes(title_text="Prior exposure · e_min", row=row, col=col)
        figure.update_yaxes(title_text="Prior exposure · e_max", row=row, col=col)
    figure.update_layout(
        title=(
            f"{LABELS[family]} · {LABELS[population]}<br>"
            "<sup>Equal-world mean; gray: insufficient coverage; ×: impossible</sup>"
        ),
        coloraxis={
            "colorscale": "RdBu" if metric == "r2_macro" else "Viridis",
            "cmin": data["limits"][0],
            "cmax": data["limits"][1],
            "colorbar": {"title": LABELS[metric]},
        },
        height=300 * len(data["positions"]) + 100,
        width=1250,
        template="plotly_white",
    )
    return figure


def probe_evaluation_figures(summary: dict[str, Any]) -> dict[str, go.Figure]:
    figures = {}
    families = list(summary["capture"]["shapes"])
    for population in summary["settings"]["splits"]:
        for metric in summary["settings"]["metrics"]:
            figure = make_subplots(rows=1, cols=3, subplot_titles=[LABELS[c] for c in CAPTURES])
            for column, capture in enumerate(CAPTURES, 1):
                for index, family in enumerate([*families, "observations", "training_mean"]):
                    baseline = family in {"observations", "training_mean"}
                    rows = [
                        r
                        for r in summary["aggregate"]
                        if r["population"] == population
                        and r["metric"] == metric
                        and r["capture"] == capture
                        and r["family"] == ("baseline" if baseline else family)
                        and r["selection"] == (family if baseline else "all")
                        and r["control"] == (family if baseline else "trained")
                    ]
                    if rows:
                        _curve(
                            figure,
                            rows,
                            LABELS[family],
                            COLORS[index % len(COLORS)],
                            column=column,
                            legend=column == 1,
                            dashed=baseline,
                        )
                figure.update_xaxes(title_text="Target task position", dtick=1, row=1, col=column)
                figure.update_yaxes(title_text=LABELS[metric], row=1, col=column)
            figure.update_layout(
                title=(f"{LABELS[population]} · {LABELS[metric]} · equal-world mean and CI"),
                template="plotly_white",
                height=440,
                width=1250,
            )
            figures[f"{population}/{metric}"] = figure
        # One matrix per family includes all three capture points.
        for family in families:
            rows = [
                r
                for r in summary["aggregate"]
                if r["population"] == population
                and r["family"] == family
                and r["control"] == "trained"
                and r["metric"] == "r2_macro"
            ]
            selections = list(dict.fromkeys(r["selection"] for r in rows))
            if len(selections) > 1:
                columns = [(p, c) for p in summary["dataset"]["target_positions"] for c in CAPTURES]
                lookup = {(r["selection"], r["position"], r["capture"]): r for r in rows}
                z = [[lookup[(s, p, c)]["mean"] for p, c in columns] for s in selections]
                figure = go.Figure(
                    go.Heatmap(
                        z=z,
                        x=[f"p{p} · {LABELS[c]}" for p, c in columns],
                        y=selections,
                        colorscale="RdBu",
                        colorbar={"title": "Mean R²"},
                    )
                )
                figure.update_layout(
                    title=f"{LABELS[family]} · {LABELS[population]} · layer/head views",
                    template="plotly_white",
                    height=max(400, 28 * len(selections)),
                    width=1100,
                )
                figures[f"{population}/detail-{family}"] = figure
            if summary["exposure"] and any(r["selection"] == "all" for r in rows):
                for metric in summary["settings"]["exposure"]["metrics"]:
                    figures[f"{population}/exposure-{family}-{metric}"] = exposure_figure(
                        summary, population, family, metric
                    )
        for family in families:
            rows = [
                r
                for r in summary["paired_controls"]
                if r["population"] == population
                and r["family"] == family
                and r["selection"] == "all"
                and r["metric"] == "r2_macro"
            ]
            if not rows:
                continue
            figure = make_subplots(rows=1, cols=3, subplot_titles=[LABELS[c] for c in CAPTURES])
            for col, capture in enumerate(CAPTURES, 1):
                for index, control in enumerate(
                    ("training_mean", "shuffled_targets", "untrained", "observations")
                ):
                    subset = [
                        r for r in rows if r["capture"] == capture and r["comparison"] == control
                    ]
                    _curve(
                        figure, subset, LABELS[control], COLORS[index], column=col, legend=col == 1
                    )
            figure.update_layout(
                title=f"{LABELS[family]} · trained minus control R² · paired worlds",
                template="plotly_white",
                height=420,
                width=1200,
            )
            figure.update_xaxes(title_text="Target task position")
            figure.update_yaxes(title_text="ΔR²")
            figures[f"{population}/control-gaps-{family}"] = figure
    if len({row["train_count"] for row in summary["validation"]}) > 1:
        figure = make_subplots(rows=1, cols=3, subplot_titles=[LABELS[c] for c in CAPTURES])
        for col, capture in enumerate(CAPTURES, 1):
            for index, family in enumerate(families):
                for position in summary["dataset"]["target_positions"]:
                    rows = [
                        r
                        for r in summary["validation"]
                        if r["capture"] == capture
                        and r["family"] == family
                        and r["control"] == "trained"
                        and r["selection"] == "all"
                        and r["metric"] == "r2_macro"
                        and r["position"] == position
                    ]
                    _curve(
                        figure,
                        rows,
                        f"{LABELS[family]} · p{position}",
                        COLORS[index % len(COLORS)],
                        column=col,
                        x="train_count",
                        legend=col == 1,
                    )
        figure.update_layout(
            title="Development data budgets · validation R²",
            template="plotly_white",
            height=500,
            width=1200,
        )
        figure.update_xaxes(title_text="Training trials per world and position", type="log")
        figure.update_yaxes(title_text="Validation R²")
        figures["development/learning-curves"] = figure
    figure = make_subplots(rows=1, cols=2, subplot_titles=["Final input", "Discarded query"])
    for col, capture in enumerate(("final_input", "branch_input"), 1):
        for index, (population, model) in enumerate(
            (p, m) for p in summary["settings"]["splits"] for m in summary["capture"]["models"]
        ):
            rows = [
                r
                for r in summary["behavior"]
                if r["capture"] == capture
                and r["population"] == population
                and r["model"] == model
                and r["metric"] == "nmse"
            ]
            _curve(
                figure, rows, f"{population} · {model}", COLORS[index], column=col, legend=col == 1
            )
    figure.update_layout(
        title="GDN query prediction error · equal-world mean and CI",
        template="plotly_white",
        height=420,
        width=1100,
    )
    figure.update_xaxes(title_text="Target task position")
    figure.update_yaxes(title_text="Query nMSE")
    figures["behavior"] = figure
    return figures


def _save_exposure_static(
    summary: dict[str, Any], population: str, family: str, metric: str, path: Path
) -> None:
    data = exposure_grid(summary, population, family, metric)
    fig, axes = plt.subplots(
        len(data["positions"]),
        3,
        figsize=(12, 3.1 * len(data["positions"])),
        squeeze=False,
        layout="constrained",
    )
    cmap = plt.get_cmap("RdBu" if metric == "r2_macro" else "viridis").with_extremes(bad="#dddddd")
    plot = None
    for axis, panel in zip(axes.flat, data["panels"], strict=True):
        plot = axis.imshow(
            panel["z"], origin="lower", cmap=cmap, vmin=data["limits"][0], vmax=data["limits"][1]
        )
        for y, x in np.ndindex(panel["z"].shape):
            if not panel["possible"][y, x]:
                axis.add_patch(
                    Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        facecolor="white",
                        hatch="///",
                        edgecolor="#bbbbbb",
                        linewidth=0.2,
                    )
                )
            elif np.isfinite(panel["z"][y, x]):
                axis.text(
                    x,
                    y,
                    f"{panel['z'][y, x]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1},
                )
        axis.set(
            xticks=range(len(data["labels"])),
            xticklabels=data["labels"],
            yticks=range(len(data["labels"])),
            yticklabels=data["labels"],
            xlabel="Prior exposure · e_min",
            ylabel="Prior exposure · e_max",
            title=f"Task {panel['position']} · {LABELS[panel['capture']]}",
        )
    if plot is not None:
        fig.colorbar(plot, ax=list(axes.flat), label=LABELS[metric], shrink=0.8)
    fig.suptitle(
        f"{LABELS[family]} · {LABELS[population]}\n"
        "Equal-world mean · gray: insufficient coverage · hatch: impossible",
        fontsize=13,
    )
    fig.savefig(path.with_suffix(".png"), dpi=170)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_probe_results(results: Path | str, out_dir: Path | str) -> list[Path]:
    summary, root = read_json(Path(results)), Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    figures = probe_evaluation_figures(summary)
    paths, links = [], []
    for name, figure in figures.items():
        path = root / (name.replace("/", "--") + ".html")
        # A shared local JS bundle keeps the whole directory usable offline.
        figure.write_html(path, include_plotlyjs="directory")
        paths.append(path)
        links.append(f'<li><a href="{html.escape(path.name)}">{html.escape(name)}</a></li>')
    for population in summary["settings"]["splits"]:
        for family in summary["capture"]["shapes"]:
            if f"{population}/exposure-{family}-r2_macro" in figures:
                for metric in summary["settings"]["exposure"]["metrics"]:
                    path = root / f"{population}--exposure-{family}-{metric}"
                    _save_exposure_static(summary, population, family, metric, path)
                    paths.extend([path.with_suffix(".png"), path.with_suffix(".pdf")])
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Task-vector decoding</title>'
        "<style>body{font:16px system-ui;max-width:1000px;margin:40px auto}li{margin:8px 0}</style>"
        f"<h1>{html.escape(summary['phase'].title())} · task-vector decoding</h1>"
        "<p>Equal-world means with world-bootstrap intervals. See ../report.md for metric"
        " definitions and scientific limits.</p>"
        "<ul>" + "".join(links) + "</ul>"
    )
    return paths
