"""Compact Plotly figures shared by local evaluation output and W&B reporting."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import plotly.graph_objects as go

from iccl.visualization import grouped_figure

PRIMARY_METRICS = {
    "within_task_nmse_mean": "within-task ICL nMSE",
    "benefit_mean": "composition benefit",
    "savings_mean": "retention savings",
    "episodic_savings_mean": "episodic savings",
    "module_savings_mean": "module savings",
}


def _has_family(row: dict[str, Any], family: str) -> bool:
    return family in str(row.get("family_memberships", "")).split("|")


def _trace_label(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    labels = {
        "constituent": "constituent history",
        "matched_prefix": "matched-prefix control",
        "no_history": "no-history control",
        "original": "original learning",
        "repeat": "repeat/relearning",
        "novel": "novel-task control",
        "shared": "same-support/new-weights control",
        "total": "total savings",
        "episodic": "episodic savings",
        "module": "module savings",
        "none": "no constituent rehearsal",
        "one": "one constituent rehearsed",
        "both": "both constituents rehearsed",
    }
    return labels.get(str(value), str(value))


def _scatter(rows: list[dict[str, Any]], name: str, *, visible: bool) -> go.Scatter:
    ordered = sorted(rows, key=lambda row: int(row["x_value"]))
    values = [float(row["nmse"]) for row in ordered]
    hover = [
        [
            row["M"],
            row["T"],
            row["S"],
            row["D"],
            row["family_memberships"],
            row["module_count_status"],
            row["sampler"],
            row["weighting"],
            row["n_sequences"],
        ]
        for row in ordered
    ]
    return go.Scatter(
        x=[row["x_value"] for row in ordered],
        y=values,
        mode="lines+markers",
        name=name,
        visible=visible,
        error_y={
            "type": "data",
            "array": [
                max(0.0, float(row["ci_high"]) - value)
                for row, value in zip(ordered, values, strict=True)
            ],
            "arrayminus": [
                max(0.0, value - float(row["ci_low"]))
                for row, value in zip(ordered, values, strict=True)
            ],
            "symmetric": False,
        },
        customdata=hover,
        hovertemplate=(
            "M=%{customdata[0]}<br>T=%{customdata[1]}<br>S=%{customdata[2]}"
            "<br>D=%{customdata[3]}<br>families=%{customdata[4]}"
            "<br>status=%{customdata[5]}<br>sampler=%{customdata[6]}"
            "<br>weighting=%{customdata[7]}<br>N=%{customdata[8]}<extra></extra>"
        ),
    )


def _cell_dropdown(
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_title: str,
    group_field: str,
) -> go.Figure:
    cells = sorted({str(row["cell_id"]) for row in rows})
    figure = go.Figure()
    trace_cells: list[str] = []
    for cell in cells:
        selected = [row for row in rows if row["cell_id"] == cell]
        groups = sorted({str(row.get(group_field)) for row in selected})
        for group in groups:
            group_rows = [row for row in selected if str(row.get(group_field)) == group]
            figure.add_trace(
                _scatter(
                    group_rows,
                    _trace_label(group_rows[0], group_field),
                    visible=cell == cells[0],
                )
            )
            trace_cells.append(cell)
    buttons = []
    for cell in cells:
        row = next(row for row in rows if row["cell_id"] == cell)
        label = f"M={row['M']}, T={row['T']}, S={row['S']}, D={row['D']}"
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": [trace_cell == cell for trace_cell in trace_cells]}],
            }
        )
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="normalized MSE",
        template="plotly_white",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "xanchor": "right"}],
    )
    return figure


def _rehearsal_figure(rows: list[dict[str, Any]]) -> go.Figure:
    """Controlled rehearsal traces with a savings-component selector."""
    components = [
        component
        for component in ("total", "module", "episodic")
        if any(row.get("retention_component") == component for row in rows)
    ]
    groups = sorted({(row["retention_component"], row["rehearsal_mode"]) for row in rows}, key=str)
    figure = grouped_figure(
        rows,
        title=f"Controlled constituent rehearsal — {components[0]} savings",
        x_field="x_value",
        y_field="nmse",
        x_title="original target-task position",
        y_title="mean nMSE savings across revisit demos",
        group_fields=("retention_component", "rehearsal_mode"),
        hover_fields=("intervening_tasks", "support_status", "n_sequences"),
    )
    for raw_trace, (component, mode) in zip(figure.data, groups, strict=True):
        trace = cast(Any, raw_trace)
        selected = sorted(
            (
                row
                for row in rows
                if row["retention_component"] == component and row["rehearsal_mode"] == mode
            ),
            key=lambda row: int(row["x_value"]),
        )
        trace.name = _trace_label(selected[0], "rehearsal_mode")
        trace.visible = component == components[0]
        trace.marker.symbol = [
            "x" if row["support_status"] == "disconnected_ood" else "circle" for row in selected
        ]
    buttons = [
        {
            "label": _trace_label({"retention_component": component}, "retention_component"),
            "method": "update",
            "args": [
                {"visible": [value == component for value, _ in groups]},
                {"title": f"Controlled constituent rehearsal — {component} savings"},
            ],
        }
        for component in components
    ]
    figure.update_layout(updatemenus=[{"buttons": buttons, "x": 1.0, "xanchor": "right"}])
    figure.add_hline(y=0, line_dash="dot", line_color="gray")
    return figure


def _task_summary(rows: list[dict[str, Any]]) -> go.Figure:
    modules = sorted({int(row["M"]) for row in rows})
    figure = go.Figure()
    trace_modules: list[int] = []
    for modules_value in modules:
        selected = [row for row in rows if int(row["M"]) == modules_value]
        for metric in PRIMARY_METRICS:
            metric_rows = sorted(
                (row for row in selected if row["metric"] == metric), key=lambda row: int(row["S"])
            )
            if not metric_rows:
                continue
            values = [float(row["value"]) for row in metric_rows]
            figure.add_trace(
                go.Scatter(
                    x=[row["S"] for row in metric_rows],
                    y=values,
                    mode="lines+markers",
                    name=PRIMARY_METRICS[metric],
                    visible=modules_value == modules[0],
                    error_y={
                        "type": "data",
                        "array": [
                            max(0.0, float(row["ci_high"]) - value)
                            for row, value in zip(metric_rows, values, strict=True)
                        ],
                        "arrayminus": [
                            max(0.0, value - float(row["ci_low"]))
                            for row, value in zip(metric_rows, values, strict=True)
                        ],
                        "symmetric": False,
                    },
                )
            )
            trace_modules.append(modules_value)
    buttons = [
        {
            "label": f"M={modules_value}",
            "method": "update",
            "args": [{"visible": [value == modules_value for value in trace_modules]}],
        }
        for modules_value in modules
    ]
    figure.update_layout(
        title="Task variation",
        xaxis_title="surplus tasks S",
        yaxis_title="metric value",
        template="plotly_white",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.0, "xanchor": "right"}],
    )
    return figure


def evaluation_figures(
    summary_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]]
) -> dict[str, go.Figure]:
    """Build the configured compact evaluation and diagnostic figures."""
    primary = [row for row in summary_rows if row["metric"] in PRIMARY_METRICS]
    task_rows = [row for row in primary if _has_family(row, "task_variation")]
    module_rows = [row for row in primary if _has_family(row, "module_variation")]
    figures: dict[str, go.Figure] = {}
    if task_rows:
        figures["evaluation/task_variation_summary"] = _task_summary(task_rows)
    if module_rows:
        figures["evaluation/module_variation_summary"] = grouped_figure(
            module_rows,
            title="Module variation at fixed task count",
            x_field="M",
            y_field="value",
            x_title="module count M",
            y_title="metric value",
            group_fields=("metric",),
            trace_names=PRIMARY_METRICS,
            hover_fields=("T", "S", "D", "module_count_status"),
        )

    position_rows = [row for row in curve_rows if row["curve_type"] == "retention_position"]
    if position_rows:
        figures["evaluation/retention_position"] = grouped_figure(
            position_rows,
            title="Paired retention by original task position",
            x_field="x_value",
            y_field="nmse",
            x_title="original target-task position",
            y_title="mean nMSE savings across revisit demos",
            group_fields=("retention_component",),
            trace_names={
                "total": "total savings",
                "module": "module savings",
                "episodic": "episodic savings",
            },
            hover_fields=("intervening_tasks", "n_sequences"),
        )
        figures["evaluation/retention_position"].add_hline(y=0, line_dash="dot", line_color="gray")
    rehearsal_rows = [row for row in curve_rows if row["curve_type"] == "retention_rehearsal"]
    if rehearsal_rows:
        figures["evaluation/retention_rehearsal"] = _rehearsal_figure(rehearsal_rows)

    panels = (
        (
            "evaluation/icl_within_task",
            "within_task_learning",
            "Within-task ICL",
            "demo index",
            "condition",
        ),
        (
            "evaluation/iccl_across_episode",
            "episode_learning",
            "ICCL across the episode",
            "task position",
            "condition",
        ),
        (
            "evaluation/composition_controls",
            "composition_learning",
            "Composition controls",
            "final-task demo index",
            "condition",
        ),
        (
            "evaluation/retention_controls",
            "retention_learning",
            "Retention controls",
            "demo index",
            "condition",
        ),
        (
            "evaluation/retention_savings",
            "retention_savings",
            "Retention savings",
            "demo index",
            "retention_component",
        ),
        (
            "evaluation/retention_vs_intervening_tasks",
            "retention_delay",
            "Retention versus delay",
            "intervening tasks",
            "retention_component",
        ),
    )
    for key, curve_type, title, x_title, group_field in panels:
        rows = [row for row in curve_rows if row["curve_type"] == curve_type]
        if rows:
            figures[key] = _cell_dropdown(
                rows, title=title, x_title=x_title, group_field=group_field
            )
    return figures


def write_html_figures(figures: Iterable[tuple[str, go.Figure]], out_dir: Path | str) -> int:
    """Write standalone HTML figures using key-shaped subdirectories."""
    root = Path(out_dir)
    count = 0
    for key, figure in figures:
        path = root / f"{key.removeprefix('evaluation/')}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(path, include_plotlyjs="directory")
        count += 1
    return count
