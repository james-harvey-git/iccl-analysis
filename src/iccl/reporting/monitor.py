"""Compact scalar and Plotly views for the canonical training monitor."""

from typing import Any

import plotly.graph_objects as go

from iccl.reporting.figures import _trace_label
from iccl.visualization import grouped_figure


def canonical_monitor_scalars(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Longitudinal summaries from the canonical fixed-demo capability cell."""
    specifications = {
        "monitor/icl_within_task_nmse_mean": ("icl", "within_task_nmse_mean"),
        "monitor/composition_nmse_benefit": ("composition", "benefit_mean"),
        "monitor/retention_total_nmse_savings": ("retention", "savings_mean"),
        "monitor/retention_episodic_nmse_savings": (
            "retention",
            "episodic_savings_mean",
        ),
        "monitor/retention_module_nmse_savings": ("retention", "module_savings_mean"),
    }
    metrics: dict[str, float] = {}
    for key, (capability, metric) in specifications.items():
        matches = [
            row
            for row in summary_rows
            if row["capability"] == capability and row["metric"] == metric
        ]
        if len(matches) > 1:
            raise ValueError(f"canonical monitor has multiple rows for {key}")
        if matches:
            metrics[key] = float(matches[0]["value"])
    return metrics


def canonical_monitor_figures(curve_rows: list[dict[str, Any]], step: int) -> dict[str, go.Figure]:
    """Canonical nMSE panels for the canonical cell at one training step."""
    specifications = (
        (
            "monitor-curves/icl_within_task",
            "Within-task ICL",
            "within_task_learning",
            "demo index within task",
            ("condition",),
        ),
        (
            "monitor-curves/icl_across_episode",
            "ICCL across episode",
            "episode_learning",
            "task position",
            ("condition",),
        ),
        (
            "monitor-curves/composition_final_task",
            "Composition on final task",
            "composition_learning",
            "final-task demo index",
            ("condition",),
        ),
        (
            "monitor-curves/retention_final_task",
            "Retention on final task",
            "retention_learning",
            "final-task demo index",
            ("condition",),
        ),
    )
    figures: dict[str, go.Figure] = {}
    for key, title, curve_type, x_title, groups in specifications:
        rows = [row for row in curve_rows if row["curve_type"] == curve_type]
        if rows:
            figures[key] = grouped_figure(
                rows,
                title=f"{title} — training step {step:,}",
                x_field="x_value",
                y_field="nmse",
                x_title=x_title,
                y_title="normalized MSE",
                group_fields=groups,
                trace_names={row["condition"]: _trace_label(row, "condition") for row in rows},
                hover_fields=("M", "T", "D", "n_sequences", "sample_scope"),
            )
    for condition, title in (
        ("unexposed", "Unexposed final-task error"),
        ("repeat", "Exact-repeat final-task error"),
        ("savings", "Total retention savings"),
    ):
        rows = [
            row
            for row in curve_rows
            if (row["curve_type"] == "retention_error_delay" and row["condition"] == condition)
            or (
                condition == "savings"
                and row["curve_type"] == "retention_delay"
                and row.get("retention_component") == "total"
            )
        ]
        if not rows:
            continue
        label = "total_savings" if condition == "savings" else condition
        key = f"monitor-curves/retention_{label}_vs_delay"
        figures[key] = grouped_figure(
            rows,
            title=f"{title} — training step {step:,}",
            x_field="x_value",
            y_field="nmse",
            x_title="intervening tasks",
            y_title="normalized MSE",
            group_fields=(),
            hover_fields=("M", "T", "D", "n_sequences", "original_task_position", "sample_scope"),
        )
        if condition == "savings":
            figures[key].add_hline(y=0, line_dash="dot", line_color="gray")
            figures[key].update_yaxes(rangemode="tozero")
    error_keys = [
        f"monitor-curves/retention_{condition}_vs_delay" for condition in ("unexposed", "repeat")
    ]
    if all(key in figures for key in error_keys):
        rows = [
            row
            for row in curve_rows
            if row["curve_type"] == "retention_error_delay"
            and row["condition"] in {"unexposed", "repeat"}
        ]
        upper = max(float(row["ci_high"]) for row in rows)
        for key in error_keys:
            figures[key].update_yaxes(range=[0, max(upper * 1.05, 1e-6)])
    return figures
