"""Compact scalar and Plotly views for the canonical training monitor."""

from typing import Any

import plotly.graph_objects as go

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
    """Four nMSE panels for the canonical cell at one training step."""
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
            )
    return figures
