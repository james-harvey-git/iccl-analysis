"""Compact scalar and Plotly views logged while training."""

from typing import Any

import plotly.graph_objects as go

from iccl.visualization import grouped_figure


def canonical_monitor_scalars(summary_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Longitudinal capability summaries for the configured monitor cell."""
    specifications = {
        "monitor/icl_nmse_aulc": ("icl", "ordinary", "nmse_aulc"),
        "monitor/composition_nmse_benefit": ("composition", "benefit", "benefit_mean"),
        "monitor/retention_total_nmse_savings": ("retention", "savings", "savings_mean"),
        "monitor/retention_episodic_nmse_savings": (
            "retention",
            "episodic_savings",
            "episodic_savings_mean",
        ),
        "monitor/retention_module_nmse_savings": (
            "retention",
            "module_savings",
            "module_savings_mean",
        ),
    }
    metrics: dict[str, float] = {}
    for key, (capability, condition, metric) in specifications.items():
        matches = [
            row
            for row in summary_rows
            if row["capability"] == capability
            and row["condition"] == condition
            and row["metric"] == metric
        ]
        if len(matches) > 1:
            raise ValueError(f"canonical monitor has multiple rows for {key}")
        if matches:
            metrics[key] = float(matches[0]["value"])
    return metrics


def canonical_monitor_figures(curve_rows: list[dict[str, Any]], step: int) -> dict[str, go.Figure]:
    """Four NMSE panels for one canonical structural cell and training step."""
    specifications = (
        (
            "monitor-curves/icl_within_task",
            "ICL within task",
            "demo index within task",
            {"learning_curve": "ICL"},
        ),
        (
            "monitor-curves/icl_across_episode",
            "ICL across episode",
            "task position",
            {"task_position_curve": "ICL"},
        ),
        (
            "monitor-curves/composition_final_task",
            "Composition on final task",
            "final-task demo index",
            {
                "constituent_curve": "constituent history",
                "matched_prefix_curve": "matched-prefix control",
                "no_history_curve": "no-history control",
            },
        ),
        (
            "monitor-curves/retention_final_task",
            "Retention on final task",
            "final-task demo index",
            {
                "original_curve": "original learning",
                "relearning_curve": "repeat/relearning",
                "novel_curve": "novel-task control",
                "shared_curve": "same-support/new-weights control",
            },
        ),
    )
    figures: dict[str, go.Figure] = {}
    for key, title, x_title, names in specifications:
        rows = [row for row in curve_rows if row["curve_type"] in names]
        if key == "monitor-curves/retention_final_task" and rows:
            maxima = {
                curve_type: max(
                    int(row["x_value"]) for row in rows if row["curve_type"] == curve_type
                )
                for curve_type in {str(row["curve_type"]) for row in rows}
            }
            matched_last_demo = min(maxima.values())
            rows = [row for row in rows if int(row["x_value"]) <= matched_last_demo]
        if rows:
            figures[key] = grouped_figure(
                rows,
                title=f"{title} — training step {step:,}",
                x_field="x_value",
                y_field="nmse",
                x_title=x_title,
                y_title="normalized MSE",
                group_fields=("curve_type",),
                trace_names=names,
            )
    return figures
