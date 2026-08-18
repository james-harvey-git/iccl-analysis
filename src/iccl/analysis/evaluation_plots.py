"""Plotly views reconstructed from evaluation summary and curve rows."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from iccl.visualization import grouped_figure


def full_evaluation_figures(
    summary_rows: list[dict[str, Any]], curve_rows: list[dict[str, Any]]
) -> dict[str, go.Figure]:
    """Detailed structural plots reconstructed from an evaluation artifact."""
    figures = _summary_figures(summary_rows)
    figures.update(_curve_figures(curve_rows))
    return figures


def _summary_figures(summary_rows: list[dict[str, Any]]) -> dict[str, go.Figure]:
    capability_metrics = {
        "icl": ("ordinary", "nmse_aulc", "aulc"),
        "composition": ("benefit", "benefit_mean", "benefit_mean"),
        "retention": ("savings", "savings_mean", "savings_mean"),
    }
    axes: dict[str, tuple[str, set[str]]] = {
        "M": ("M", {"fixed_surplus", "matched_task_count", "matched_prediction_tokens"}),
        "task_count": ("T", {"task_count", "matched_serialized_prefix"}),
        "history_tokens": (
            "L_history",
            {"task_count", "history_demos", "matched_prediction_tokens"},
        ),
        "history_demos": ("D_mean", {"history_demos", "demo_allocation"}),
        "demo_allocation": ("variant", {"demo_allocation"}),
    }
    figures: dict[str, go.Figure] = {}
    for capability, (condition, metric, label) in capability_metrics.items():
        for axis, (field, slices) in axes.items():
            rows = [
                row
                for row in summary_rows
                if row["capability"] == capability
                and row["condition"] == condition
                and row["metric"] == metric
                and row.get(field) is not None
                and row["slice"] in slices
            ]
            if rows:
                key = f"capability/{capability}/{label}_vs_{axis}"
                figures[key] = grouped_figure(
                    rows,
                    title=key,
                    x_field=field,
                    y_field="value",
                    x_title=field.replace("_", " "),
                    y_title=metric,
                    group_fields=("slice",),
                    hover_fields=("status", "M", "T", "B_history", "L_history", "variant"),
                )
    extra = {
        (
            "composition",
            "constituent_exposure_min",
        ): "composition/benefit_vs_constituent_task_exposure",
        (
            "composition",
            "constituent_demo_exposure_min",
        ): "composition/benefit_vs_constituent_demo_exposure",
        ("retention", "intervening_tasks"): "retention/savings_vs_intervening_tasks",
        ("retention", "prediction_token_delay"): "retention/savings_vs_prediction_token_delay",
        ("retention", "serialized_token_delay"): "retention/savings_vs_serialized_token_delay",
    }
    for (capability, field), key in extra.items():
        condition, metric = (
            ("benefit", "benefit_mean")
            if capability == "composition"
            else ("savings", "savings_mean")
        )
        rows = [
            row
            for row in summary_rows
            if row["capability"] == capability
            and row["condition"] == condition
            and row["metric"] == metric
            and row.get(field) is not None
        ]
        if rows:
            figures[key] = grouped_figure(
                rows,
                title=key,
                x_field=field,
                y_field="value",
                x_title=field.replace("_", " "),
                y_title=metric,
                group_fields=("slice",),
                hover_fields=("status", "M", "T", "B_history", "L_history", "variant"),
            )
    return figures


def _curve_figures(curve_rows: list[dict[str, Any]]) -> dict[str, go.Figure]:
    mappings: dict[tuple[str, str], str] = {
        ("icl", "learning_curve"): "capability/icl/learning_curve",
        ("icl", "task_position_curve"): "icl/nmse_by_task_position",
        ("icl", "nmse_by_prediction_tokens_observed"): "icl/nmse_by_prediction_tokens_observed",
        ("icl", "nmse_by_unique_supports_observed"): "icl/nmse_by_unique_supports_observed",
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
        ("retention", "episodic_savings_curve"): "capability/retention/episodic_savings_curve",
        ("retention", "module_savings_curve"): "capability/retention/module_savings_curve",
    }
    figures: dict[str, go.Figure] = {}
    for (capability, curve_type), key in mappings.items():
        rows = [
            row
            for row in curve_rows
            if row["capability"] == capability and row["curve_type"] == curve_type
        ]
        if rows:
            x_names = {str(row["x_name"]) for row in rows}
            figures[key] = grouped_figure(
                rows,
                title=key,
                x_field="x_value",
                y_field="nmse",
                x_title=(
                    next(iter(x_names)).replace("_", " ") if len(x_names) == 1 else "position"
                ),
                y_title="normalized MSE",
                group_fields=("slice", "status", "M", "T", "condition", "variant"),
                hover_fields=("status", "M", "T", "B_history", "L_history", "variant"),
            )
    return figures


def write_html_figures(figures: Iterable[tuple[str, go.Figure]], out_dir: Path | str) -> int:
    """Write standalone HTML figures using key-shaped subdirectories."""
    root = Path(out_dir)
    count = 0
    for key, figure in figures:
        path = root / f"{key}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(path, include_plotlyjs="cdn")
        count += 1
    return count
