import json
from pathlib import Path
from typing import Any, cast

from iccl.reporting.figures import evaluation_figures, write_html_figures


def summary_row(metric: str, modules: int, surplus: int, families: str) -> dict[str, Any]:
    value = 0.1 * modules + surplus
    return {
        "capability": "icl" if metric == "within_task_nmse_mean" else "retention",
        "condition": "ordinary",
        "metric": metric,
        "family_memberships": families,
        "cell_id": f"m{modules:02d}__t{modules - 1 + surplus:02d}__d032",
        "module_count_status": "seen",
        "sampler": "constructive",
        "weighting": "discrete",
        "M": modules,
        "T": modules - 1 + surplus,
        "S": surplus,
        "D": 32,
        "value": value,
        "ci_low": value - 0.1,
        "ci_high": value + 0.1,
        "n_sequences": 32,
    }


def curve_row(cell: str, modules: int, curve_type: str, position: int) -> dict[str, Any]:
    return {
        "capability": "icl",
        "condition": "ordinary",
        "curve_type": curve_type,
        "retention_component": None,
        "family_memberships": "task_variation",
        "cell_id": cell,
        "module_count_status": "seen",
        "sampler": "constructive",
        "weighting": "discrete",
        "M": modules,
        "T": modules,
        "S": 1,
        "D": 32,
        "x_name": "demo_index",
        "x_value": position,
        "mse": 0.7,
        "nmse": 0.5,
        "ci_low": 0.4,
        "ci_high": 0.6,
        "n_sequences": 32,
    }


def test_summary_and_detail_panels_are_compact_and_interactive(tmp_path: Path) -> None:
    summary = [
        summary_row(metric, modules, surplus, "task_variation|module_variation")
        for metric in ("within_task_nmse_mean", "savings_mean")
        for modules in (4, 5)
        for surplus in (0, 1)
    ]
    curves = [
        curve_row(f"m{modules:02d}__t{modules:02d}__d032", modules, curve_type, position)
        for curve_type in ("within_task_learning", "episode_learning")
        for modules in (4, 5)
        for position in range(2)
    ]
    figures = evaluation_figures(summary, curves)
    assert set(figures) == {
        "evaluation/task_variation_summary",
        "evaluation/module_variation_summary",
        "evaluation/icl_within_task",
        "evaluation/iccl_across_episode",
    }
    detail = figures["evaluation/icl_within_task"]
    assert len(cast(Any, detail.layout).updatemenus[0].buttons) == 2
    assert sum(bool(trace.visible) for trace in cast(Any, detail.data)) == 1

    assert write_html_figures(figures.items(), tmp_path) == 4
    assert (tmp_path / "icl_within_task.html").exists()
    assert (tmp_path / "plotly.min.js").exists()


def test_installed_wandb_adapter_preserves_plotly_dropdowns() -> None:
    import plotly.graph_objects as go
    import wandb

    figure = go.Figure()
    figure.update_layout(
        updatemenus=[
            {"buttons": [{"label": "cell", "method": "update", "args": [{"visible": [True]}]}]}
        ]
    )
    media = wandb.Plotly(figure)
    serialized = json.loads(Path(vars(media)["_path"]).read_text())
    assert serialized["layout"]["updatemenus"][0]["buttons"][0]["label"] == "cell"


def test_position_diagnostic_adds_only_two_compact_figures() -> None:
    rows = []
    for component in ("total", "module", "episodic"):
        for position in range(4):
            rows.append(
                curve_row("m04__t04__d032", 4, "retention_position", position)
                | {
                    "capability": "retention_position",
                    "condition": "savings",
                    "diagnostic_family": "paired_permutation",
                    "retention_component": component,
                    "original_task_position": position,
                    "intervening_tasks": 3 - position,
                }
            )
        for mode in ("none", "one", "both"):
            for position in (0, 1):
                rows.append(
                    curve_row("m04__t04__d032", 4, "retention_rehearsal", position)
                    | {
                        "capability": "retention_position",
                        "condition": mode,
                        "diagnostic_family": "controlled_rehearsal",
                        "retention_component": component,
                        "rehearsal_mode": mode,
                        "original_task_position": position,
                        "intervening_tasks": 3 - position,
                        "support_status": (
                            "disconnected_ood"
                            if mode == "none" and position == 0
                            else "connected_id"
                        ),
                    }
                )

    figures = evaluation_figures([], rows)
    assert set(figures) == {
        "evaluation/retention_position",
        "evaluation/retention_rehearsal",
    }
    assert len(cast(Any, figures["evaluation/retention_position"].data)) == 3
    rehearsal = figures["evaluation/retention_rehearsal"]
    assert len(cast(Any, rehearsal.data)) == 9
    assert len(cast(Any, rehearsal.layout).updatemenus[0].buttons) == 3
    assert sum(bool(trace.visible) for trace in cast(Any, rehearsal.data)) == 3
    none = next(
        trace for trace in cast(Any, rehearsal.data) if trace.name == "no constituent rehearsal"
    )
    assert "x" in list(none.marker.symbol)
