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
