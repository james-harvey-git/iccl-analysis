"""Selection and presentation safeguards for paper learning curves."""

import runpy
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/plotting/plot_learning_curves.py"
namespace = runpy.run_path(str(SCRIPT))
select_curves = namespace["select_curves"]
render_panels = namespace["render_panels"]


def example_rows() -> list[dict[str, Any]]:
    rows = []
    for modules in (8, 9):
        for kind, count, axis in (
            ("within_task_learning", 3, "demo_index"),
            ("episode_learning", 8, "task_position"),
        ):
            for index in range(count):
                mean = 1 / (index + 1) if kind == "within_task_learning" else 0.6 - 0.02 * index
                rows.append(
                    {
                        "capability": "icl",
                        "condition": "ordinary",
                        "curve_type": kind,
                        "M": modules,
                        "T": 8,
                        "D": 3,
                        "x_value": index,
                        "x_name": axis,
                        "nmse": mean,
                        "ci_low": mean - 0.01,
                        "ci_high": mean + 0.02,
                        "step": 2100000,
                        "checkpoint_reference": "test.pt",
                        "suite": "ordinary",
                        "cell_id": f"m{modules:02d}__t08__d003",
                        "n_sequences": 256,
                    }
                )
    return rows[::-1]


def test_selects_one_cell_and_preserves_saved_estimates() -> None:
    curves = select_curves(example_rows(), modules=8, tasks=8, demos=3)
    assert len(curves["within_task_learning"]) == 3
    assert len(curves["episode_learning"]) == 8
    assert {r["M"] for rows in curves.values() for r in rows} == {8}
    assert [r["nmse"] for r in curves["within_task_learning"]] == [1, 0.5, 1 / 3]
    assert curves["episode_learning"][0]["ci_low"] == pytest.approx(0.59)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "checkpoint", "axis", "interval"])
def test_rejects_ambiguous_or_invalid_curves(damage: str) -> None:
    rows = [row for row in example_rows() if row["M"] == 8]
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(dict(rows[0]))
    elif damage == "checkpoint":
        rows[0]["step"] = 1900000
    elif damage == "axis":
        rows[0]["x_name"] = "demo_index"
    else:
        rows[0]["ci_low"] = 2.0
    with pytest.raises(ValueError):
        select_curves(rows, modules=8, tasks=8, demos=3)


def test_panels_preserve_data_and_have_independent_latex_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    close = plt.close

    def capture_close(figure: Any = None) -> None:
        if hasattr(figure, "axes"):
            figures.append(figure)
        close(figure)

    monkeypatch.setattr(plt, "close", capture_close)
    curves = select_curves(example_rows(), modules=8, tasks=8, demos=3)
    paths = render_panels(curves, tmp_path, show_ci=True)
    assert len(figures) == 2
    for figure, kind in zip(figures, ("within_task_learning", "episode_learning"), strict=True):
        np.testing.assert_allclose(figure.get_size_inches(), [3.3, 2.5])
        axis = figure.axes[0]
        np.testing.assert_array_equal(
            axis.lines[0].get_xdata(), np.arange(1, len(curves[kind]) + 1)
        )
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [row["nmse"] for row in curves[kind]])
        assert len(axis.collections) == 1
    assert all(path.stat().st_size > 100 for path in paths)
    snippet = (tmp_path / "learning-figure.tex").read_text()
    assert r"\label{fig:icl-within-task}" in snippet
    assert r"\label{fig:iccl-across-tasks}" in snippet
    assert r"95\% pointwise" in snippet
    with ZipFile(tmp_path / "learning-panels.zip") as archive:
        assert set(archive.namelist()) == {
            "figures/icl-within-task.pdf",
            "figures/iccl-across-tasks.pdf",
            "learning-figure.tex",
        }
