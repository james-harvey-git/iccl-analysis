from typing import Any, cast

from iccl.analysis.evaluation_plots import full_evaluation_figures


def test_module_count_summary_is_one_curve_across_module_counts() -> None:
    rows = [
        {
            "capability": "icl",
            "condition": "ordinary",
            "metric": "nmse_aulc",
            "slice": "fixed_surplus",
            "status": "seen",
            "variant": "",
            "M": modules,
            "T": modules,
            "B_history": 32 * modules,
            "L_history": 65 * modules,
            "value": value,
            "ci_low": value - 0.1,
            "ci_high": value + 0.1,
        }
        for modules, value in ((4, 0.8), (8, 0.6), (12, 0.5))
    ]
    figure = full_evaluation_figures(rows, [])["capability/icl/aulc_vs_M"]

    data = cast(Any, figure.data)
    assert len(data) == 1
    assert list(data[0].x) == [4, 8, 12]


def test_detailed_curve_uses_its_recorded_axis_name() -> None:
    rows = [
        {
            "capability": "icl",
            "condition": "ordinary",
            "curve_type": "task_position_curve",
            "x_name": "task_position",
            "x_value": position,
            "nmse": 0.5,
            "ci_low": 0.4,
            "ci_high": 0.6,
            "slice": "fixed_surplus",
            "status": "seen",
            "variant": "",
            "M": 8,
            "T": 8,
        }
        for position in range(2)
    ]
    figure = full_evaluation_figures([], rows)["icl/nmse_by_task_position"]

    assert figure.layout.xaxis.title.text == "task position"
