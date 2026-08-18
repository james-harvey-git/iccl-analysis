from typing import Any, cast

from iccl.reporting.monitor import canonical_monitor_figures


def test_canonical_retention_curves_share_a_matched_demo_range() -> None:
    rows = [
        {
            "capability": "retention",
            "condition": condition,
            "curve_type": curve_type,
            "x_value": position,
            "nmse": 0.5,
            "ci_low": 0.4,
            "ci_high": 0.6,
        }
        for condition, curve_type, count in (
            ("original", "original_curve", 4),
            ("repeat", "relearning_curve", 2),
            ("novel", "novel_curve", 2),
            ("shared", "shared_curve", 2),
        )
        for position in range(count)
    ]
    figure = canonical_monitor_figures(rows, 5000)["monitor-curves/retention_final_task"]
    data = cast(Any, figure.data)

    assert {len(trace.x) for trace in data} == {2}
