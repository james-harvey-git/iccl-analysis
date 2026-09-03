from typing import Any, cast

from iccl.reporting.monitor import canonical_monitor_figures, canonical_monitor_scalars


def curve_rows() -> list[dict[str, Any]]:
    specifications = (
        ("icl", "ordinary", "within_task_learning"),
        ("icl", "ordinary", "episode_learning"),
        ("composition", "constituent", "composition_learning"),
        ("composition", "matched_prefix", "composition_learning"),
        ("composition", "no_history", "composition_learning"),
        ("retention", "original", "retention_learning"),
        ("retention", "repeat", "retention_learning"),
        ("retention", "novel", "retention_learning"),
        ("retention", "shared", "retention_learning"),
    )
    return [
        {
            "capability": capability,
            "condition": condition,
            "curve_type": curve_type,
            "x_value": position,
            "nmse": 0.5,
            "ci_low": 0.4,
            "ci_high": 0.6,
        }
        for capability, condition, curve_type in specifications
        for position in range(3)
    ]


def test_canonical_monitor_has_four_fixed_demo_capability_panels() -> None:
    figures = canonical_monitor_figures(curve_rows(), 5000)
    assert set(figures) == {
        "monitor-curves/icl_within_task",
        "monitor-curves/icl_across_episode",
        "monitor-curves/composition_final_task",
        "monitor-curves/retention_final_task",
    }
    retention = cast(Any, figures["monitor-curves/retention_final_task"].data)
    assert len(retention) == 4
    assert {len(trace.x) for trace in retention} == {3}


def test_canonical_monitor_scalars_use_only_requested_summaries() -> None:
    metrics = (
        ("icl", "within_task_nmse_mean"),
        ("composition", "benefit_mean"),
        ("retention", "savings_mean"),
        ("retention", "episodic_savings_mean"),
        ("retention", "module_savings_mean"),
    )
    rows = [
        {"capability": capability, "metric": metric, "value": index / 10}
        for index, (capability, metric) in enumerate(metrics)
    ]
    assert len(canonical_monitor_scalars(rows)) == 5
