import json
from pathlib import Path

import numpy as np

from iccl.evaluation.metrics import METRIC_VERSION, EvaluationReport
from iccl.evaluation.results import read_rows, write_evaluation_results


def structural_row() -> dict[str, object]:
    return {
        "family_memberships": "canonical|task_variation",
        "cell_id": "m08__t08__d032",
        "suite": "icl__ordinary__m08__t08__d032",
        "module_count_status": "seen",
        "sampler": "constructive",
        "weighting": "discrete",
        "M": 8,
        "T": 8,
        "S": 1,
        "D": 32,
        "capability": "icl",
        "condition": "ordinary",
        "retention_component": None,
        "original_task_position": None,
        "intervening_tasks": None,
        "ci_low": 0.4,
        "ci_high": 0.6,
        "n_sequences": 2,
    }


def test_evaluation_results_round_trip_rows_and_raw_metadata(tmp_path: Path) -> None:
    summary = [structural_row() | {"metric": "within_task_nmse_mean", "value": 0.5}]
    curves = [
        structural_row()
        | {
            "curve_type": "within_task_learning",
            "x_name": "demo_index",
            "x_value": 0,
            "mse": 1.0,
            "nmse": 0.5,
        }
    ]
    raw = {"icl__ordinary/mse": np.arange(8).reshape(2, 2, 2)}
    path = write_evaluation_results(
        EvaluationReport({"validation/token_mse": 0.25}, {}, summary, curves, raw),
        tmp_path,
        5000,
        {"checkpoint_reference": "step.pt"},
    )

    summary_row = read_rows(path / "summary.csv")[0]
    assert summary_row["M"] == 8
    assert summary_row["checkpoint_reference"] == "step.pt"
    assert read_rows(path / "curves.csv")[0]["x_value"] == 0
    with np.load(path / "raw_errors.npz") as arrays:
        np.testing.assert_array_equal(arrays["icl__ordinary.mse"], raw["icl__ordinary/mse"])
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["step"] == 5000
    assert manifest["metric_version"] == METRIC_VERSION
    assert "savings_mean" in manifest["metric_definitions"]
    assert manifest["raw_arrays"]["icl__ordinary.mse"] == {
        "logical_name": "icl__ordinary/mse",
        "shape": [2, 2, 2],
        "dtype": "int64",
    }
    assert {"python", "numpy", "torch", "platform"} <= manifest["runtime"].keys()


def test_diagnostic_columns_round_trip_without_affecting_standard_rows(tmp_path: Path) -> None:
    row = structural_row() | {
        "capability": "retention_position",
        "condition": "rehearsal_effect",
        "metric": "rehearsal_effect_mean",
        "value": 0.2,
        "diagnostic_family": "controlled_rehearsal",
        "rehearsal_mode": "one",
        "support_status": "connected_id",
        "original_task_position": 3,
        "intervening_tasks": 4,
    }
    path = write_evaluation_results(
        EvaluationReport({}, {}, [row], [], {}), tmp_path, 10, {"checkpoint_reference": "x.pt"}
    )
    restored = read_rows(path / "summary.csv")[0]
    assert restored["diagnostic_family"] == "controlled_rehearsal"
    assert restored["rehearsal_mode"] == "one"
    assert restored["support_status"] == "connected_id"
    assert restored["original_task_position"] == 3
