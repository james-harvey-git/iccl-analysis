import json
from pathlib import Path

import numpy as np

from iccl.evaluation.metrics import EvaluationReport
from iccl.evaluation.results import read_rows, write_evaluation_results


def test_evaluation_results_round_trip_rows_and_raw_errors(tmp_path: Path) -> None:
    summary = [
        {
            "capability": "icl",
            "condition": "ordinary",
            "metric": "nmse_aulc",
            "value": 0.5,
            "ci_low": 0.4,
            "ci_high": 0.6,
            "M": 8,
            "T": 8,
            "S": 1,
            "n_sequences": 2,
        }
    ]
    curves = [
        {
            "capability": "icl",
            "condition": "ordinary",
            "curve_type": "learning_curve",
            "x_name": "demo_index",
            "x_value": 0,
            "mse": 1.0,
            "nmse": 0.5,
            "ci_low": 0.4,
            "ci_high": 0.6,
            "M": 8,
            "T": 8,
            "S": 1,
            "n_sequences": 2,
        }
    ]
    raw = {"icl__ordinary/mse": np.arange(8).reshape(2, 2, 2)}
    path = write_evaluation_results(
        EvaluationReport({"validation/token_mse": 0.25}, {}, summary, curves, raw),
        tmp_path,
        5000,
        {"checkpoint_reference": "step.pt"},
    )

    assert read_rows(path / "summary.csv")[0]["M"] == 8
    assert read_rows(path / "curves.csv")[0]["x_value"] == 0
    with np.load(path / "raw_errors.npz") as arrays:
        np.testing.assert_array_equal(arrays["icl__ordinary.mse"], raw["icl__ordinary/mse"])
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["step"] == 5000
    assert manifest["raw_arrays"]["icl__ordinary.mse"] == "icl__ordinary/mse"
