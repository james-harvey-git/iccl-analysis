"""Paper scripts and shared caches consume the evaluator's saved estimates."""

import json
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from test_retention_contract import bundle_path as bundle_path

from iccl.checkpoints import checkpoint_model_digest
from iccl.evaluation.metrics import _evaluate, demo_mse, demo_nmse, load_eval_suites
from iccl.evaluation.results import (
    evaluation_identity,
    read_evaluation_results,
    read_rows,
    validate_cached_results,
    write_evaluation_results,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/plotting"


@pytest.fixture
def result(bundle_path: Path, tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    suites = load_eval_suites(
        bundle_path, select=lambda m: m["capability"] in {"icl", "composition", "retention"}
    )
    mses, nmses = {}, {}
    for name, suite in suites.items():
        predictions = np.zeros_like(suite["targets"])
        mses[name], nmses[name] = demo_mse(predictions, suite), demo_nmse(predictions, suite)
    report = _evaluate(suites, mses, nmses, {}, bootstrap_seed=2, bootstrap_replicates=20)
    checkpoint = {
        "step": 100,
        "config": {"model": {"backend": "reference"}, "data": {"input_dim": 4, "output_dim": 4}},
        "model": {"weight": torch.zeros(2)},
        "wandb_run": {"entity": "e", "project": "p", "run_id": "r", "name": "test"},
    }
    bundle = json.loads((bundle_path / "manifest.json").read_text())
    identity = evaluation_identity(
        checkpoint,
        suites,
        bundle,
        batch_size=32,
        bootstrap_seed=2,
        bootstrap_replicates=20,
        backend="reference",
        device=torch.device("cpu"),
        dtype=None,
    )
    path = write_evaluation_results(
        report,
        tmp_path,
        100,
        {
            "checkpoint_reference": "test.pt",
            "evaluation_identity": identity,
            "suites": {name: suite["__meta__"] for name, suite in suites.items()},
        },
    )
    return path, identity, suites


def test_retention_and_composition_paper_readers_preserve_saved_curves(
    result: tuple, tmp_path: Path
) -> None:
    path, _, _ = result
    curves, summaries = read_rows(path / "curves.csv"), read_rows(path / "summary.csv")
    retention = runpy.run_path(str(SCRIPTS / "plot_retention_learning.py"))
    learning, savings, summary = retention["select_retention"](
        curves, summaries, modules=8, tasks=8, demos=2
    )
    assert set(learning) == {"original", "repeat", "shared", "unexposed"}
    assert set(savings) == {"total", "module", "episodic"}
    paths = retention["render_panels"](learning, summary, tmp_path / "retention", show_ci=True)
    assert all(p.stat().st_size > 100 for p in paths)
    assert retention["audit_configurations"](curves, summaries)
    composition = runpy.run_path(str(SCRIPTS / "plot_composition_learning.py"))
    selected, benefit = composition["select_composition"](
        curves,
        summaries,
        json.loads((path / "manifest.json").read_text()),
        modules=8,
        tasks=8,
        demos=2,
    )
    assert set(selected) == {"exposed", "unexposed", "no_history", "benefit"}
    assert benefit["metric"] == "benefit_mean"
    del selected["no_history"]
    assert composition["render_panels"](selected, tmp_path / "composition", show_ci=True)


def test_retention_plot_rejects_monitor_scope_and_handles_binary(
    result: tuple, tmp_path: Path
) -> None:
    path, _, _ = result
    curves, summaries = read_rows(path / "curves.csv"), read_rows(path / "summary.csv")
    script = runpy.run_path(str(SCRIPTS / "plot_retention_learning.py"))

    def kept(row: dict[str, Any]) -> bool:
        return row["condition"] not in {"shared", "module_savings", "episodic_savings"} and row.get(
            "retention_component"
        ) not in {"module", "episodic"}

    curves = [r for r in curves if kept(r)]
    summaries = [r for r in summaries if kept(r)]
    learning, savings, summary = script["select_retention"](
        curves, summaries, modules=8, tasks=8, demos=2
    )
    assert set(savings) == {"total"}
    assert script["render_panels"](learning, summary, tmp_path / "binary", show_ci=False)
    for row in curves:
        if row["capability"] == "retention":
            row["sample_scope"] = "monitor"
    with pytest.raises(ValueError, match="full paired-world"):
        script["select_retention"](curves, summaries, modules=8, tasks=8, demos=2)


def test_trajectory_reads_full_evaluation_errors_without_rebootstrapping(
    result: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, identity, suites = result
    script = runpy.run_path(str(SCRIPTS / "plot_retention_trajectory.py"))
    cached_identity, rows = script["cached_curve"](path)
    assert [r["x_value"] for r in rows] == list(range(8))
    assert cached_identity["source_run"] == "e/p/r"
    assert len(cached_identity["suite_files"]) == 4
    monkeypatch.setattr(
        "iccl.evaluation.metrics._aggregate",
        lambda *_args, **_kwargs: pytest.fail("must consume saved CIs"),
    )
    estimates = script["condition_curves"](path, rows)
    saved = read_rows(path / "curves.csv")
    for condition, (_, low, high) in estimates.items():
        selected = [
            r
            for r in saved
            if r["curve_type"] == "retention_error_delay" and r["condition"] == condition
        ]
        np.testing.assert_array_equal(low, [r["ci_low"] for r in selected])
        np.testing.assert_array_equal(high, [r["ci_high"] for r in selected])
    subset = {name: s for name, s in suites.items() if s["__meta__"]["capability"] == "retention"}
    expected = dict(
        identity,
        suite_files={
            k: v
            for k, v in identity["suite_files"].items()
            if any(k == name + ext for name in subset for ext in (".npz", ".meta.json"))
        },
        selections={name: None for name in subset},
    )
    report = read_evaluation_results(path, expected, subset)
    assert report and {r["capability"] for r in report.curve_rows} == {"retention"}


@pytest.mark.parametrize("damage", ["protocol", "monitor", "missing", "weights", "checksum"])
def test_shared_cache_rejects_incompatible_or_incomplete_results(
    result: tuple, damage: str
) -> None:
    path, identity, _ = result
    expected = dict(identity)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if damage == "protocol":
        manifest["metric_version"] = "old"
    elif damage == "monitor":
        manifest["evaluation_identity"]["sample_scope"] = "monitor"
    elif damage == "missing":
        expected["suite_files"] = identity["suite_files"] | {"missing.npz": "sha"}
    elif damage == "weights":
        expected["model_sha256"] = "changed"
    else:
        (path / "curves.csv").write_text("bad")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        validate_cached_results(path, expected)


def test_snapshot_discovery_includes_staged_snapshots_and_checks_run_and_weights(
    tmp_path: Path,
) -> None:
    script = runpy.run_path(str(SCRIPTS / "plot_retention_trajectory.py"))
    checkpoint = {
        "step": 100,
        "config": {"model": {}, "data": {"input_dim": 4, "output_dim": 4}},
        "model": {"x": torch.ones(2)},
        "wandb_run": {"entity": "e", "project": "p", "run_id": "r", "name": "test"},
    }
    paths = [tmp_path / stage / "snapshots/step_0000100.pt" for stage in ("stage1", "stage2")]
    for path in paths:
        path.parent.mkdir(parents=True)
        torch.save(checkpoint, path)
    snapshots = script["discover_snapshot_series"]([tmp_path], [100], "e/p/r")
    assert len(snapshots) == 1 and snapshots[0].model_sha256 == checkpoint_model_digest(checkpoint)
    checkpoint["model"]["x"] += 1
    torch.save(checkpoint, paths[1])
    with pytest.raises(ValueError, match="Conflicting weights"):
        script["discover_snapshot_series"]([tmp_path], [100], "e/p/r")
    assert (
        len(
            script["discover_snapshot_series"](
                [tmp_path], [100], "e/p/r", explicit_paths=[paths[0]]
            )
        )
        == 1
    )
    with pytest.raises(ValueError, match="belongs to"):
        script["discover_snapshot_series"](
            [tmp_path], [100], "e/p/other", explicit_paths=[paths[0]]
        )
