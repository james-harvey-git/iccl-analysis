import json
import runpy
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from test_retention_position import block, make_cfg, make_family

from iccl.data.eval_bundle import prepare_eval_bundle, select_evaluation_suite
from iccl.data.retention_factorial import build_factorial_cell, factorial_axis
from iccl.evaluation.metrics import _aggregate, _evaluate, load_eval_suites, predict_suite
from iccl.evaluation.results import (
    evaluation_identity,
    read_evaluation_results,
    write_evaluation_results,
)
from iccl.evaluation.validation import validate_factorial_grid, validate_retention_group
from iccl.models.model import model_from_config
from iccl.reporting.factorial import FactorialGrid, factorial_figures, factorial_paper_figures
from iccl.reporting.figures import evaluation_figures, write_html_figures
from iccl.training.trainer import is_monitor_suite


@pytest.mark.parametrize("bad", [[], [1, 1], [-1], [0.5], [True], "0,1"])
def test_invalid_axis(bad: object) -> None:
    with pytest.raises(ValueError, match="factorial axes"):
        factorial_axis(bad)


def test_indexed_banks_and_fresh_examples() -> None:
    family, cfg = make_family(), make_cfg()
    small = build_factorial_cell(family, cfg, seed=12, world=2, preceding=1, delay=1)
    large = build_factorial_cell(family, cfg, seed=12, world=2, preceding=7, delay=7)
    assert factorial_axis([7, 0, 1]) == (0, 1, 7)
    for condition, a in small.items():
        b = large[condition]
        for i, logical in enumerate(a.info["logical_task_id"]):
            j = int(np.flatnonzero(b.info["logical_task_id"] == logical)[0])
            np.testing.assert_array_equal(block(a, i), block(b, j))
        assert not a.info["target_module_pre_exposures"].any()
        assert not a.info["target_module_post_exposures"].any()
        assert not np.array_equal(block(a, 1)[0][::2], block(a, -1)[0][::2])
    empty = build_factorial_cell(family, cfg, seed=12, world=2, preceding=0, delay=0)
    assert not empty["repeat"].info["background_covered"]
    assert len(empty["repeat"].info["latents"]) == 2
    binary = build_factorial_cell(
        make_family("binary", 4),
        cfg,
        seed=0,
        world=0,
        preceding=0,
        delay=0,
        control_modes=("unexposed",),
    )
    assert set(binary) == {"repeat", "unexposed"}
    for identifiable, full_rank in ((False, False), (False, True), (True, True)):
        overridden = build_factorial_cell(
            family,
            replace(cfg, require_identifiable=identifiable, require_full_rank=full_rank),
            seed=12,
            world=2,
            preceding=0,
            delay=0,
        )
        for condition, sample in overridden.items():
            np.testing.assert_array_equal(sample.tokens, empty[condition].tokens)
            np.testing.assert_array_equal(sample.targets, empty[condition].targets)
            np.testing.assert_array_equal(sample.info["latents"], empty[condition].info["latents"])


@pytest.fixture(scope="module")
def factorial_bundle(tmp_path_factory: pytest.TempPathFactory):
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.eval_sets.out_dir={tmp_path_factory.mktemp('factorial') / 'bundle'}",
                "data.input_dim=4",
                "data.output_dim=4",
                "data.hidden_dims=[4]",
                "data.eval_sets.module_counts={min:4,max:4}",
                "data.eval_sets.canonical.module_count=4",
                "data.eval_sets.canonical.task_count=4",
                "data.eval_sets.capabilities=[retention]",
                "data.eval_sets.task_variation.surplus_tasks={min:1,max:1}",
                "data.eval_sets.num_sequences=2",
                "data.eval_sets.demos_per_task=3",
                "data.eval_sets.retention.num_worlds=4",
                "data.eval_sets.retention.monitor_num_sequences=4",
                "data.eval_sets.retention_factorial.num_worlds=4",
                "data.eval_sets.retention_factorial.preceding_tasks=[0,1,3]",
                "data.eval_sets.retention_factorial.intervening_tasks=[0,1,3]",
                "model.d_model=16",
                "model.n_heads=2",
                "model.n_layers=1",
                "model.d_ffw=32",
            ],
        )
    prepare_eval_bundle(cfg)
    return cfg


def factorial_suites(cfg):
    return load_eval_suites(
        Path(cfg.data.eval_sets.out_dir), select=lambda m: m["capability"] == "retention_factorial"
    )


def synthetic_report(suites):
    errors = {}
    for name, suite in suites.items():
        n, tasks = suite["demo_counts"].shape
        condition = suite["__meta__"]["condition"]
        fraction = {"repeat": 0.0, "shared": 0.25, "unexposed": 1.0}[condition]
        errors[name] = np.zeros((n, tasks, 3))
        errors[name][:, -1] = fraction * np.arange(1, n + 1)[:, None]
    return _evaluate(suites, errors, errors, {}, bootstrap_seed=3, bootstrap_replicates=100)


def test_full_standalone_pairing_metrics_and_monitor(factorial_bundle) -> None:
    root = Path(factorial_bundle.data.eval_sets.out_dir)
    full = load_eval_suites(root)
    alone = load_eval_suites(
        root, select=lambda m: select_evaluation_suite(m, "retention_position")
    )
    monitor = load_eval_suites(root, select=is_monitor_suite, monitor=True)
    assert not any(s["__meta__"]["capability"] == "retention_factorial" for s in monitor.values())
    scoped = {k: v for k, v in full.items() if k in alone}
    assert synthetic_report(scoped).summary_rows == synthetic_report(alone).summary_rows
    report = synthetic_report(factorial_suites(factorial_bundle))
    means = {
        r["retention_component"]: r["value"]
        for r in report.summary_rows
        if r["retention_component"]
    }
    assert means == {"total": 2.5, "module": 1.875, "episodic": 0.625}
    expected = _aggregate(np.arange(1, 5), seed=3, replicates=100)
    for row in report.summary_rows:
        assert row["n_sequences"] == 4
        if row["retention_component"] == "total":
            np.testing.assert_allclose([row[k] for k in ("value", "ci_low", "ci_high")], expected)
    assert len(report.summary_rows) == 9 * 6


@pytest.mark.parametrize(
    "damage", ["shared", "background", "inputs", "missing", "duplicate", "bank"]
)
def test_corruption_is_rejected(factorial_bundle, damage: str) -> None:
    suites = deepcopy(factorial_suites(factorial_bundle))
    target = next(
        s
        for s in suites.values()
        if s["__meta__"]["condition"] == "shared"
        and s["__meta__"]["original_task_position"] == 1
        and s["__meta__"]["delay"] == 1
    )
    if damage == "shared":
        target["latents"][0, 1] = target["latents"][0, -1]
    elif damage == "background":
        target["latents"][0, 0] = target["latents"][0, -1]
    elif damage == "inputs":
        target["tokens"][0, target["task_spans"][0, 1, 0], 0] += 1
    elif damage == "bank":
        target["tokens"][0, target["task_spans"][0, 0, 0], 0] += 1
    if damage in {"shared", "background", "inputs"}:
        conditions = {
            s["__meta__"]["condition"]: s
            for s in suites.values()
            if s["__meta__"]["pair_group"] == target["__meta__"]["pair_group"]
        }
        with pytest.raises(ValueError, match="invalid retention bundle"):
            validate_retention_group(conditions)
    else:
        items = list(suites.values())
        if damage == "missing":
            items.pop()
        if damage == "duplicate":
            items.append(items[0])
        with pytest.raises(ValueError, match="factorial"):
            validate_factorial_grid(items)


def test_original_prefix_predictions_are_causal(factorial_bundle) -> None:
    torch.manual_seed(1)
    model = model_from_config(factorial_bundle).eval()
    suites = factorial_suites(factorial_bundle)
    outputs = {}
    for suite in suites.values():
        meta = suite["__meta__"]
        if meta["condition"] != "repeat" or meta["original_task_position"] != 1:
            continue
        preds = predict_suite(model, suite, torch.device("cpu"), batch_size=4)
        start, end = suite["task_spans"][0, 1]
        outputs[meta["delay"]] = preds[:, start:end]
    for output in outputs.values():
        np.testing.assert_allclose(output, outputs[0], atol=1e-6, rtol=1e-5)


def test_four_plot_layouts_preserve_values(factorial_bundle, tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = synthetic_report(factorial_suites(factorial_bundle))
    figures = evaluation_figures(report.summary_rows, report.curve_rows)
    assert len(figures) == 4
    assert write_html_figures(figures.items(), tmp_path) == 4
    rows = []
    for step in (100_000, 2_100_000):
        for row in report.summary_rows:
            modified = dict(row, step=step)
            modified.update(value=-row["value"], ci_low=-row["ci_high"], ci_high=-row["ci_low"])
            rows.append(modified)
    grid = FactorialGrid.from_rows(rows)
    interactive = factorial_figures(grid)
    heatmap = interactive["total_savings_overview"]
    np.testing.assert_array_equal(cast(Any, heatmap.data[0]).z, np.full((3, 3), -2.5))
    assert heatmap.layout.coloraxis.cmin == -2.5
    assert list(cast(Any, interactive["key_slices"].data[2]).x) == [0, 1, 3]
    for name, figure in factorial_paper_figures(grid).items():
        for extension in ("png", "pdf"):
            path = tmp_path / f"{name}.{extension}"
            figure.savefig(path, dpi=60)
            assert path.stat().st_size > 1000
        plt.close(figure)
    with pytest.raises(ValueError, match="absent"):
        grid.layouts(preceding=2)
    component_rows = [r for r in rows if r["retention_component"]]
    with pytest.raises(ValueError, match="Duplicate"):
        FactorialGrid.from_rows(rows + [component_rows[0]])
    # A missing cell/component cannot be filled by a second checkpoint.
    with pytest.raises(ValueError, match="Incomplete"):
        FactorialGrid.from_rows(component_rows[1:])
    total = [
        r
        for r in rows
        if r["retention_component"] == "total"
        and r["original_task_position"] == 0
        and r["intervening_tasks"] == 0
    ]
    assert len(FactorialGrid.from_rows(total).components) == 1
    assert len(factorial_figures(FactorialGrid.from_rows(total))) == 4


def test_saved_trajectory_and_subset_cache(factorial_bundle, tmp_path: Path) -> None:
    suites = factorial_suites(factorial_bundle)
    report = synthetic_report(suites)
    bundle = json.loads(
        (Path(factorial_bundle.data.eval_sets.out_dir) / "manifest.json").read_text()
    )
    script = Path(__file__).resolve().parents[1] / "scripts/plotting/plot_retention_factorial.py"
    load = runpy.run_path(str(script))["load_trajectory"]
    for step in (100, 200):
        checkpoint = {
            "step": step,
            "config": {
                "model": {"backend": "reference"},
                "data": {"input_dim": 4, "output_dim": 4},
            },
            "model": {"weight": torch.zeros(2)},
            "wandb_run": {"entity": "e", "project": "p", "run_id": "r", "name": "test"},
        }
        identity = evaluation_identity(
            checkpoint,
            suites,
            bundle,
            batch_size=4,
            bootstrap_seed=3,
            bootstrap_replicates=100,
            backend="reference",
            device=torch.device("cpu"),
            dtype=None,
        )
        path = write_evaluation_results(
            report,
            tmp_path,
            step,
            {
                "evaluation_identity": identity,
                "suites": {n: s["__meta__"] for n, s in suites.items()},
            },
        )
        restored = read_evaluation_results(path, identity, suites)
        assert restored is not None
        assert set(evaluation_figures(restored.summary_rows, restored.curve_rows)) == set(
            evaluation_figures(report.summary_rows, report.curve_rows)
        )
        cached_figures = evaluation_figures(restored.summary_rows, restored.curve_rows)
        fresh_figures = evaluation_figures(report.summary_rows, report.curve_rows)
        assert all(cached_figures[k].to_json() == fresh_figures[k].to_json() for k in fresh_figures)
    grid, provenance = load([tmp_path])
    assert grid.steps == [100, 200] and len(provenance) == 2
    assert load([tmp_path], [200])[0].steps == [200]
    with pytest.raises(ValueError, match="Missing"):
        load([tmp_path], [300])
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation_identity"]["source_run"] = "e/p/another"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="same factorial"):
        load([tmp_path])


def test_default_grid_export_and_disabled_configuration(factorial_bundle, tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    from iccl.data.export import export_factorial_sets

    cfg = deepcopy(factorial_bundle)
    defaults = OmegaConf.load("configs/data/hyperteacher.yaml").eval_sets.retention_factorial
    assert defaults.num_worlds == 64
    cfg.data.eval_sets.retention_factorial = defaults
    cfg.data.eval_sets.retention_factorial.num_worlds = 1
    assert export_factorial_sets(cfg, out_dir=tmp_path) == 8 * 8 * 3
    suites = load_eval_suites(tmp_path)
    assert len(suites) == 192
    assert {s["__meta__"]["num_modules"] for s in suites.values()} == {4}
    assert {s["__meta__"]["num_tasks"] for s in suites.values()} == set(range(1, 16))
    cfg.data.eval_sets.retention_factorial.enabled = False
    assert export_factorial_sets(cfg, out_dir=tmp_path / "disabled") == 0
    assert not (tmp_path / "disabled").exists()
