from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from test_retention_position import block, make_cfg, make_family

from iccl.data.eval_bundle import prepare_eval_bundle, select_evaluation_suite
from iccl.data.retention_factorial import build_factorial_cell, factorial_axis
from iccl.evaluation.metrics import _aggregate, _evaluate, load_eval_suites, predict_suite
from iccl.evaluation.validation import validate_factorial_grid, validate_retention_group
from iccl.models.model import model_from_config
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
    with pytest.raises(ValueError, match="full_rank"):
        build_factorial_cell(
            family, replace(cfg, require_full_rank=True), seed=0, world=0, preceding=0, delay=0
        )


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
