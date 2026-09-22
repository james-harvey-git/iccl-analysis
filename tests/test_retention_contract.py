from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize

from iccl.data.eval_bundle import prepare_eval_bundle, select_evaluation_suite
from iccl.evaluation.metrics import _evaluate, demo_mse, demo_nmse, load_eval_suites
from iccl.evaluation.validation import validate_retention_group


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("history-contract") / "bundle"
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.eval_sets.out_dir={root}",
                "data.eval_sets.retention_factorial.enabled=false",
                "data.input_dim=4",
                "data.output_dim=4",
                "data.hidden_dims=[4]",
                "data.eval_sets.module_counts.min=8",
                "data.eval_sets.module_counts.max=8",
                "data.eval_sets.task_variation.surplus_tasks={min:1,max:1}",
                "data.eval_sets.num_sequences=2",
                "data.eval_sets.demos_per_task=2",
                "data.eval_sets.retention.num_worlds=9",
                "data.eval_sets.retention.monitor_num_sequences=9",
                "data.eval_sets.rehearsal.enabled=true",
                "data.eval_sets.rehearsal.num_worlds=2",
            ],
        )
    prepare_eval_bundle(cfg)
    return root


def test_standalone_reuses_full_data_and_estimates_with_light_monitor_subset(
    bundle_path: Path,
) -> None:
    all_suites = load_eval_suites(
        bundle_path, select=lambda m: select_evaluation_suite(m, "capabilities")
    )
    standalone = load_eval_suites(
        bundle_path, select=lambda m: select_evaluation_suite(m, "retention_position")
    )
    monitor = load_eval_suites(
        bundle_path, select=lambda m: select_evaluation_suite(m, "retention_position"), monitor=True
    )
    for name, suite in standalone.items():
        assert len(suite["tokens"]) == 9 * 8
        assert suite["__meta__"]["archive_sha256"] == all_suites[name]["__meta__"]["archive_sha256"]
        rows = monitor[name]["__meta__"]["selected_indices"]
        np.testing.assert_array_equal(suite["tokens"][rows], monitor[name]["tokens"])
        assert len(rows) == 9
        assert len(np.unique(monitor[name]["position_group_id"])) == 9
        assert monitor[name]["__meta__"]["sample_scope"] == "monitor"
    reports = []
    for suites in (all_suites, standalone):
        mses, nmses = {}, {}
        for name, suite in suites.items():
            predictions = np.zeros_like(suite["targets"])
            mses[name], nmses[name] = demo_mse(predictions, suite), demo_nmse(predictions, suite)
        reports.append(
            _evaluate(suites, mses, nmses, {}, bootstrap_seed=9, bootstrap_replicates=30)
        )
    assert [r for r in reports[0].curve_rows if r["capability"] == "retention"] == reports[
        1
    ].curve_rows
    assert [r for r in reports[0].summary_rows if r["capability"] == "retention"] == reports[
        1
    ].summary_rows
    assert not any("retention_position__" in p.name for p in bundle_path.iterdir())


@pytest.mark.parametrize(
    "damage",
    ["weights", "final", "pre", "post", "metadata", "background", "pair", "world", "protocol"],
)
def test_corrupt_retention_archives_fail_before_inference(bundle_path: Path, damage: str) -> None:
    suites = load_eval_suites(
        bundle_path, select=lambda m: select_evaluation_suite(m, "retention_position")
    )
    conditions = {s["__meta__"]["condition"]: deepcopy(s) for s in suites.values()}
    repeat, shared = conditions["repeat"], conditions["shared"]
    if damage == "weights":
        shared["latents"][0, 0] = repeat["latents"][0, -1]
    elif damage == "final":
        shared["tokens"][0, -2, 0] += 1
    elif damage in {"pre", "post"}:
        row, task = (1, 0) if damage == "pre" else (0, 1)
        repeat["latents"][row, task] = repeat["latents"][row, -1]
    elif damage == "metadata":
        shared["prior_target_latent_count"][0] = 1
    elif damage == "background":
        remaining = np.flatnonzero(repeat["latents"][0, -1] == 0)
        repeat["latents"][0, 1:-1] = 0
        repeat["latents"][0, 1:-1, remaining[:2]] = 0.5
    elif damage == "pair":
        shared["pair_id"][0] += 1
    elif damage == "world":
        repeat["position_group_id"][0] = 99
    else:
        repeat["__meta__"]["protocol"] = "old"
    with pytest.raises(ValueError, match="retention bundle"):
        validate_retention_group(conditions)


def test_rehearsal_validation_does_not_weaken_standard_contract(bundle_path: Path) -> None:
    suites = load_eval_suites(bundle_path, select=lambda m: select_evaluation_suite(m, "rehearsal"))
    conditions = {s["__meta__"]["condition"]: s for s in suites.values()}
    control = conditions["unexposed"]
    assert not control["target_module_pre_exposures"].any()
    assert control["target_module_post_exposures"].any()
    slot = int(control["rehearsal_positions"][1, 0])
    start = int(control["task_spans"][1, slot, 0])
    control["tokens"][1, start + 1, 0] += 1
    with pytest.raises(ValueError, match="outside original"):
        validate_retention_group(conditions)
