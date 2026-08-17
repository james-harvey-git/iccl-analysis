from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from iccl.data.eval_cells import evaluation_module_counts, resolve_eval_cells
from iccl.data.export import (
    export_eval_sets,
    load_suite,
    load_suite_metadata,
    suite_paths,
)


def test_suite_paths_resolve_default_and_explicit_suites(tmp_path: Path) -> None:
    suite_path = tmp_path / "in_dist.npz"
    metadata_path = tmp_path / "in_dist.meta.json"
    suite_path.write_bytes(b"arrays")
    metadata_path.write_text('{"suite": "in_dist"}')

    assert suite_paths(tmp_path, "in_dist") == (suite_path, metadata_path)
    assert suite_paths(tmp_path / "unused", "unused", tmp_path / "in_dist") == (
        suite_path,
        metadata_path,
    )
    assert load_suite_metadata(metadata_path) == {"suite": "in_dist"}


def test_suite_paths_report_missing_array_and_metadata_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="frozen suite not found"):
        suite_paths(tmp_path, "in_dist")

    (tmp_path / "in_dist.npz").write_bytes(b"arrays")
    with pytest.raises(FileNotFoundError, match="frozen suite metadata not found"):
        suite_paths(tmp_path, "in_dist")


def test_load_suite_metadata_requires_a_json_object(tmp_path: Path) -> None:
    metadata_path = tmp_path / "in_dist.meta.json"
    metadata_path.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_suite_metadata(metadata_path)


def make_variable_cfg(out_dir: Path, *, all_slices: bool = False) -> DictConfig:
    slices: dict[str, Any] = {
        "enabled": ["fixed_surplus"],
        "fixed_surplus": {"surplus_tasks": 1, "history_demos_per_task": 2},
    }
    if all_slices:
        slices = {
            "enabled": [
                "fixed_surplus",
                "matched_task_count",
                "matched_prediction_tokens",
                "task_count",
                "history_demos",
                "matched_serialized_prefix",
                "demo_allocation",
            ],
            "fixed_surplus": {"surplus_tasks": 1, "history_demos_per_task": 32},
            "matched_task_count": {"task_count": 15, "history_demos_per_task": 32},
            "matched_prediction_tokens": {"prediction_tokens": 480, "surplus_tasks": 1},
            "task_count": {
                "module_count": 8,
                "values": [7, 8, 10, 12, 15],
                "history_demos_per_task": 32,
            },
            "history_demos": {"module_count": 8, "task_count": 8, "values": [8, 16, 32, 64]},
            "matched_serialized_prefix": {
                "module_count": 8,
                "serialized_tokens": 520,
                "task_counts": [8, 10, 12],
            },
            "demo_allocation": {
                "module_count": 8,
                "task_count": 8,
                "prediction_tokens": 256,
                "patterns": ["uniform", "alternating", "front_loaded", "back_loaded"],
            },
        }
    module_spec = (
        {"min": 4, "max": 12, "held_out": [6, 10]}
        if all_slices
        else {"min": 4, "max": 5, "held_out": []}
    )
    return OmegaConf.create(
        {
            "seed": 0,
            "data": {
                "name": "hyperteacher",
                "input_dim": 4,
                "output_dim": 4,
                "hidden_dims": [4],
                "use_bias": True,
                "num_modules": module_spec,
                "scale": 3.0,
                "weighting": "discrete",
                "sequence": {
                    "curriculum_sampler": "constructive",
                    "hotness": 2,
                    "surplus_tasks": 1,
                    "demos_per_task": 2,
                    "signal_boundaries": True,
                    "require_identifiable": True,
                    "require_full_rank": False,
                },
                "eval_sets": {
                    "num_sequences": 2,
                    "out_dir": str(out_dir),
                    "module_counts": {
                        "seen_selection": "endpoints_and_canonical",
                        "canonical_seen": 8 if all_slices else 4,
                        "ood": [13, 16] if all_slices else [6],
                    },
                    "capabilities": {
                        "enabled": ["icl", "composition", "retention"],
                        "composition": {
                            "target_demos": 2,
                            "constituent_task_exposures": 1,
                            "controls": ["matched_prefix", "no_history"],
                        },
                        "retention": {"revisit_demos": 2, "controls": ["novel", "shared"]},
                    },
                    "structural_slices": slices,
                },
            },
        }
    )


def test_resolves_the_default_structural_matrix() -> None:
    cfg = make_variable_cfg(Path("unused"), all_slices=True)
    modules, statuses = evaluation_module_counts(cfg.data)
    assert modules == (4, 6, 8, 10, 12, 13, 16)
    assert statuses[6] == "heldout" and statuses[13] == "ood"

    cells = resolve_eval_cells(cfg.data, seed=0)
    assert len(cells) == 37
    matched_tasks = [cell for cell in cells if cell.slice == "matched_task_count"]
    assert {cell.num_tasks for cell in matched_tasks} == {15}
    assert {cell.prediction_tokens for cell in matched_tasks} == {480}
    assert {cell.serialized_tokens for cell in matched_tasks} == {975}
    matched_prediction = [cell for cell in cells if cell.slice == "matched_prediction_tokens"]
    assert {cell.prediction_tokens for cell in matched_prediction} == {480}
    matched_length = [cell for cell in cells if cell.slice == "matched_serialized_prefix"]
    assert {cell.serialized_tokens for cell in matched_length} == {520}


def test_variable_export_writes_capability_condition_cells(tmp_path: Path) -> None:
    cfg = make_variable_cfg(tmp_path)
    assert export_eval_sets(cfg) == 21
    paths = sorted(tmp_path.glob("*.npz"))
    assert len(paths) == 21
    assert not any("category_probe" in path.stem for path in paths)

    constituent_path = next(path for path in paths if "composition__constituent" in path.stem)
    matched_path = Path(str(constituent_path).replace("__constituent__", "__matched_prefix__"))
    no_history_path = Path(str(constituent_path).replace("__constituent__", "__no_history__"))
    constituent = load_suite(constituent_path.with_suffix(""))
    matched = load_suite(matched_path.with_suffix(""))
    no_history = load_suite(no_history_path.with_suffix(""))
    np.testing.assert_array_equal(constituent["pair_id"], matched["pair_id"])
    np.testing.assert_array_equal(constituent["pair_id"], no_history["pair_id"])
    np.testing.assert_array_equal(constituent["constituent_task_exposures"], [[1, 1]] * 2)
    np.testing.assert_array_equal(matched["constituent_task_exposures"], [[0, 0]] * 2)

    metadata = load_suite_metadata(constituent_path.with_suffix(".meta.json"))
    assert metadata["capability"] == "composition"
    assert metadata["condition"] == "constituent"
    assert metadata["structural_slice"] == "fixed_surplus"
    assert metadata["enum_mappings"]["task_origin"]["backbone"] == 1


def test_retention_export_resamples_histories_that_exhaust_all_supports(
    tmp_path: Path,
) -> None:
    cfg = make_variable_cfg(tmp_path)
    cfg.data.num_modules = 4
    cfg.data.eval_sets.module_counts.ood = []
    cfg.data.eval_sets.capabilities.enabled = ["retention"]
    cfg.data.eval_sets.structural_slices.enabled = ["matched_task_count"]
    cfg.data.eval_sets.structural_slices.matched_task_count = {
        "task_count": 15,
        "history_demos_per_task": 2,
    }
    assert export_eval_sets(cfg) == 3
