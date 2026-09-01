from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from iccl.data.dataset import collate_sequences, sequence_dataset_from_config, to_tensors
from iccl.data.eval_cells import evaluation_module_counts, resolve_eval_cells
from iccl.data.export import (
    VALIDATION_SEED_OFFSET,
    VALIDATION_SUITE,
    balanced_repeat_positions,
    export_eval_sets,
    load_suite,
    load_suite_metadata,
    suite_paths,
)
from iccl.data.sequences import TOKEN_PAD


def make_cfg(out_dir: Path, capabilities: list[str] | None = None) -> DictConfig:
    return OmegaConf.create(
        {
            "seed": 3,
            "data": {
                "name": "hyperteacher",
                "input_dim": 4,
                "output_dim": 4,
                "hidden_dims": [4],
                "use_bias": True,
                "num_modules": {"min": 4, "max": 6, "held_out": [5]},
                "scale": 3.0,
                "weighting": "discrete",
                "sequence": {
                    "curriculum_sampler": "constructive",
                    "hotness": 2,
                    "surplus_tasks": [0, 1],
                    "demos_per_task": {"min": 1, "max": 3, "scope": "per_task"},
                    "signal_boundaries": True,
                    "require_identifiable": True,
                    "require_full_rank": False,
                    "max_attempts": 1000,
                },
                "eval_sets": {
                    "num_sequences": 8,
                    "demos_per_task": 2,
                    "out_dir": str(out_dir),
                    "bootstrap_seed": 0,
                    "bootstrap_replicates": 20,
                    "best_metric": "validation/token_mse",
                    "capabilities": capabilities or ["icl", "composition", "retention"],
                    "canonical": {"module_count": 4, "task_count": 4},
                    "module_counts": {"min": 4, "max": 7},
                    "task_variation": {"surplus_tasks": {"min": 0, "max": 1}},
                    "composition": {
                        "constituent_task_exposures": 1,
                        "controls": ["matched_prefix", "no_history"],
                    },
                    "retention": {
                        "controls": ["novel", "shared"],
                    },
                },
            },
        }
    )


def test_suite_paths_and_metadata_validation(tmp_path: Path) -> None:
    arrays = tmp_path / "in_dist.npz"
    metadata = tmp_path / "in_dist.meta.json"
    arrays.write_bytes(b"arrays")
    metadata.write_text('{"suite": "in_dist"}')
    assert suite_paths(tmp_path, "in_dist") == (arrays, metadata)
    assert load_suite_metadata(metadata) == {"suite": "in_dist"}

    metadata.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_suite_metadata(metadata)
    arrays.unlink()
    with pytest.raises(FileNotFoundError, match="frozen suite not found"):
        suite_paths(tmp_path, "in_dist")


def test_resolves_and_deduplicates_the_three_cell_families(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    modules, statuses = evaluation_module_counts(cfg.data)
    assert modules == (4, 5, 6, 7)
    assert statuses == {4: "seen", 5: "heldout", 6: "seen", 7: "ood"}

    cells = resolve_eval_cells(cfg.data)
    assert len(cells) == 11
    assert all(cell.demos_per_task == 2 for cell in cells)
    assert {
        (cell.num_modules, cell.num_tasks)
        for cell in cells
        if "module_variation" in cell.family_memberships
    } == {
        (4, 7),
        (5, 7),
        (6, 7),
        (7, 7),
    }
    canonical = next(cell for cell in cells if "canonical" in cell.family_memberships)
    assert (canonical.num_modules, canonical.num_tasks, canonical.num_surplus) == (4, 4, 1)
    assert canonical.family_memberships == ("canonical", "task_variation")
    overlap = next(cell for cell in cells if (cell.num_modules, cell.num_tasks) == (7, 7))
    assert overlap.family_memberships == ("task_variation", "module_variation")


def test_explicit_evaluation_module_counts_are_supported(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.data.eval_sets.module_counts = [4, 6, 8]
    modules, statuses = evaluation_module_counts(cfg.data)
    assert modules == (4, 6, 8)
    assert statuses == {4: "seen", 6: "seen", 8: "ood"}
    module_cells = [
        cell
        for cell in resolve_eval_cells(cfg.data)
        if "module_variation" in cell.family_memberships
    ]
    assert {(cell.num_modules, cell.num_tasks) for cell in module_cells} == {
        (4, 8),
        (6, 8),
        (8, 8),
    }


@pytest.mark.parametrize("values", [[], [1, 4], [4, 4]])
def test_explicit_evaluation_module_counts_are_validated(tmp_path: Path, values: list[int]) -> None:
    cfg = make_cfg(tmp_path)
    cfg.data.eval_sets.module_counts = values
    with pytest.raises(ValueError, match="module_counts"):
        evaluation_module_counts(cfg.data)


def test_removed_structural_slice_fields_are_rejected(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    cfg.data.eval_sets.structural_slices = {"enabled": ["fixed_surplus"]}
    with pytest.raises(ValueError, match="unknown eval_sets fields.*structural_slices"):
        resolve_eval_cells(cfg.data)


def test_repeat_positions_are_deterministic_and_balanced() -> None:
    first = balanced_repeat_positions(19, 6, seed=7)
    second = balanced_repeat_positions(19, 6, seed=7)
    np.testing.assert_array_equal(first, second)
    counts = np.bincount(first, minlength=6)
    assert counts.max() - counts.min() == 1
    assert set(first) == set(range(6))


def test_export_uses_fixed_capability_d_and_variable_training_validation(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    (tmp_path / "obsolete.npz").write_bytes(b"stale")
    (tmp_path / "obsolete.meta.json").write_text("{}")
    assert export_eval_sets(cfg) == 78  # validation + 11 cells x seven conditions
    assert not (tmp_path / "obsolete.npz").exists()

    validation = load_suite(tmp_path / VALIDATION_SUITE)
    assert np.all(validation["loss_mask"][validation["token_type"] == TOKEN_PAD] == 0)
    dataset = sequence_dataset_from_config(cfg.data, base_seed=cfg.seed + VALIDATION_SEED_OFFSET)
    validation_samples = [dataset.build(index) for index in range(cfg.data.eval_sets.num_sequences)]
    expected = collate_sequences([to_tensors(sample) for sample in validation_samples])
    for key, value in expected.items():
        np.testing.assert_array_equal(validation[key], value.numpy())
    assert len({tuple(sample.info["demo_counts"]) for sample in validation_samples}) > 1

    constituent_path = next(tmp_path.glob("composition__constituent__*.npz"))
    matched_path = Path(str(constituent_path).replace("__constituent__", "__matched_prefix__"))
    no_history_path = Path(str(constituent_path).replace("__constituent__", "__no_history__"))
    constituent = load_suite(constituent_path.with_suffix(""))
    matched = load_suite(matched_path.with_suffix(""))
    no_history = load_suite(no_history_path.with_suffix(""))
    np.testing.assert_array_equal(constituent["pair_id"], matched["pair_id"])
    np.testing.assert_array_equal(constituent["pair_id"], no_history["pair_id"])
    assert np.all(constituent["demo_counts"] == 2)
    assert np.all(matched["demo_counts"] == 2)
    assert np.all(no_history["demo_counts"] == 2)

    repeat_path = next(tmp_path.glob("retention__repeat__m04__t04__d002.npz"))
    repeat = load_suite(repeat_path.with_suffix(""))
    assert np.all(repeat["demo_counts"] == 2)
    np.testing.assert_array_equal(repeat["intervening_tasks"], 3 - repeat["original_task_position"])
    counts = np.bincount(repeat["original_task_position"], minlength=4)
    assert counts.max() - counts.min() <= 1

    metadata = load_suite_metadata(constituent_path.with_suffix(".meta.json"))
    assert metadata["demos_per_task"] == 2
    assert metadata["family_memberships"]
    assert len(metadata["archive_sha256"]) == 64
    assert metadata["enum_mappings"]["task_origin"]["backbone"] == 1


def test_retention_requires_enough_sequences_to_represent_every_position(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, ["retention"])
    cfg.data.eval_sets.num_sequences = 6
    with pytest.raises(ValueError, match="largest evaluated task count"):
        export_eval_sets(cfg)


def test_binary_evaluation_omits_the_degenerate_shared_control(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, ["retention"])
    cfg.data.weighting = "binary"
    assert export_eval_sets(cfg) == 23  # validation + 11 cells x repeat/novel
    assert not list(tmp_path.glob("retention__shared__*.npz"))
