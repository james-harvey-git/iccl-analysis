from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from iccl.data.export import (
    export_eval_sets,
    load_suite,
    load_suite_metadata,
    suite_paths,
)

SHARED_SUITES = [
    "in_dist",
    "composite",
    "composite_control",
    "structural_chain",
    "retention",
]


def make_cfg(out_dir: Path, **data_overrides: Any) -> DictConfig:
    """The pilot config's shape at a size that exports in well under a second."""
    data = {
        "name": "hyperteacher",
        "input_dim": 4,
        "output_dim": 4,
        "hidden_dims": [4],
        "use_bias": True,
        "num_modules": 8,
        "scale": 3.0,
        "weighting": "discrete",
        "sequence": {
            "phases": [{"num_tasks": 8, "hotness": [2, 2]}],
            "demos_per_task": 3,
            "signal_boundaries": True,
            "require_identifiable": True,
            "require_full_rank": False,
            "task_graph": "random",
            "graph_ordered": False,
        },
        "eval_sets": {
            "num_sequences": 2,
            "out_dir": str(out_dir),
            "composite": {"hotness": 2, "num_demos": 2},
            "structural_graphs": ["chain"],
            "retention": {"revisit_demos": 2, "controls": ["novel", "shared"]},
        },
    }
    return OmegaConf.create({"seed": 0, "data": {**data, **data_overrides}})


def suite_arrays(out_dir: Path, name: str) -> dict[str, np.ndarray]:
    return load_suite(out_dir / name)


def test_exports_every_suite(tmp_path: Path) -> None:
    assert export_eval_sets(make_cfg(tmp_path)) == 7
    for name in [*SHARED_SUITES, "retention_control", "retention_control_shared"]:
        arrays = suite_arrays(tmp_path, name)
        assert arrays["tokens"].shape[0] == 2
        assert (tmp_path / f"{name}.meta.json").exists()


def test_retention_controls_leave_the_other_suites_untouched(tmp_path: Path) -> None:
    """Suites take consecutive index blocks, so requesting controls must not
    shift the sequences of any suite exported before them."""
    with_controls, without = tmp_path / "with", tmp_path / "without"
    export_eval_sets(make_cfg(with_controls))
    cfg = make_cfg(without)
    cfg.data.eval_sets.retention.controls = []
    export_eval_sets(cfg)

    for name in SHARED_SUITES:
        left, right = suite_arrays(with_controls, name), suite_arrays(without, name)
        assert left.keys() == right.keys(), name
        for key in left:
            np.testing.assert_array_equal(left[key], right[key], err_msg=f"{name}/{key}")


def test_binary_weighting_skips_the_shared_control(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert export_eval_sets(make_cfg(tmp_path, weighting="binary")) == 6
    assert (tmp_path / "retention_control.npz").exists()
    assert not (tmp_path / "retention_control_shared.npz").exists()
    assert "weighting=binary" in capsys.readouterr().out


def test_shared_control_meta_records_its_distance_to_the_revisited_task(tmp_path: Path) -> None:
    import json

    export_eval_sets(make_cfg(tmp_path))
    meta = json.loads((tmp_path / "retention_control_shared.meta.json").read_text())
    distances = meta["latent_distance_to_revisited"]
    assert 0.0 < distances["min"] <= distances["mean"]


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
