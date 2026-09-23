from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from iccl.analysis.capture import CaptureEpisodes, collate_capture
from iccl.analysis.probe_evaluation import (
    episode_interval,
    functional_reconstruction,
    score_predictions,
)
from iccl.analysis.probe_loss import aligned_targets
from iccl.analysis.probe_matching import Assignment, AssignmentSolver
from iccl.analysis.probe_results import read_results, require_comparable, write_results
from iccl.analysis.probe_targets import flat_targets


def test_exact_parameters_with_swaps_and_common_hidden_permutation_reconstruct_functions(
    probe_cfg: DictConfig,
) -> None:
    episodes = CaptureEpisodes(probe_cfg.data, 921, 0, 2)
    batch = collate_capture([episodes[0], episodes[1]])
    batch["target"] = flat_targets(batch["modules"], batch["readout"])
    rng = np.random.default_rng(32)
    chosen = Assignment(
        np.zeros(2),
        np.array([173, 91], np.int32),
        np.stack([rng.permutation(16), rng.permutation(16)]).astype(np.int32),
        np.zeros((2, 2), np.int64),
        np.zeros((2, 2)),
    )
    predictions = aligned_targets(torch.from_numpy(batch["target"]), chosen).numpy()
    with AssignmentSolver(2, probe_cfg.probe.solver.cache_dir) as solver:
        scores = score_predictions(
            predictions, batch, solver, readout_weight=1, inputs_per_task=24, functional_seed=292
        )
        assert np.max(scores["joint_mse"]) == 0
        assert np.max(scores["repeated_module_mse"]) == 0
        assert np.max(scores["functional_nmse_by_task"]) < 2e-12
        np.testing.assert_array_equal(scores["hidden_permutation"], chosen.permutations)
        np.testing.assert_array_equal(scores["swap_mask"], chosen.masks)
        predictions[:, -256:] = 0
        broken = functional_reconstruction(predictions, batch, chosen, inputs_per_task=24, seed=292)
        assert np.min(broken["functional_nmse_by_task"]) > 0.5
        zeros = score_predictions(
            np.zeros_like(predictions),
            batch,
            solver,
            readout_weight=1,
            inputs_per_task=24,
            functional_seed=292,
        )
        assert np.all(zeros["repeated_module_mse"] == 0)
        assert np.all(zeros["joint_mse"] > 0)


def test_functional_zero_variance_and_whole_episode_intervals(probe_cfg: DictConfig) -> None:
    batch = collate_capture([CaptureEpisodes(probe_cfg.data, 88, 0, 1)[0]])
    assignment = Assignment(
        np.zeros(1),
        np.zeros(1, np.int32),
        np.arange(16)[None].astype(np.int32),
        np.zeros((1, 2), np.int64),
        np.zeros((1, 2)),
    )
    batch["world_readout"].fill(0)
    result = functional_reconstruction(
        np.zeros((1, 4608), np.float32), batch, assignment, inputs_per_task=8, seed=3
    )
    assert result["functional_variance_floored"].all()
    assert (result["functional_nmse_by_task"] == 0).all()
    values = np.array([[1, 2], [2, 4], [4, 8]], np.float64)
    interval = episode_interval(values, seed=112, replicates=100)
    assert interval["n_episodes"] == 3
    for key in ("mean", "ci_low", "ci_high"):
        assert interval[key][1] == 2 * interval[key][0]
    assert episode_interval(values, seed=112, replicates=0)["ci_low"] is None


def test_portable_results_reject_corruption_and_incompatible_comparisons(tmp_path: Path) -> None:
    metadata = dict(
        dataset_id="a",
        split_signature="b",
        readout_weight=1,
        target_layout={},
        evaluation_precision="float32",
        functional_inputs_per_task=8,
        functional_seed=0,
        functional_stream_seed=22,
        variance_floor=1e-12,
    )
    path = write_results(
        tmp_path / "results", metadata, {"joint": np.array([1.0, 2.0])}, {"mean": 1.5}
    )
    loaded, arrays, summary = read_results(path)
    assert loaded == metadata and summary["mean"] == 1.5
    np.testing.assert_array_equal(arrays["joint"], [1, 2])
    require_comparable([metadata, dict(metadata)])
    with pytest.raises(ValueError, match="identical"):
        require_comparable([metadata, dict(metadata, functional_seed=1)])
    with pytest.raises(FileExistsError):
        write_results(path, metadata, arrays, summary)
    (path / "summary.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        read_results(path)
