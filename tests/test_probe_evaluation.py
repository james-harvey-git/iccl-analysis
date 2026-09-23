from pathlib import Path

import numpy as np
import plotly.io as pio
import pytest
import torch
from omegaconf import DictConfig

from iccl.analysis.capture import CaptureEpisodes, collate_capture
from iccl.analysis.plotting import probe_evaluation_figures
from iccl.analysis.probe_evaluation import (
    episode_interval,
    functional_reconstruction,
    score_predictions,
    summarize_scores,
)
from iccl.analysis.probe_loss import aligned_targets
from iccl.analysis.probe_matching import Assignment, AssignmentSolver
from iccl.analysis.probe_results import (
    probe_summary_rows,
    read_results,
    require_comparable,
    write_results,
)
from iccl.analysis.probe_targets import flat_targets


@pytest.mark.parametrize("control", ["none", "constant", "shuffled_targets"])
def test_dashboard_figures_use_saved_episode_statistics_and_baselines(control: str) -> None:
    metadata = {
        "control": control,
        "split": "validation",
        "step": 200,
        "confidence": 0.95,
        "variance_floor": 1e-12,
    }
    arrays = {"episode_index": np.array([10, 11, 12, 13]), "latent_rank": np.array([7, 8, 8, 8])}
    for label, scale in (("decoder", 0.5), ("zero", 1.0)):
        base = np.arange(1, 5, dtype=np.float64) * scale
        for metric in ("joint_mse", "weight_mse", "bias_mse", "readout_mse", "repeated_module_mse"):
            arrays[f"{label}_{metric}"] = (
                base if label == "decoder" or metric != "repeated_module_mse" else np.zeros(4)
            )
        for metric in ("module_mse_by_task", "functional_mse_by_task", "functional_nmse_by_task"):
            arrays[f"{label}_{metric}"] = base[:, None] * np.arange(1, 9)[None, :]
        arrays[f"{label}_functional_variance_floored"] = np.zeros((4, 8), dtype=bool)
        arrays[f"{label}_repeated_module_count"] = np.full(4, 4)
    summary = summarize_scores(arrays, seed=42, replicates=100)
    figures = probe_evaluation_figures(metadata, arrays, summary)
    prefix = "probe/validation/figures/"
    assert set(figures) == {
        prefix + name
        for name in (
            "module_by_task",
            "functional_by_task",
            "parameter_components",
            "episode_error_distribution",
            "latent_rank",
            "repeated_module_consistency",
        )
    }
    for figure in figures.values():
        assert pio.from_json(figure.to_json()).layout.title.text == figure.layout.title.text
    module = figures[prefix + "module_by_task"].to_plotly_json()["data"]
    estimate = summary["decoder"]["all"]["metrics"]["module_mse_by_task"]
    np.testing.assert_allclose(module[1]["y"], estimate["mean"])
    np.testing.assert_allclose(
        np.array(module[0]["y"]) - np.array(module[0]["error_y"]["array"]), estimate["ci_low"]
    )
    np.testing.assert_allclose(
        np.array(module[0]["y"]) + np.array(module[0]["error_y"]["array"]), estimate["ci_high"]
    )
    rows = probe_summary_rows(summary)
    singleton = next(
        row for row in rows if row["population"] == "rank_7" and row["metric"] == "joint_mse"
    )
    assert singleton["n_episodes"] == 1 and singleton["ci_low"] is None
    rank_intervals = figures[prefix + "latent_rank"].to_plotly_json()["data"][0]
    assert tuple(rank_intervals["x"]) == (8,)
    distribution = figures[prefix + "episode_error_distribution"].to_plotly_json()["data"]
    np.testing.assert_array_equal(distribution[0]["x"], [0.5, 1, 1.5, 2])
    np.testing.assert_array_equal(distribution[0]["y"], [0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(
        distribution[1]["x"], arrays["decoder_functional_nmse_by_task"].mean(axis=1)
    )
    consistency = figures[prefix + "repeated_module_consistency"].to_plotly_json()
    np.testing.assert_array_equal(consistency["data"][1]["y"], np.zeros(4))
    assert "Consistency alone" in consistency["layout"]["annotations"][-1]["text"]
    # Preserve a saved interval even when few bootstrap replicates place it above the mean.
    estimate["ci_low"], estimate["ci_high"] = [100.0] * 8, [101.0] * 8
    unusual = probe_evaluation_figures(metadata, arrays, summary)[prefix + "module_by_task"]
    assert tuple(unusual.to_plotly_json()["data"][0]["y"]) == (100.5,) * 8


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
