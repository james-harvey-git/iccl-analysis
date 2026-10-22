from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from iccl.analysis.capture import capture_dataset
from iccl.analysis.probe_config import validate_probe_config
from iccl.analysis.probe_evaluation import evaluate_probe
from iccl.analysis.probe_features import feature_selections
from iccl.analysis.probe_storage import (
    CapturedSplit,
    file_digest,
    read_artifact,
    read_json,
    read_manifest,
)
from iccl.analysis.probe_training import train_probe


def test_three_stages_resume_split_isolation_and_saved_reports(probe_cfg, tmp_path, monkeypatch):
    cfg = probe_cfg
    cfg.probe.dataset.capture.representations = {
        "state": ["all_layers"],
        "mlp_hidden": ["all_layers"],
    }
    capture_dataset(cfg, tmp_path / "logs")
    manifest = read_manifest(cfg.probe.dataset.path)
    target = (
        Path(cfg.probe.dataset.path) / "world-0000/position-001/train/shard-00000000/target.npy"
    )
    before = (file_digest(target), target.stat().st_mtime_ns)
    cfg.probe.dataset.capture.batch_size = 3
    capture_dataset(cfg, tmp_path / "logs")
    assert (file_digest(target), target.stat().st_mtime_ns) == before

    class TrainingOnly(CapturedSplit):
        def __init__(self, root, world, position, split):
            assert split in {"train", "validation"}, "training read a held-out population"
            super().__init__(root, world, position, split)

    monkeypatch.setattr("iccl.analysis.probe_training.CapturedSplit", TrainingOnly)
    train_probe(cfg, tmp_path / "logs")
    probes = read_artifact(Path(cfg.probe.training.output), "probes")
    assert probes["identity"]["dataset_id"] == manifest["dataset_id"]
    selected = read_json(Path(cfg.probe.training.output) / "world-0000/complete.json")["selected"]
    assert len(selected) == 3 * 3 * (2 * 3 + 2)
    for item in selected:
        if item["candidates"]:
            expected = max(
                item["candidates"], key=lambda x: (x["validation"]["r2_macro"], x["regularizer"])
            )
            assert item["regularizer"] == expected["regularizer"]
    monkeypatch.setattr(
        "iccl.analysis.probe_training.fit_ridge",
        lambda *a, **k: pytest.fail("resume refitted a completed candidate"),
    )
    train_probe(cfg, tmp_path / "logs")
    path = evaluate_probe(cfg, tmp_path / "logs")
    summary = read_json(path)
    assert summary["per_world"] and summary["exposure"]
    assert {r["population"] for r in summary["per_world"]} == {"test_familiar", "test_unseen"}
    assert all(r["upper"] is None for r in summary["aggregate"])
    root = Path(cfg.probe.evaluation.output)
    assert (root / "report.md").exists()
    assert (root / "plots/index.html").exists()
    assert (root / "plots/test_familiar--exposure-state-r2_macro.png").stat().st_size > 10000
    # All three captures use the same target labels and trial ordering.
    z = [
        np.load(
            root
            / "world-0000/position-005"
            / point
            / "state/all/trained/n-00000032/test_unseen.npz"
        )["targets"]
        for point in ("final_input", "final_answer", "branch_input")
    ]
    np.testing.assert_array_equal(z[0], z[1])
    np.testing.assert_array_equal(z[0], z[2])


def test_corruption_and_protocol_mismatch_are_rejected(probe_cfg, tmp_path):
    probe_cfg.probe.dataset.target_positions = [1]
    probe_cfg.probe.dataset.capture.representations = {"state": ["all_layers"]}
    capture_dataset(probe_cfg, tmp_path / "logs")
    path = (
        Path(probe_cfg.probe.dataset.path)
        / "world-0000/position-001/train/shard-00000000/target.npy"
    )
    targets = np.load(path)
    targets[0, 0] += 0.1
    np.save(path, targets)
    with pytest.raises(ValueError, match="checksum"):
        CapturedSplit(probe_cfg.probe.dataset.path, 0, 1, "train")
    with pytest.raises(ValueError, match="checksum"):
        capture_dataset(probe_cfg, tmp_path / "logs")
    probe_cfg.probe.dataset.demos_per_task += 1
    with pytest.raises(ValueError, match="differs"):
        capture_dataset(probe_cfg, tmp_path / "logs")


def test_feature_groups_and_configuration_contract(probe_cfg):
    shapes = {"state": [2, 2, 3, 4], "head_read": [2, 2, 3]}
    selections = feature_selections(
        shapes, {"state": ["all_layers", "each_head"]}, {"paired": [[1, 2], [2, 1]]}
    )
    values = np.arange(2 * 2 * 3 * 4).reshape(1, 2, 2, 3, 4)
    np.testing.assert_array_equal(
        selections[-1].select(values),
        np.concatenate([values[:, 0, 1].reshape(1, -1), values[:, 1, 0].reshape(1, -1)], axis=1),
    )
    with pytest.raises(ValueError, match="outside"):
        feature_selections(shapes, {"state": ["all_layers"]}, {"bad": [[3, 1]]})
    probe_cfg.experiment.phase = "confirmation"
    with pytest.raises(ValueError, match="freeze"):
        validate_probe_config(probe_cfg, "train")
    probe_cfg.experiment.phase = "development"
    probe_cfg.execution.max_gpus = 9
    with pytest.raises(ValueError, match="eight"):
        validate_probe_config(probe_cfg, "capture")


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA correctness check")
@torch.inference_mode()
def test_cuda_terminal_capture_matches_reference(probe_cfg):
    from iccl.analysis.capture import capture_trials
    from iccl.analysis.probe_worlds import make_trial, make_world
    from iccl.checkpoints import checkpoint_model_config
    from iccl.data.teacher import TeacherConfig
    from iccl.models.model import model_from_config

    checkpoint = torch.load(probe_cfg.probe.dataset.checkpoint, weights_only=False)
    architecture = OmegaConf.create(checkpoint_model_config(checkpoint))
    architecture.model.backend = "reference"
    model = model_from_config(architecture).eval()
    model.load_state_dict(checkpoint["model"])
    world = make_world(
        TeacherConfig(
            input_dim=4,
            output_dim=3,
            hidden_dims=(4,),
            num_modules=8,
            use_bias=True,
            scale=3**0.5,
            weighting="discrete",
        ),
        seed=0,
        phase="development",
        world_id=0,
        pair_counts={"train": 16, "validation": 6, "test_unseen": 6},
    )
    trials = [
        make_trial(world, split="train", position=5, index=i, tasks=9, demos=3) for i in range(3)
    ]
    families = ["state", "head_read", "mlp_hidden", "residual_after_mixer", "residual_after_mlp"]
    reference = capture_trials(
        model,
        trials,
        backend="reference",
        device=torch.device("cpu"),
        dtype=None,
        families=families,
    )
    accelerated = capture_trials(
        model.cuda(),
        trials,
        backend="fla",
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        families=families,
    )
    for key in reference:
        scale = max(float(np.sqrt(np.mean(reference[key] ** 2))), 0.01)
        error = float(np.sqrt(np.mean((reference[key] - accelerated[key]) ** 2)))
        assert error / scale < 0.05, (key, error / scale)
