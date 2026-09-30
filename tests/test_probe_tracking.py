"""Captured data and checkpoint lineage take precedence over launch defaults."""

from pathlib import Path

from omegaconf import DictConfig

from iccl.analysis.capture import capture_dataset
from iccl.analysis.probe_config import resolved_config
from iccl.analysis.probe_dataset import read_manifest
from iccl.analysis.probe_tracking import tracking_config
from iccl.checkpoints import SourceRun


def test_reused_dataset_and_evaluation_record_actual_training_settings(
    probe_cfg: DictConfig,
    tmp_path: Path,
) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    manifest = read_manifest(probe_cfg.probe.dataset.path)
    training = resolved_config(probe_cfg)
    training["probe"]["training"]["schedule"] = "cosine"
    training["seed"] = 7
    probe_cfg.probe.dataset.counts.train = 999
    probe_cfg.probe.dataset.shard_size = 888
    cfg = tracking_config(probe_cfg, manifest, "eval", training_config=training)
    assert cfg.probe.dataset.counts.train == 4
    assert cfg.probe.dataset.shard_size == 3
    assert cfg.probe.training.schedule == "cosine"
    assert cfg.seed == 7
    assert cfg.wandb.name == "Cosine | full | seed 7 | eval"
    assert cfg.captured_dataset.dataset_id == manifest["dataset_id"]
    assert probe_cfg.probe.dataset.counts.train == 999
    assert probe_cfg.probe.training.schedule == "constant"
    prior = SourceRun("entity", "project", "id", "existing run name", 50)
    resumed = tracking_config(probe_cfg, manifest, "train", resume_run=prior)
    assert resumed.wandb.name == prior.name
