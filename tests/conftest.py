"""Small explicit checkpoint fixtures for the module-decoder integration tests."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from iccl.models.model import model_from_config


@pytest.fixture
def probe_cfg(tmp_path: Path) -> DictConfig:
    configs = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        cfg = compose(config_name="probe", overrides=["probe=smoke"])
    cfg.device = "cpu"
    cfg.probe.dataset.path = str(tmp_path / "dataset")
    cfg.probe.dataset.counts = {"train": 4, "validation": 2, "test": 2}
    cfg.probe.dataset.shard_size = 3
    cfg.probe.capture.batch_size = 2
    cfg.probe.training.batch_size = 3
    cfg.probe.training.num_steps = 4
    cfg.probe.training.warmup_steps = 0
    cfg.probe.training.validation_every = 2
    cfg.probe.training.checkpoint_every = 2
    cfg.probe.training.log_every = 1
    cfg.probe.solver.cache_dir = str(configs.parent / "outputs/.cache/probe-assignment-tests")
    cfg.probe.evaluation.functional_inputs_per_task = 8
    cfg.probe.evaluation.bootstrap_replicates = 20
    cfg.probe.benchmark.warmup_steps = 1
    cfg.probe.benchmark.measured_steps = 2
    cfg.probe.benchmark.profile_steps = 1
    architecture = OmegaConf.load(configs / "model/gdn.yaml")
    architecture.d_model = 8
    architecture.n_heads = 2
    architecture.n_layers = 1
    architecture.d_ffw = 16
    architecture.backend = "reference"
    source = OmegaConf.create({"seed": 11, "data": cfg.data, "model": architecture})
    torch.manual_seed(11)
    model = model_from_config(source)
    checkpoint = tmp_path / "gdn.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": OmegaConf.to_container(source, resolve=True),
            "step": 17,
        },
        checkpoint,
    )
    cfg.probe.capture.checkpoint = str(checkpoint)
    return cfg
