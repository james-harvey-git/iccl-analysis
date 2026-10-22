"""Small explicit checkpoint fixtures for fixed-world task-vector integration checks."""

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
    cfg.experiment.root = str(tmp_path / "study")
    cfg.probe.dataset.num_worlds = 1
    cfg.probe.dataset.demos_per_task = 2
    cfg.probe.dataset.target_positions = [1, 5, 9]
    cfg.probe.dataset.capture.batch_size = 8
    cfg.probe.dataset.capture.precision = "fp32"
    cfg.probe.training.l2_grid = [0.1, 1.0]
    cfg.probe.evaluation.bootstrap_replicates = 20
    cfg.probe.evaluation.exposure.min_worlds_per_cell = 1
    architecture = OmegaConf.load(configs / "model/gdn.yaml")
    architecture.d_model = 8
    architecture.n_heads = 2
    architecture.n_layers = 2
    architecture.d_ffw = 16
    architecture.backend = "reference"
    data = OmegaConf.load(configs / "data/hyperteacher.yaml")
    data.input_dim = 4
    data.output_dim = 3
    data.hidden_dims = [4]
    source = OmegaConf.create({"seed": 11, "data": data, "model": architecture})
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
    cfg.probe.dataset.checkpoint = str(checkpoint)
    return cfg
