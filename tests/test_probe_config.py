from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from iccl.analysis.probe_config import stream_seed, task_count, validate_probe_config


def test_both_presets_and_phase_inputs(probe_cfg: DictConfig) -> None:
    for stage in ("capture", "train", "benchmark"):
        validate_probe_config(probe_cfg, stage)
    with pytest.raises(ValueError, match="evaluation.checkpoint"):
        validate_probe_config(probe_cfg, "eval")
    probe_cfg.probe.evaluation.checkpoint = "/tmp/probe.pt"
    probe_cfg.probe.capture.checkpoint = None
    validate_probe_config(probe_cfg, "eval")
    validate_probe_config(probe_cfg, "train")
    with pytest.raises(ValueError, match="capture.checkpoint"):
        validate_probe_config(probe_cfg, "capture")
    root = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(config_dir=str(root), version_base=None):
        for preset in ("module_decoder", "smoke"):
            cfg = compose(
                config_name="probe",
                overrides=[f"probe={preset}", "probe.dataset.path=/tmp/dataset"],
            )
            validate_probe_config(cfg, "train")
            assert "model" not in cfg
            assert cfg.data.sequence.demos_per_task == 32
            assert cfg.data.num_modules == 4 and task_count(cfg.data) == 7


def test_fixed_contract_and_thread_allocation(
    probe_cfg: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe_cfg.data.sequence.demos_per_task = 31
    with pytest.raises(ValueError, match="M=4"):
        validate_probe_config(probe_cfg, "train")
    probe_cfg.data.sequence.demos_per_task = 32
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    with pytest.raises(ValueError, match="SLURM"):
        validate_probe_config(probe_cfg, "train")


def test_stream_namespace_is_stable_and_separate() -> None:
    values = [stream_seed(0, f"episodes/{split}") for split in ("train", "validation", "test")]
    assert len(set(values)) == 3
    assert all(2**63 <= value < 2**64 for value in values)
    assert values[0] == stream_seed(0, "episodes/train")
    assert stream_seed(1, "episodes/train") not in values


@pytest.mark.parametrize("surplus", [0, 1, 4, 5])
def test_fixed_task_count_is_configurable(probe_cfg: DictConfig, surplus: int) -> None:
    probe_cfg.data.sequence.surplus_tasks = surplus
    validate_probe_config(probe_cfg, "capture")
    assert task_count(probe_cfg.data) == 3 + surplus


@pytest.mark.parametrize("surplus", [-1, [0, 4], 1.5, True])
def test_task_count_must_be_fixed_nonnegative_integer(probe_cfg: DictConfig, surplus) -> None:
    probe_cfg.data.sequence.surplus_tasks = surplus
    with pytest.raises(ValueError, match="surplus_tasks"):
        validate_probe_config(probe_cfg, "capture")


def test_old_module_count_is_rejected(probe_cfg: DictConfig) -> None:
    probe_cfg.data.num_modules = 8
    with pytest.raises(ValueError, match="four"):
        validate_probe_config(probe_cfg, "capture")


def test_wandb_resume_requires_online_checkpoint_training(probe_cfg: DictConfig) -> None:
    probe_cfg.wandb.resume = True
    with pytest.raises(ValueError, match="online training with probe.training.resume"):
        validate_probe_config(probe_cfg, "train")
    probe_cfg.probe.training.resume = "/tmp/last.pt"
    with pytest.raises(ValueError, match="online training with probe.training.resume"):
        validate_probe_config(probe_cfg, "train")
    probe_cfg.wandb.mode = "online"
    validate_probe_config(probe_cfg, "train")
    probe_cfg.probe.evaluation.checkpoint = "/tmp/best.pt"
    for stage in ("capture", "eval", "benchmark"):
        with pytest.raises(ValueError, match="online training with probe.training.resume"):
            validate_probe_config(probe_cfg, stage)
    probe_cfg.wandb.resume = "must"
    with pytest.raises(ValueError, match="true or false"):
        validate_probe_config(probe_cfg, "train")
