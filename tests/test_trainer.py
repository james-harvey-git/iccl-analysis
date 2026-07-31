import math
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from iccl.data.dataset import make_family, sequence_config_from, sequence_rng
from iccl.data.export import export_suite
from iccl.data.sequences import build_sequence
from iccl.models.model import model_from_config
from iccl.training.trainer import (
    Trainer,
    build_optimizer,
    build_scheduler,
    masked_mse,
    split_decay_params,
)


def make_cfg(tmp_path: Path, **training_overrides: Any) -> DictConfig:
    training = {
        "batch_size": 2,
        "num_steps": 3,
        "optimizer": "adamw",
        "lr": 1.0e-3,
        "warmup_steps": 2,
        "schedule": "constant",
        "weight_decay": 0.003,
        "grad_clip": 1.0,
        "precision": "fp32",
        "num_workers": 0,
        "log_every": 100,
        "eval_every": None,
        "checkpoint_every": 100,
        "resume": None,
    }
    training.update(training_overrides)
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "data": {
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
                    "demos_per_task": 4,
                    "signal_boundaries": True,
                    "require_identifiable": True,
                },
                "eval_sets": {"out_dir": str(tmp_path / "eval_sets")},
            },
            "model": {
                "d_model": 32,
                "n_heads": 2,
                "n_layers": 2,
                "d_ffw": 64,
                "expand_v": 2,
                "use_short_conv": True,
                "conv_size": 4,
                "use_gate": True,
                "allow_neg_eigval": False,
                "norm_eps": 1.0e-5,
                "backend": "reference",
            },
            "training": training,
            "wandb": {"mode": "disabled", "project": "test", "entity": None},
        }
    )


def test_split_decay_params_by_name(tmp_path: Path) -> None:
    model = model_from_config(make_cfg(tmp_path))
    decay, no_decay = split_decay_params(model)
    all_names = {name for name, _ in model.named_parameters()}
    assert set(decay) | set(no_decay) == all_names
    assert not set(decay) & set(no_decay)
    for param in decay.values():
        assert param.ndim >= 2 and not getattr(param, "_no_weight_decay", False)
    for param in no_decay.values():
        assert param.ndim < 2 or getattr(param, "_no_weight_decay", False)
    assert "embed.bias" in no_decay
    assert "blocks.0.mixer.A_log" in no_decay
    assert "blocks.0.mixer.dt_bias" in no_decay
    assert "blocks.0.mixer.q_proj.weight" in decay
    assert "head.weight" in decay

    optimizer = build_optimizer(model, make_cfg(tmp_path).training)
    weight_decays = [group["weight_decay"] for group in optimizer.param_groups]
    assert weight_decays == [0.003, 0.0]


def test_scheduler_warmup_then_constant_or_cosine(tmp_path: Path) -> None:
    def lr_at_each_step(schedule: str, num_steps: int) -> list[float]:
        cfg = make_cfg(tmp_path, schedule=schedule, num_steps=num_steps, warmup_steps=4)
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=1.0)
        scheduler = build_scheduler(optimizer, cfg.training)
        lrs = []
        for _ in range(num_steps):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()
        return lrs

    constant = lr_at_each_step("constant", 8)
    assert constant[:4] == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert constant[4:] == pytest.approx([1.0] * 4)

    cosine = lr_at_each_step("cosine", 8)
    assert cosine[:4] == pytest.approx(constant[:4])
    assert cosine[4:] == sorted(cosine[4:], reverse=True)
    assert cosine[-1] == pytest.approx(0.5 * (1 + math.cos(math.pi * 3 / 4)))


def test_masked_mse_ignores_unmasked_positions() -> None:
    preds = torch.zeros(1, 4, 2)
    targets = torch.ones(1, 4, 2)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    assert masked_mse(preds, targets, mask).item() == pytest.approx(1.0)
    preds[0, 1] = 100.0  # unmasked position must not contribute
    assert masked_mse(preds, targets, mask).item() == pytest.approx(1.0)


def test_trainer_smoke_and_checkpoint(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, num_steps=3, checkpoint_every=2)
    torch.manual_seed(0)
    trainer = Trainer(cfg, out_dir=tmp_path / "run")
    trainer.fit()
    assert math.isfinite(trainer.last_loss)
    assert trainer.step == 3
    assert (tmp_path / "run" / "checkpoints" / "last.pt").exists()


def test_checkpoint_resume_is_bit_exact(tmp_path: Path) -> None:
    torch.manual_seed(0)
    full = Trainer(make_cfg(tmp_path, num_steps=6, checkpoint_every=3), tmp_path / "full")
    full.fit()

    torch.manual_seed(0)
    half = Trainer(make_cfg(tmp_path, num_steps=3, checkpoint_every=3), tmp_path / "half")
    half.fit()

    torch.manual_seed(123)  # resume must not depend on the ambient RNG
    resume_path = str(tmp_path / "half" / "checkpoints" / "last.pt")
    resumed = Trainer(
        make_cfg(tmp_path, num_steps=6, checkpoint_every=3, resume=resume_path),
        tmp_path / "resumed",
    )
    assert resumed.step == 3
    resumed.fit()

    full_state = full.model.state_dict()
    resumed_state = resumed.model.state_dict()
    assert full_state.keys() == resumed_state.keys()
    for key in full_state:
        assert torch.equal(full_state[key], resumed_state[key]), key


def test_eval_requires_frozen_suites(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, eval_every=2)
    trainer = Trainer(cfg, out_dir=tmp_path / "run")
    with pytest.raises(FileNotFoundError, match="make_eval_sets"):
        trainer.fit()


def test_trainer_runs_eval_and_tracks_best(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, num_steps=2, eval_every=2)
    family = make_family(cfg.data)
    seq_cfg = sequence_config_from(cfg.data)
    samples = [build_sequence(family, seq_cfg, sequence_rng(1, i)) for i in range(2)]
    out_dir = Path(cfg.data.eval_sets.out_dir)
    export_suite(samples, out_dir / "in_dist", {"suite": "in_dist"})

    torch.manual_seed(0)
    trainer = Trainer(cfg, out_dir=tmp_path / "run")
    trainer.fit()
    assert math.isfinite(trainer.best_metric)
    assert (tmp_path / "run" / "checkpoints" / "best.pt").exists()
    assert (tmp_path / "run" / "eval" / "step_0000002.npz").exists()
