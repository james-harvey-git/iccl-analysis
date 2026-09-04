import math
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from iccl.data.eval_bundle import prepare_eval_bundle
from iccl.models.model import model_from_config
from iccl.training.trainer import (
    Trainer,
    build_optimizer,
    build_scheduler,
    is_monitor_suite,
    masked_mse,
    resolve_snapshot_steps,
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
        "validation_every": None,
        "monitor_every": None,
        "checkpoint_every": 100,
        "resume": None,
        "snapshots": {
            "every": None,
            "at": None,
            "log": None,
            "include_optimizer_state": False,
        },
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
                "eval_sets": {
                    "out_dir": str(tmp_path / "eval_sets"),
                    "best_metric": "validation/token_mse",
                },
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


def test_canonical_monitor_selector_uses_family_membership(tmp_path: Path) -> None:
    metadata = {
        "capability": "icl",
        "family_memberships": ["canonical", "task_variation"],
    }
    assert is_monitor_suite(metadata)
    assert not is_monitor_suite(metadata | {"family_memberships": ["task_variation"]})
    assert not is_monitor_suite(metadata | {"capability": "validation"})


def snapshot_steps(num_steps: int, **snapshots: Any) -> list[int]:
    defaults = {"every": None, "at": None, "log": None}
    return resolve_snapshot_steps(
        OmegaConf.create({"num_steps": num_steps, "snapshots": defaults | snapshots})
    )


def test_snapshot_schedule_generators_compose() -> None:
    assert snapshot_steps(100, every=25) == [25, 50, 75, 100]
    assert snapshot_steps(100, at=[3, 7]) == [3, 7, 100]
    assert snapshot_steps(100, log={"count": 3, "start": 1}) == [1, 10, 100]
    # The union deduplicates where generators overlap, and stays sorted.
    assert snapshot_steps(100, every=25, at=[7, 50], log={"count": 3, "start": 1}) == [
        1,
        7,
        10,
        25,
        50,
        75,
        100,
    ]


def test_snapshot_schedule_covers_the_configured_default() -> None:
    """The shipped config: 2k, 5k, then every 10k."""
    steps = snapshot_steps(100000, every=10000, at=[2000, 5000])
    assert steps == [2000, 5000] + list(range(10000, 100001, 10000))
    assert len(steps) == 12


def test_final_step_is_always_snapshotted() -> None:
    # Without it the terminal weights could fall outside every generator, and
    # analysis code would have to special-case the end of training.
    assert snapshot_steps(100) == [100]
    assert snapshot_steps(100, every=30) == [30, 60, 90, 100]


def test_snapshot_schedule_drops_out_of_range_and_dedupes_log_collisions() -> None:
    assert snapshot_steps(100, at=[0, 50, 250]) == [50, 100]
    # Log spacing over a short run rounds many points onto the same integers.
    dense = snapshot_steps(10, log={"count": 40, "start": 1})
    assert dense == sorted(set(dense))
    assert dense[0] == 1 and dense[-1] == 10


def test_snapshot_schedule_rejects_nonsense_cadences() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        snapshot_steps(100, every=0)
    with pytest.raises(ValueError, match="start >= 1"):
        snapshot_steps(100, log={"count": 3, "start": 0})


def test_trainer_writes_the_resolved_snapshot_series(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, num_steps=4, snapshots={"every": 2, "at": [1], "log": None})
    torch.manual_seed(0)
    trainer = Trainer(cfg, out_dir=tmp_path / "run")
    trainer.fit()

    snapshot_dir = tmp_path / "run" / "snapshots"
    assert sorted(p.name for p in snapshot_dir.glob("*.pt")) == [
        "step_0000001.pt",
        "step_0000002.pt",
        "step_0000004.pt",
    ]
    state = torch.load(snapshot_dir / "step_0000004.pt", weights_only=False)
    # eval.py needs both to score a snapshot and link it to its training run.
    assert state["step"] == 4
    assert "wandb_run" in state
    assert "optimizer" not in state


def test_snapshots_can_carry_optimizer_state(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, num_steps=2, snapshots={"include_optimizer_state": True})
    torch.manual_seed(0)
    Trainer(cfg, out_dir=tmp_path / "run").fit()
    state = torch.load(tmp_path / "run" / "snapshots" / "step_0000002.pt", weights_only=False)
    assert "optimizer" in state and "scheduler" in state


def test_trainer_smoke_and_checkpoint(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, num_steps=3, checkpoint_every=2)
    torch.manual_seed(0)
    trainer = Trainer(cfg, out_dir=tmp_path / "run")
    trainer.fit()
    assert math.isfinite(trainer.last_loss)
    assert trainer.step == 3
    assert (tmp_path / "run" / "checkpoints" / "last.pt").exists()


def test_variable_world_mixed_length_training_smoke(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, num_steps=2, batch_size=4)
    cfg.data.num_modules = {"min": 4, "max": 7, "held_out": [6]}
    del cfg.data.sequence.phases
    cfg.data.sequence.curriculum_sampler = "constructive"
    cfg.data.sequence.hotness = 2
    cfg.data.sequence.surplus_tasks = [0, 2]
    cfg.data.sequence.demos_per_task = {"min": 2, "max": 5, "scope": "per_task"}

    torch.manual_seed(0)
    trainer = Trainer(cfg, out_dir=tmp_path / "variable")
    trainer.fit()
    assert math.isfinite(trainer.last_loss)
    assert trainer.step == 2


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


def test_validation_requires_frozen_suites(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, validation_every=2)
    trainer = Trainer(cfg, out_dir=tmp_path / "run")
    with pytest.raises(FileNotFoundError, match="make_eval_sets"):
        trainer.fit()


def test_trainer_runs_validation_and_tracks_best(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path, num_steps=2, validation_every=2)
    out_dir = Path(cfg.data.eval_sets.out_dir)
    cfg.data.eval_sets = OmegaConf.load("configs/data/hyperteacher.yaml").eval_sets
    cfg.data.eval_sets.out_dir = str(out_dir)
    cfg.data.eval_sets.module_counts = [8]
    cfg.data.eval_sets.num_sequences = 8
    cfg.data.eval_sets.demos_per_task = 2
    cfg.data.eval_sets.task_variation.surplus_tasks = {"min": 1, "max": 1}
    cfg.data.eval_sets.retention.position_diagnostic.num_worlds = 2
    prepare_eval_bundle(cfg)

    torch.manual_seed(0)
    trainer = Trainer(cfg, out_dir=tmp_path / "run")
    trainer.fit()
    assert math.isfinite(trainer.best_metric)
    assert (tmp_path / "run" / "checkpoints" / "best.pt").exists()
