import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from iccl.analysis.capture import capture_dataset
from iccl.analysis.probe_dataset import CapturedDataset
from iccl.analysis.probe_evaluation import evaluate_probe
from iccl.analysis.probe_matching import AssignmentSolver
from iccl.analysis.probe_results import probe_summary_rows, read_results
from iccl.analysis.probe_training import ProbeTrainer, optimizer_update
from iccl.analysis.probes import LinearModuleDecoder
from iccl.reporting.logger import RunLogger
from iccl.training.trainer import build_optimizer, build_scheduler


def test_capture_train_interrupted_resume_and_evaluate(
    probe_cfg: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    with ProbeTrainer(probe_cfg, tmp_path / "complete") as trainer:
        expected = trainer.fit()
        weights = {
            name: value.detach().clone() for name, value in trainer.model.state_dict().items()
        }
        optimizer = trainer.optimizer.state_dict()
        next_indices = trainer.next_batch()["episode_index"].clone()

    class Interrupted(Exception):
        pass

    with ProbeTrainer(probe_cfg, tmp_path / "interrupted") as trainer:
        original = trainer.next_batch

        def interrupted_batch() -> dict[str, torch.Tensor]:
            if trainer.step == 2:
                raise Interrupted
            return original()

        trainer.next_batch = interrupted_batch
        with pytest.raises(Interrupted):
            trainer.fit()
    probe_cfg.probe.training.resume = str(tmp_path / "interrupted/checkpoints/last.pt")
    # Worker count affects prefetching, never consumed indices or optimizer continuity.
    probe_cfg.probe.training.num_workers = 1
    with ProbeTrainer(probe_cfg, tmp_path / "interrupted") as trainer:
        actual = trainer.fit()
        assert actual["samples_consumed"] == expected["samples_consumed"] == 8
        assert actual["best_step"] == expected["best_step"]
        assert actual["best_validation_joint_mse"] == expected["best_validation_joint_mse"]
        for key, value in trainer.model.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        for key, state in trainer.optimizer.state_dict()["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(value, optimizer["state"][key][name], rtol=0, atol=0)
        torch.testing.assert_close(trainer.next_batch()["episode_index"], next_indices)
    probe_cfg.probe.training.num_workers = 0
    probe_cfg.probe.evaluation.checkpoint = str(tmp_path / "interrupted/checkpoints/best.pt")
    dashboard = Mock()
    monkeypatch.setattr(RunLogger, "log_probe_evaluation", dashboard)
    result = evaluate_probe(probe_cfg, tmp_path / "evaluation")
    metadata, arrays, summary = read_results(result)
    assert metadata["split"] == "test"
    assert arrays["decoder_module_mse_by_module"].shape == (2, 4)
    assert summary["decoder"]["all"]["n_episodes"] == 2
    assert (result / "plots/reconstruction-by-exposure.png").is_file()
    assert (result / "plots/control-comparison.png").is_file()
    assert len(list((result / "plots").glob("*.html"))) == 5
    dashboard.assert_called_once()
    metrics, rows, figures, step = dashboard.call_args.args
    assert step == metadata["step"] == expected["best_step"]
    assert dashboard.call_args.kwargs == {"namespace": "probe/test"}
    assert len(metrics) == 14 and len(figures) == 5

    def row_key(row: dict) -> tuple:
        return (
            row["prediction"],
            row["population"],
            row["metric"],
            row["task_position"] or 0,
            row["exposure_count"] or 0,
        )

    assert sorted(rows, key=row_key) == sorted(probe_summary_rows(summary), key=row_key)
    for prediction in ("decoder", "zero"):
        for metric in ("joint_mse", "functional_nmse", "module_mse"):
            assert (
                metrics[f"probe/test/{prediction}/{metric}"]
                == summary[prediction]["all"]["metrics"][metric]["mean"]
            )
    assert all(key.startswith("probe/test/figures/") for key in figures)
    with pytest.raises(FileExistsError):
        evaluate_probe(probe_cfg, tmp_path / "evaluation")


@pytest.mark.parametrize("control", ["constant", "shuffled_targets"])
def test_explicit_controls_train_and_leave_validation_paired(
    probe_cfg: DictConfig, tmp_path: Path, control: str
) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    probe_cfg.probe.training.control = control
    with ProbeTrainer(probe_cfg, tmp_path / control) as trainer:
        if control == "constant":
            assert sum(p.numel() for p in trainer.model.parameters()) == 1344
            torch.testing.assert_close(
                trainer.model(torch.randn(2, 64))[0], trainer.model(torch.randn(3, 64))[2]
            )
        else:
            assert sum(p.numel() for p in trainer.model.parameters()) == 64 * 1344 + 1344
            assert trainer.mapping is not None
            assert np.all(trainer.mapping != np.arange(4))
        assert trainer.validation.target_permutation is None
        trainer.fit()
        checkpoint = torch.load(tmp_path / control / "checkpoints/last.pt", weights_only=False)
        assert checkpoint["control"] == control
        assert checkpoint["step"] == 4
    probe_cfg.probe.evaluation.checkpoint = str(tmp_path / control / "checkpoints/best.pt")
    # Evaluation takes model kind and loss from the checkpoint, not current training defaults.
    probe_cfg.probe.training.control = "none"
    probe_cfg.probe.loss.readout_weight = 17
    metadata, _, _ = read_results(evaluate_probe(probe_cfg, tmp_path / "evaluation"))
    assert metadata["control"] == control and metadata["readout_weight"] == 1


def test_resume_rejects_changed_settings_and_population(
    probe_cfg: DictConfig, tmp_path: Path
) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    with ProbeTrainer(probe_cfg, tmp_path / "train") as trainer:
        trainer.update(trainer.next_batch())
        path = trainer.save("last")
    probe_cfg.probe.training.resume = str(path)
    probe_cfg.probe.training.lr *= 2
    with pytest.raises(ValueError, match="same training"):
        ProbeTrainer(probe_cfg, tmp_path / "invalid")
    probe_cfg.probe.training.lr /= 2
    probe_cfg.probe.dataset.counts.train = 6
    capture_dataset(probe_cfg, tmp_path / "extended")
    with pytest.raises(ValueError, match="split changed"):
        ProbeTrainer(probe_cfg, tmp_path / "invalid2")
    CapturedDataset(probe_cfg.probe.dataset.path, "test").close()


def test_old_checkpoint_requires_new_capture_and_training(
    probe_cfg: DictConfig, tmp_path: Path
) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    old = tmp_path / "old.pt"
    torch.save({"version": "module-decoder-v1/checkpoint-v1"}, old)
    probe_cfg.probe.training.resume = str(old)
    with pytest.raises(ValueError, match="capture fresh M=4"):
        ProbeTrainer(probe_cfg, tmp_path / "train")


@pytest.mark.parametrize("resume_wandb", [False, True])
def test_checkpoint_resume_can_continue_its_wandb_run(
    probe_cfg: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume_wandb: bool
) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    with ProbeTrainer(probe_cfg, tmp_path / "interrupted") as trainer:
        for _ in range(2):
            trainer.update(trainer.next_batch())
        state = trainer.checkpoint()
    path = tmp_path / "last.pt"
    probe_cfg.probe.training.resume = str(path)
    probe_cfg.wandb.mode = "online"
    probe_cfg.wandb.resume = True
    torch.save(state, path)
    with pytest.raises(ValueError, match="checkpoint with a W&B run identity"):
        ProbeTrainer(probe_cfg, tmp_path / "missing-identity")
    reference = {
        "entity": probe_cfg.wandb.entity,
        "project": probe_cfg.wandb.project,
        "run_id": "existing-probe",
        "name": "original-training",
    }
    state["wandb_run"] = reference
    torch.save(state, path)
    probe_cfg.wandb.resume = resume_wandb
    sink = SimpleNamespace(
        entity=reference["entity"],
        project=reference["project"],
        id=reference["run_id"] if resume_wandb else "new-probe",
        name=reference["name"],
        step=4 if resume_wandb else 0,
        log=Mock(),
        finish=Mock(),
    )
    init = Mock(return_value=sink)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    with ProbeTrainer(probe_cfg, tmp_path / "resumed") as trainer:
        assert trainer.step == 2
        trainer.fit()
        assert trainer.checkpoint()["wandb_run"]["run_id"] == sink.id
    assert init.call_args.kwargs["resume"] == ("must" if resume_wandb else "never")
    assert init.call_args.kwargs["id"] == (reference["run_id"] if resume_wandb else None)
    assert [call.kwargs["step"] for call in sink.log.call_args_list] == (
        [4] if resume_wandb else [3, 4]
    )


def test_training_flushes_first_and_validation_metrics_before_next_update(
    probe_cfg: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    probe_cfg.probe.training.log_every = 2
    sink = SimpleNamespace(
        entity="unit", project="unit", id="unit", name="unit", log=Mock(), finish=Mock()
    )
    with ProbeTrainer(probe_cfg, tmp_path / "train") as trainer:
        monkeypatch.setattr(trainer.logger, "start", lambda: None)
        monkeypatch.setattr(trainer.logger, "run", sink)
        original = trainer.next_batch

        def next_batch() -> dict[str, torch.Tensor]:
            if trainer.step == 1:
                assert sink.log.call_count == 1
                call = sink.log.call_args
                assert call.kwargs["step"] == 1
                assert "probe/train/loss" in call.args[0]
                assert call.args[0]["probe/train/data_wait_seconds"] >= 0
                assert (
                    call.args[0]["probe/train/seconds_per_update"]
                    >= call.args[0]["probe/train/data_wait_seconds"]
                )
            if trainer.step == 2:
                assert sink.log.call_count == 2
                call = sink.log.call_args
                assert call.kwargs["step"] == 2
                assert "probe/train/loss" in call.args[0]
                assert "probe/validation/joint_mse" in call.args[0]
            return original()

        monkeypatch.setattr(trainer, "next_batch", next_batch)
        trainer.fit()
        assert [call.kwargs["step"] for call in sink.log.call_args_list] == [1, 2, 4]
        assert all(call.kwargs["commit"] is True for call in sink.log.call_args_list)


def test_known_linear_signal_learns_through_the_production_assignment_update(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(44)
    model = LinearModuleDecoder(3)
    states = torch.randn(32, 3)
    truth = torch.randn(1344, 3) * 0.08
    offset = torch.randn(1344) * 3
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.copy_(offset)
    targets = states @ truth.T + offset
    config = OmegaConf.create(
        {
            "optimizer": "adamw",
            "lr": 0.03,
            "weight_decay": 0,
            "schedule": "constant",
            "warmup_steps": 0,
            "num_steps": 70,
        }
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    batch = {"states": states, "target": targets, "episode_index": torch.arange(32)}
    losses = []
    cache = Path(__file__).resolve().parents[1] / "outputs/.cache/probe-assignment-tests"
    with AssignmentSolver(2, cache) as solver:
        for _ in range(70):
            result = optimizer_update(
                model, optimizer, scheduler, batch, solver, torch.device("cpu"), None, 1, None
            )
            losses.append(result.loss)
    assert losses[-1] < losses[0] * 0.002
    assert torch.mean((model(states) - targets) ** 2).item() < losses[0] * 0.002
