from operator import itemgetter
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
    assert arrays["decoder_module_mse_by_task"].shape == (2, 8)
    assert summary["decoder"]["all"]["n_episodes"] == 2
    assert (result / "plots/reconstruction-by-position.png").is_file()
    assert (result / "plots/control-comparison.png").is_file()
    dashboard.assert_called_once()
    metrics, rows, figures, step = dashboard.call_args.args
    assert step == metadata["step"] == expected["best_step"]
    assert dashboard.call_args.kwargs == {"namespace": "probe/test"}
    assert len(metrics) == 14 and len(figures) == 6
    row_key = itemgetter("prediction", "population", "metric", "task_position")
    assert sorted(rows, key=row_key) == sorted(probe_summary_rows(summary), key=row_key)
    for prediction in ("decoder", "zero"):
        for metric in ("joint_mse", "functional_nmse", "repeated_module_mse"):
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
            assert sum(p.numel() for p in trainer.model.parameters()) == 4608
            torch.testing.assert_close(
                trainer.model(torch.randn(2, 64))[0], trainer.model(torch.randn(3, 64))[2]
            )
        else:
            assert sum(p.numel() for p in trainer.model.parameters()) == 64 * 4608 + 4608
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
    truth = torch.randn(4608, 3) * 0.08
    offset = torch.randn(4608) * 3
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
