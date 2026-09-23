from pathlib import Path

import pytest
from omegaconf import DictConfig

from iccl.analysis.probe_benchmark import benchmark_probe


def test_benchmark_runs_real_updates_with_separate_windows(
    probe_cfg: DictConfig, tmp_path: Path
) -> None:
    probe_cfg.probe.benchmark.capture_first = True
    report = benchmark_probe(probe_cfg, tmp_path / "benchmark")
    assert report["parameters"] == 64 * 4608 + 4608
    assert report["decoder"] == "full_affine"
    assert not report["reference_parameter_count"]
    assert report["capture"]["new_episodes"] == 8
    warm, measured, profiled = [report[key] for key in ("warmup", "measured", "profiled")]
    assert [value["updates"] for value in (warm, measured, profiled)] == [1, 2, 1]
    assert [value["end_step_inclusive"] for value in (warm, measured, profiled)] == [1, 3, 4]
    assert measured["episodes"] == 4  # One partial batch plus one complete batch.
    assert measured["episodes_per_second"] > 0
    assert measured["stage_seconds_distributions"] == {}
    stages = profiled["stage_seconds_distributions"]
    assert {
        "data_wait",
        "host_to_device",
        "decoder_forward",
        "prediction_to_cpu",
        "native_matching_wall",
        "assignment_return_and_loss",
        "backward",
        "gradient_check_and_clip",
        "optimizer_and_scheduler",
    } == set(stages)
    assert all(
        row["loss_and_gradient_finite"]
        for value in (warm, measured, profiled)
        for row in value["records"]
    )
    assert report["resources_after_measured"]["host_peak_rss_bytes"] > 0
    assert report["timing_scope"]["checkpoint_write_seconds"] == 0
    assert (tmp_path / "benchmark/benchmark.json").is_file()
    assert not (tmp_path / "benchmark/checkpoints").exists()
    probe_cfg.probe.benchmark.capture_first = False
    probe_cfg.probe.benchmark.warmup_steps = 0
    probe_cfg.probe.benchmark.measured_steps = 1
    probe_cfg.probe.benchmark.profile_steps = 0
    unprofiled = benchmark_probe(probe_cfg, tmp_path / "unprofiled")
    assert unprofiled["profiled"]["updates"] == 0
    assert unprofiled["profiled"]["records"] == []
    assert unprofiled["measured"]["updates"] == 1


def test_benchmark_rejects_constant_control_before_capture(
    probe_cfg: DictConfig, tmp_path: Path
) -> None:
    probe_cfg.probe.training.control = "constant"
    with pytest.raises(ValueError, match="full-decoder benchmark"):
        benchmark_probe(probe_cfg, tmp_path)
