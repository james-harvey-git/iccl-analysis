"""Run the resumable kernel/particle convergence grid for the full observer."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from iccl.analysis.bayes_oracle import load_suite_metadata, suite_paths
from iccl.analysis.structured_observer.cache import (
    settings_from_config,
    structured_suite_spec,
)
from iccl.analysis.structured_observer.calibration import run_convergence_calibration
from iccl.data.export import load_suite


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    observer_cfg = cfg.structured_observer
    calibration_cfg = observer_cfg.calibration
    settings = settings_from_config(observer_cfg)
    suite_path, metadata_path = suite_paths(
        Path(cfg.data.eval_sets.out_dir),
        str(observer_cfg.suite),
        observer_cfg.get("suite_path"),
    )
    suite = load_suite(suite_path.with_suffix(""))
    spec = structured_suite_spec(load_suite_metadata(metadata_path))
    threshold_cfg = calibration_cfg.thresholds
    written = run_convergence_calibration(
        suite=suite,
        suite_path=suite_path,
        metadata_path=metadata_path,
        spec=spec,
        base_settings=settings,
        feature_counts=tuple(int(value) for value in calibration_cfg.feature_counts),
        particle_counts=tuple(int(value) for value in calibration_cfg.particle_counts),
        sequence_limit=int(calibration_cfg.sequence_limit),
        cache_dir=Path(calibration_cfg.cache_dir),
        output_dir=Path(calibration_cfg.output_dir),
        thresholds={
            "kernel_relative_frobenius": float(
                threshold_cfg.kernel_relative_frobenius
            ),
            "curve_max_abs_change": float(threshold_cfg.curve_max_abs_change),
            "seed_standard_error": float(threshold_cfg.seed_standard_error),
            "relative_jitter": float(threshold_cfg.relative_jitter),
        },
    )
    print("wrote structured-observer convergence outputs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
