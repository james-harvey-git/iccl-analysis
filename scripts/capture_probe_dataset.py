"""Capture fixed-world trials at the three end-of-task points."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from iccl.analysis.capture import capture_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="probe")
def main(cfg: DictConfig) -> None:
    report = capture_dataset(cfg, HydraConfig.get().runtime.output_dir)
    print(f"capture complete: {report}")


if __name__ == "__main__":
    main()
