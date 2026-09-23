"""Capture independent teacher episodes and terminal GDN memory for module decoding."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from iccl.analysis.capture import capture_dataset


@hydra.main(version_base=None, config_path="../configs", config_name="probe")
def main(cfg: DictConfig) -> None:
    report = capture_dataset(cfg, HydraConfig.get().runtime.output_dir)
    print(f"captured {report['new_episodes']} new episodes at {report['dataset_path']}")


if __name__ == "__main__":
    main()
