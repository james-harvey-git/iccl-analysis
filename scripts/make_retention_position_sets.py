"""Export the frozen paired serial-position and module-rehearsal diagnostic."""

import hydra
from omegaconf import DictConfig

from iccl.data.export import export_retention_position_sets


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    export_retention_position_sets(cfg)


if __name__ == "__main__":
    main()
