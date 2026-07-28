"""Exports frozen evaluation and analysis sets via iccl.data.export."""

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    raise NotImplementedError("implemented once the HyperTeacher port lands")


if __name__ == "__main__":
    main()
