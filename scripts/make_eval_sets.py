"""Prepare the authoritative validation, capability and position-diagnostic bundle."""

import hydra
from omegaconf import DictConfig

from iccl.data.eval_bundle import prepare_eval_bundle


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    prepare_eval_bundle(cfg)


if __name__ == "__main__":
    main()
