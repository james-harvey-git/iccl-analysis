"""Exports the frozen evaluation suites defined by the data config.

See ``iccl.data.export.export_eval_sets`` for what the suites are.
"""

import hydra
from omegaconf import DictConfig

from iccl.data.export import export_eval_sets


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    export_eval_sets(cfg)


if __name__ == "__main__":
    main()
