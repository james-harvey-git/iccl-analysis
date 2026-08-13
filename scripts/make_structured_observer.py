"""Generate or reuse structured GP observer predictions for a frozen suite."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from iccl.analysis.bayes_oracle import suite_paths
from iccl.analysis.structured_observer.cache import (
    generate_or_reuse_cache,
    settings_from_config,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    settings = settings_from_config(cfg.structured_observer)
    suite_path, metadata_path = suite_paths(
        Path(cfg.data.eval_sets.out_dir),
        str(cfg.structured_observer.suite),
        cfg.structured_observer.get("suite_path"),
    )
    cache_path, generated = generate_or_reuse_cache(
        suite_path,
        metadata_path,
        Path(cfg.structured_observer.cache_dir),
        settings,
    )
    action = "generated" if generated else "reused"
    print(f"{action} structured-observer cache: {cache_path}")


if __name__ == "__main__":
    main()
