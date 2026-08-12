"""Generate or reuse the deterministic oracle cache for one frozen suite."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from iccl.analysis.bayes_oracle import (
    generate_or_reuse_cache,
    oracle_config_from,
    suite_paths,
)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    oracle_cfg = oracle_config_from(cfg.bayes_oracle)
    suite_path, metadata_path = suite_paths(
        Path(cfg.data.eval_sets.out_dir),
        str(cfg.bayes_oracle.suite),
        cfg.bayes_oracle.get("suite_path"),
    )
    cache_path, generated = generate_or_reuse_cache(
        suite_path,
        metadata_path,
        Path(cfg.bayes_oracle.cache_dir),
        oracle_cfg,
    )
    action = "generated" if generated else "reused"
    print(f"{action} known-world per-task Bayes oracle cache: {cache_path}")


if __name__ == "__main__":
    main()
