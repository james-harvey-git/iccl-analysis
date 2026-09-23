"""Evaluate a probe checkpoint on correctly paired held-out captured episodes."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from iccl.analysis.probe_evaluation import evaluate_probe


@hydra.main(version_base=None, config_path="../configs", config_name="probe")
def main(cfg: DictConfig) -> None:
    path = evaluate_probe(cfg, HydraConfig.get().runtime.output_dir)
    print(f"probe results: {path}")


if __name__ == "__main__":
    main()
