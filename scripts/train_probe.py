"""Fit separate within-world ridge probes and select lambda using validation only."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from iccl.analysis.probe_training import train_probe


@hydra.main(version_base=None, config_path="../configs", config_name="probe")
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    report = train_probe(cfg, out_dir)
    print(f"probe fitting complete: {report}")


if __name__ == "__main__":
    main()
