"""Train the full linear module decoder or an explicitly selected control."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from iccl.analysis.probe_training import train_probe


@hydra.main(version_base=None, config_path="../configs", config_name="probe")
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    report = train_probe(cfg, out_dir)
    print(f"probe training reached step {report['step']}; run directory: {out_dir}")


if __name__ == "__main__":
    main()
