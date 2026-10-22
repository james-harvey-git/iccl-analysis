"""Measure a bounded one-worker capture, fitting and reporting pilot."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from iccl.analysis.probe_benchmark import benchmark_probe


@hydra.main(version_base=None, config_path="../configs", config_name="probe")
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    report = benchmark_probe(cfg, out_dir)
    print(f"stage seconds: {report['stage_seconds']}; report: {out_dir}/benchmark.json")


if __name__ == "__main__":
    main()
