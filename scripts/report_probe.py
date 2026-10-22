"""Replay durable study events and publish saved evaluation figures to the same W&B run."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from iccl.analysis.probe_storage import lock
from iccl.analysis.probe_tracking import StudyReporter


@hydra.main(version_base=None, config_path="../configs", config_name="probe")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.experiment.root)
    with lock(root / ".study-writer.lock", blocking=False):
        reporter = StudyReporter(cfg, root / cfg.experiment.phase)
        try:
            reporter.flush()
            results = Path(cfg.probe.evaluation.output)
            if (results / "summary.json").exists():
                reporter.publish_results(results)
        finally:
            reporter.finish(success=True)


if __name__ == "__main__":
    main()
