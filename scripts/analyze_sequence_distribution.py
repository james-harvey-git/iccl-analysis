"""Sample and report the configured on-the-fly training distribution."""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from iccl.data.diagnostics import analyze_sequence_distribution
from iccl.reporting.logger import RunLogger


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    report = analyze_sequence_distribution(cfg, out_dir)
    logger = RunLogger(cfg, out_dir, job_type="distribution-diagnostic")
    logger.start()
    metrics = {
        "distribution/generated_sequences": float(report["generated_sequences"]),
        "distribution/failure_count": float(report["failure_count"]),
        "distribution/sequences_per_second": float(report["sequences_per_second"]),
        "distribution/padding_expansion": float(report["batch_padding"]["expansion"]),
        "distribution/padding_fraction": float(report["batch_padding"]["padding_fraction"]),
    }
    for relative, categories in report["categories"].items():
        for category, estimators in categories.items():
            if category == "p_surplus_zero":
                metrics[f"distribution/categories/{relative}/p_surplus_zero"] = float(estimators)
                continue
            for estimator, estimate in estimators.items():
                if "estimate" not in estimate:
                    continue
                if estimate["estimate"] is None:
                    continue
                metrics[f"distribution/categories/{relative}/{estimator}/{category}"] = float(
                    estimate["estimate"]
                )
    logger.log(metrics, step=0)
    if logger.run is not None:
        import wandb

        distribution_rows = [
            [dimension, int(value), count, distribution["frequencies"][value]]
            for dimension, distribution in report["distributions"].items()
            for value, count in distribution["counts"].items()
        ]
        logger.run.log(
            {
                "distribution/tables/structural": wandb.Table(
                    columns=["dimension", "value", "count", "frequency"],
                    data=distribution_rows,
                ),
                "distribution/tables/categories": wandb.Table(
                    columns=list(report["stratified_categories"][0]),
                    data=[list(row.values()) for row in report["stratified_categories"]],
                ),
                "distribution/tables/rejection_efficiency": wandb.Table(
                    columns=list(report["rejection_efficiency"][0]),
                    data=[list(row.values()) for row in report["rejection_efficiency"]],
                ),
            },
            step=0,
        )
    logger.finish()
    print(f"distribution report written to {out_dir}")


if __name__ == "__main__":
    main()
