"""Readable probe run identities and dataset facts from the captured manifest."""

from dataclasses import asdict
from typing import Any

from omegaconf import DictConfig, OmegaConf

from iccl.analysis.probe_config import resolved_config
from iccl.analysis.probe_results import source_run
from iccl.checkpoints import SourceRun


def tracking_config(
    cfg: DictConfig,
    manifest: dict[str, Any],
    stage: str,
    *,
    training_config: dict[str, Any] | None = None,
    resume_run: SourceRun | None = None,
) -> DictConfig:
    """Describe the actual data and trained probe without changing execution settings."""
    values = resolved_config(cfg)
    identity = manifest["identity"]
    values["data"].update(identity["data"])
    values["probe"]["dataset"].update(
        seed=identity["dataset_seed"],
        counts=manifest["requested_counts"],
        shard_size=manifest["shard_size"],
    )
    if training_config is not None:
        values["probe"]["training"] = training_config["probe"]["training"]
        values["seed"] = training_config["seed"]
    values["captured_dataset"] = {
        "dataset_id": manifest["dataset_id"],
        "identity": identity,
        "requested_counts": manifest["requested_counts"],
    }
    gdn = source_run(manifest)
    if gdn is not None:
        values["gdn_source_run"] = asdict(gdn) | {"url": gdn.url}
    wandb = values["wandb"]
    training = values["probe"]["training"]
    schedule = {"cosine": "Cosine", "constant": "Constant LR"}[training["schedule"]]
    control = {"none": "full", "constant": "constant output", "shuffled_targets": "shuffled"}[
        training["control"]
    ]
    if not wandb.get("name"):
        wandb["name"] = (
            resume_run.name
            if resume_run is not None and resume_run.name
            else f"{schedule} | {control} | seed {values['seed']} | {stage}"
        )
    if not wandb.get("group"):
        wandb["group"] = (
            f"Module set M4 T{identity['task_count']} D32 | {manifest['dataset_id'][:8]}"
        )
    return OmegaConf.create(values)
