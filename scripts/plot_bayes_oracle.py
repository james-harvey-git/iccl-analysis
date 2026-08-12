"""Compare one explicit GDN checkpoint with the cached Bayes reference."""

import subprocess
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from iccl.analysis.bayes_oracle import (
    load_suite_metadata,
    oracle_config_from,
    resolve_cache,
    suite_paths,
)
from iccl.analysis.checkpoints import resolve_checkpoint_path
from iccl.analysis.plotting import plot_bayes_oracle_comparison
from iccl.data.export import load_suite
from iccl.models.model import model_from_config
from iccl.training.logger import source_from_checkpoint
from iccl.training.metrics import demo_mse, demo_nmse, predict_suite
from iccl.training.trainer import resolve_autocast_dtype
from iccl.utils import resolve_device, seed_everything


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.training.resume is None:
        raise ValueError(
            "pass an explicit checkpoint with training.resume=<local path or wandb:// reference>"
        )
    seed_everything(cfg.seed)
    oracle_cfg = oracle_config_from(cfg.bayes_oracle)
    suite_name = str(cfg.bayes_oracle.suite)
    suite_path, metadata_path = suite_paths(
        Path(cfg.data.eval_sets.out_dir),
        suite_name,
        cfg.bayes_oracle.get("suite_path"),
    )
    suite = load_suite(suite_path.with_suffix(""))
    suite_metadata = load_suite_metadata(metadata_path)
    cache_path, oracle, cache_metadata = resolve_cache(
        suite_path,
        metadata_path,
        Path(cfg.bayes_oracle.cache_dir),
        oracle_cfg,
        cfg.bayes_oracle.get("cache_path"),
    )

    checkpoint_reference = str(cfg.training.resume)
    checkpoint_path, is_artifact = resolve_checkpoint_path(checkpoint_reference)
    device = resolve_device(cfg.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions = predict_suite(
        model,
        suite,
        device,
        autocast_dtype=resolve_autocast_dtype(cfg.training.precision, device),
    )
    gdn_nmse = demo_nmse(predictions, suite)
    gdn_raw_mse = demo_mse(predictions, suite)
    step = int(checkpoint["step"])
    source = source_from_checkpoint(checkpoint)
    source_record = None if source is None else asdict(source) | {"url": source.url}

    written = plot_bayes_oracle_comparison(
        gdn_nmse,
        gdn_raw_mse,
        oracle,
        suite_name=suite_name,
        task_position=int(cfg.bayes_oracle.plotting.task_position),
        checkpoint_step=step,
        output_dir=Path(cfg.bayes_oracle.plotting.output_dir),
        bootstrap_replicates=int(cfg.bayes_oracle.plotting.bootstrap_replicates),
        bootstrap_seed=int(cfg.bayes_oracle.plotting.bootstrap_seed),
        include_raw_mse=bool(cfg.bayes_oracle.plotting.include_raw_mse),
        provenance={
            "checkpoint_reference": checkpoint_reference,
            "resolved_checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_step": step,
            "checkpoint_from_wandb_artifact": is_artifact,
            "source_run": source_record,
            "suite_path": str(suite_path.resolve()),
            "suite_metadata": suite_metadata,
            "oracle_cache_path": str(cache_path.resolve()),
            "oracle_cache_identity": cache_metadata["identity"],
            "plotting_git_commit": git_commit(),
        },
    )
    print("wrote Bayes-oracle comparison outputs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
