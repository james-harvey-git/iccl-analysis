"""Evaluate one checkpoint or an explicit checkpoint trajectory.

Re-scores finished weights without retraining. ``evaluation.checkpoints`` holds
one reference or an ordered trajectory of local paths or
``wandb://entity/project/name:alias`` artifacts. Downloaded artifacts are
recorded as inputs so W&B ties the measurements to their source weights. The
W&B run contains a compact summary, while complete aggregate rows and per-demo
errors are written locally and optionally uploaded as one evaluation artifact.
"""

from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from iccl.checkpoints import (
    WANDB_SCHEME,
    evaluation_checkpoint_references,
    resolve_checkpoint_path,
    source_from_checkpoint,
)
from iccl.evaluation.metrics import evaluate_suites, load_eval_suites
from iccl.evaluation.results import write_evaluation_results
from iccl.models.model import model_from_config
from iccl.reporting.logger import RunLogger
from iccl.training.trainer import resolve_autocast_dtype
from iccl.utils import resolve_device, seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    references = evaluation_checkpoint_references(cfg)
    first_path, first_is_artifact = resolve_checkpoint_path(references[0])
    first_checkpoint = torch.load(first_path, map_location=device, weights_only=False)
    model = model_from_config(cfg).to(device)
    model.eval()

    out_dir = Path(HydraConfig.get().runtime.output_dir)
    suites = load_eval_suites(Path(cfg.data.eval_sets.out_dir))
    results_dir = out_dir / "evaluation-results"
    first_source = source_from_checkpoint(first_checkpoint)
    job_type = "eval" if len(references) == 1 else "eval-trajectory"
    logger = RunLogger(cfg, out_dir, job_type=job_type, source=first_source)
    logger.start()
    previous_step = -1
    source_id = first_source.run_id if first_source is not None else None
    for index, reference in enumerate(references):
        if index == 0:
            path, is_artifact, checkpoint = first_path, first_is_artifact, first_checkpoint
        else:
            path, is_artifact = resolve_checkpoint_path(reference)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        step = int(checkpoint["step"])
        if step <= previous_step:
            raise ValueError(
                "evaluation.checkpoints must have strictly increasing training steps; "
                f"got {step} after {previous_step}"
            )
        source = source_from_checkpoint(checkpoint)
        current_source_id = source.run_id if source is not None else None
        if current_source_id != source_id:
            raise ValueError("an evaluation trajectory must come from one training run")
        previous_step = step
        print(f"device: {device}; checkpoint step {step}; suites: {len(suites)}")
        if is_artifact:
            logger.use_artifact(reference.removeprefix(WANDB_SCHEME))
        model.load_state_dict(checkpoint["model"])
        report = evaluate_suites(
            model,
            suites,
            device,
            autocast_dtype=resolve_autocast_dtype(cfg.training.precision, device),
            bootstrap_seed=int(cfg.data.eval_sets.get("bootstrap_seed", 0)),
            bootstrap_replicates=int(cfg.data.eval_sets.get("bootstrap_replicates", 1000)),
        )
        logger.log_full_evaluation(report, step)
        result_path = write_evaluation_results(
            report,
            results_dir,
            step,
            {
                "checkpoint_reference": reference,
                "checkpoint_path": str(path),
                "source_run": None if source is None else asdict(source),
                "evaluation_config": OmegaConf.to_container(cfg, resolve=True),
                "training_config": checkpoint.get("config"),
                "suites": {name: suite["__meta__"] for name, suite in suites.items()},
            },
        )
        print(f"evaluation results written to {result_path}")
    logger.upload_evaluation_results(results_dir)
    logger.finish()


if __name__ == "__main__":
    main()
