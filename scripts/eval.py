"""Evaluates a saved checkpoint on the frozen eval suites.

Re-scores a finished run without retraining — for metric changes, which alter
what the suites report about weights that are already trained. The checkpoint
comes from ``training.resume``, as in training: either a local path, or a
``wandb://entity/project/name:alias`` artifact reference, which is downloaded
and recorded as an input so W&B ties these numbers to the weights behind them.
Results go wherever training's do: stdout, an ``.npz`` of curves in the hydra run
dir, and a W&B run unless ``wandb.mode=disabled``, tagged ``eval`` and linked to
the run that produced the checkpoint when it recorded one.
"""

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from iccl.models.model import model_from_config
from iccl.training.logger import RunLogger, source_from_checkpoint
from iccl.training.metrics import evaluate_suites, load_eval_suites
from iccl.training.trainer import resolve_autocast_dtype
from iccl.utils import resolve_device, seed_everything

WANDB_SCHEME = "wandb://"


def download_artifact(reference: str) -> Path:
    """The single checkpoint inside a ``wandb://entity/project/name:alias``
    artifact. An explicit scheme rather than pattern-matching, so a local path is
    never mistaken for an artifact reference."""
    import wandb

    artifact = wandb.Api().artifact(reference.removeprefix(WANDB_SCHEME), type="model")
    checkpoints = sorted(Path(artifact.download()).glob("*.pt"))
    if len(checkpoints) != 1:
        names = ", ".join(path.name for path in checkpoints) or "none"
        raise ValueError(
            f"{reference} holds {len(checkpoints)} checkpoints ({names}); a promoted "
            "series has many, so download it and pass the path of the one to score"
        )
    return checkpoints[0]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.training.resume is None:
        raise ValueError(
            "pass the checkpoint to evaluate, e.g. `uv run python scripts/eval.py "
            "training.resume=outputs/<run>/checkpoints/last.pt`"
        )
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    reference = str(cfg.training.resume)
    is_artifact = reference.startswith(WANDB_SCHEME)
    path = download_artifact(reference) if is_artifact else Path(reference)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    step = int(checkpoint["step"])
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    suites = load_eval_suites(Path(cfg.data.eval_sets.out_dir))
    print(f"device: {device}; checkpoint step {step}; suites: {', '.join(sorted(suites))}")

    logger = RunLogger(cfg, out_dir, job_type="eval", source=source_from_checkpoint(checkpoint))
    logger.start()
    if is_artifact:
        # After start, so there is a run for the lineage edge to attach to.
        logger.use_artifact(reference.removeprefix(WANDB_SCHEME))
    scalars, curves = evaluate_suites(
        model,
        suites,
        device,
        autocast_dtype=resolve_autocast_dtype(cfg.training.precision, device),
    )
    path = logger.log_eval(scalars, curves, step)
    logger.finish()
    print(f"curves written to {path}")


if __name__ == "__main__":
    main()
