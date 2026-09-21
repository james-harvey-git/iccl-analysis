"""Checkpoint resolution and source-run provenance."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

WANDB_SCHEME = "wandb://"


@dataclass(frozen=True)
class SourceRun:
    """The W&B run that produced an evaluated checkpoint."""

    entity: str | None
    project: str | None
    run_id: str
    name: str | None
    step: int

    @property
    def label(self) -> str:
        """Display name, falling back to the stable run id."""
        return self.name or self.run_id

    @property
    def url(self) -> str | None:
        """W&B run URL when its entity and project are known."""
        if self.entity is None or self.project is None:
            return None
        return f"https://wandb.ai/{self.entity}/{self.project}/runs/{self.run_id}"


def source_from_checkpoint(checkpoint: dict[str, Any]) -> SourceRun | None:
    """Return the source run recorded in a checkpoint, when available."""
    reference = checkpoint.get("wandb_run")
    if reference is None:
        return None
    return SourceRun(step=int(checkpoint["step"]), **reference)


def evaluation_checkpoint_references(cfg: DictConfig) -> list[str]:
    """Resolve the explicitly configured evaluation checkpoint or trajectory."""
    references = [str(value) for value in cfg.evaluation.get("checkpoints", [])]
    if not references:
        raise ValueError("evaluation.checkpoints must contain at least one checkpoint reference")
    return references


def resolve_checkpoint_path(reference: str) -> tuple[Path, bool]:
    """Resolve a local path or download one checkpoint from a W&B artifact.

    The boolean reports whether the reference was an artifact. Resolving an
    artifact uses the read-only public API and does not initialize a W&B run.
    """
    if not reference.startswith(WANDB_SCHEME):
        return Path(reference), False

    import wandb

    artifact = wandb.Api().artifact(reference.removeprefix(WANDB_SCHEME), type="model")
    checkpoints = sorted(Path(artifact.download()).glob("*.pt"))
    if len(checkpoints) != 1:
        names = ", ".join(path.name for path in checkpoints) or "none"
        raise ValueError(
            f"{reference} holds {len(checkpoints)} checkpoints ({names}); a promoted "
            "series has many, so download it and pass the path of the one to score"
        )
    return checkpoints[0], True


def checkpoint_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Architecture and token dimensions, independent of execution backend."""
    cfg = checkpoint["config"]
    model = dict(cfg["model"])
    model.pop("backend", None)
    return {
        "model": model,
        "data": {key: cfg["data"][key] for key in ("input_dim", "output_dim")},
    }


def checkpoint_model_digest(checkpoint: dict[str, Any]) -> str:
    """Hash architecture and tensors rather than serialization or optimizer state."""
    digest = hashlib.sha256(
        json.dumps(checkpoint_model_config(checkpoint), sort_keys=True).encode()
    )
    for name, tensor in sorted(checkpoint["model"].items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_evaluation_config(checkpoint: dict[str, Any], cfg: DictConfig) -> None:
    """Require compatible architecture and teacher parameters, allowing new eval suites."""
    model = dict(cfg.model)
    model.pop("backend", None)
    if model != checkpoint_model_config(checkpoint)["model"]:
        raise ValueError("Evaluation model architecture differs from the checkpoint")
    trained = OmegaConf.create(checkpoint["config"]["data"])
    assert isinstance(trained, DictConfig)
    for key in ("input_dim", "output_dim", "hidden_dims", "use_bias", "scale"):
        if cfg.data[key] != trained[key]:
            raise ValueError(f"Evaluation teacher data.{key} differs from the checkpoint")
    if cfg.data.sequence.signal_boundaries != trained.sequence.signal_boundaries:
        raise ValueError("Evaluation signal_boundaries differs from the checkpoint")
