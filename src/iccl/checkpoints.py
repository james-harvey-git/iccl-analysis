"""Checkpoint resolution and source-run provenance."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
