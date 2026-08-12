"""Checkpoint-source resolution shared by standalone analysis entry points."""

from pathlib import Path

WANDB_SCHEME = "wandb://"


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
