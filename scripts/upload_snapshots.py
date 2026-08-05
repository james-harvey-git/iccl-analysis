"""Promotes a finished run's whole snapshot series to a W&B artifact.

Training uploads only the final snapshot, and only when ``wandb.upload_weights``
is set. The full series is worth keeping for the runs whose results get
reported — a judgement made after seeing the results, hence a separate
invocation rather than a training-time flag. Running this script is itself the
expression of intent, so it needs no config flag of its own.

    uv run python scripts/upload_snapshots.py outputs/<date>/<time>

The run identity is read out of the snapshots, so the artifact attaches to the
run that produced them rather than appearing as an orphan. A run trained offline
must be synced before promoting it: until then its id does not exist server-side,
and resuming it would open a new run instead of attaching to the original.
"""

import argparse
from pathlib import Path
from typing import Any

import torch

from iccl.training.logger import weights_artifact_name


def run_reference(snapshot: Path) -> dict[str, Any]:
    """The W&B run recorded in a snapshot by the trainer that wrote it."""
    reference = torch.load(snapshot, map_location="cpu", weights_only=False).get("wandb_run")
    if reference is None:
        raise SystemExit(
            f"{snapshot} carries no W&B run reference: it was trained with "
            "wandb.mode=disabled, so there is no run for an artifact to attach to."
        )
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="hydra run dir holding snapshots/")
    parser.add_argument(
        "--alias", default="canonical", help="alias for the uploaded version (default: canonical)"
    )
    args = parser.parse_args()

    snapshots = sorted((args.run_dir / "snapshots").glob("step_*.pt"))
    if not snapshots:
        raise SystemExit(f"no snapshots under {args.run_dir / 'snapshots'}")
    reference = run_reference(snapshots[0])

    import wandb

    run = wandb.init(
        entity=reference["entity"],
        project=reference["project"],
        id=reference["run_id"],
        resume="allow",
    )
    artifact = wandb.Artifact(
        weights_artifact_name(reference["name"] or reference["run_id"]), type="model"
    )
    for snapshot in snapshots:
        artifact.add_file(str(snapshot))
    total_mb = sum(snapshot.stat().st_size for snapshot in snapshots) / 1e6
    print(
        f"uploading {len(snapshots)} snapshots ({total_mb:.0f} MB) as {artifact.name}:{args.alias}"
    )
    run.log_artifact(artifact, aliases=[args.alias])
    run.finish()


if __name__ == "__main__":
    main()
