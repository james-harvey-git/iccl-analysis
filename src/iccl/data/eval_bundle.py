"""Manage one frozen evaluation bundle shared by training and standalone scoring.

The manifest binds validation, capability sweeps and position diagnostics to
their generation config and file checksums. Builds finish in a staging directory
before replacing the active bundle; previous contents remain recoverable under
``outputs/eval-set-backups/``. No run silently regenerates its evaluation data.
"""

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from iccl.data.export import _sha256, export_eval_sets, export_retention_position_sets

# Increment when frozen sampling or archive semantics change incompatibly.
BUNDLE_VERSION = 1


def generation_config(cfg: DictConfig) -> dict[str, Any]:
    """Exclude optimization, reporting and paths from frozen-data identity."""
    data = cast(dict[str, Any], OmegaConf.to_container(cfg.data, resolve=True))
    data.pop("distribution_diagnostic", None)
    for key in ("out_dir", "bootstrap_seed", "bootstrap_replicates", "best_metric"):
        data["eval_sets"].pop(key, None)
    return {"seed": int(cfg.seed), "data": data}


def validate_eval_bundle(cfg: DictConfig) -> dict[str, Any]:
    """Reject incomplete, stale or differently configured frozen data."""
    root = Path(cfg.data.eval_sets.out_dir)
    path = root / "manifest.json"
    remedy = "Run scripts/make_eval_sets.py with the same data/seed overrides."
    if not path.is_file():
        raise FileNotFoundError(f"No authoritative eval manifest in {root}. {remedy}")
    manifest = json.loads(path.read_text())
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError(f"Obsolete evaluation bundle in {root}. {remedy}")
    if manifest.get("generation_config") != generation_config(cfg):
        raise ValueError(
            f"Evaluation bundle in {root} does not match the data/seed config "
            f"(including training-distribution validation). {remedy}"
        )
    files = manifest["files"]
    actual = {p.name for p in root.iterdir() if p.name not in {"manifest.json", ".DS_Store"}}
    if actual != set(files):
        raise ValueError(f"Evaluation bundle in {root} has missing or extra files. {remedy}")
    for name, checksum in files.items():
        file = root / name
        if not file.is_file() or _sha256(file) != checksum:
            raise ValueError(f"Evaluation bundle checksum mismatch: {file}. {remedy}")
    return manifest


def prepare_eval_bundle(cfg: DictConfig) -> Path:
    """Reuse a matching bundle or build and publish a complete replacement."""
    root = Path(cfg.data.eval_sets.out_dir).absolute()
    # An export may replace only a dedicated directory, never a project/data root.
    if root.is_symlink() or root.resolve() in {
        Path.home(),
        *Path.cwd().resolve().parents,
        Path.cwd().resolve(),
        Path.cwd() / "data",
    }:
        raise ValueError(f"Use a dedicated, non-symlink evaluation directory, not {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = root.with_name(f".{root.name}.lock")
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"Evaluation build already locked at {lock}; retry after it finishes"
        ) from None
    try:
        try:
            manifest = validate_eval_bundle(cfg)
        except (FileNotFoundError, ValueError) as error:
            print(error)
        else:
            print(f"Reusing {manifest['num_suites']} verified suites in {root}")
            return root
        with tempfile.TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temporary:
            staging = Path(temporary) / "bundle"
            count = export_eval_sets(cfg, out_dir=staging)
            count += export_retention_position_sets(cfg, out_dir=staging)
            manifest = {
                "bundle_version": BUNDLE_VERSION,
                "generation_config": generation_config(cfg),
                "num_suites": count,
                "files": {p.name: _sha256(p) for p in sorted(staging.iterdir())},
            }
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
            backup = None
            if root.exists():
                backup = Path("outputs/eval-set-backups").absolute() / (
                    root.name + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                )
                if root.resolve() == backup.parent.resolve() or root.resolve() in backup.parents:
                    raise ValueError("Evaluation directory cannot contain its backup directory")
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(root), str(backup))
                print(f"Previous evaluation directory preserved at {backup}")
            try:
                staging.rename(root)
            except OSError:
                if backup is not None:
                    shutil.move(str(backup), str(root))
                raise
        print(f"Authoritative evaluation bundle: {root} ({count} suites)")
        return root
    finally:
        lock.rmdir()


def select_evaluation_suite(metadata: dict[str, Any], selection: str) -> bool:
    """Select scientific suites by purpose, without selecting different directories."""
    if selection not in {"all", "capabilities", "retention_position"}:
        raise ValueError("evaluation.suites must be all, capabilities or retention_position")
    capability = metadata.get("capability")
    return selection == "all" or (
        capability == "retention_position"
        if selection == "retention_position"
        else capability in {"icl", "composition", "retention"}
    )
