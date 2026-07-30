"""Export of frozen evaluation and analysis sets.

Each suite is written as one ``.npz`` of stacked arrays plus a ``meta.json``
recording the resolved config, git commit, and generation parameters, so
reported numbers and interp analyses always run on byte-identical sequences.
Worlds (module pools) are included as probe targets for the interp stage.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from iccl.data.sequences import SequenceSample


def _stack_info(samples: list[SequenceSample], key: str) -> np.ndarray:
    return np.stack([s.info[key] for s in samples])


def export_suite(samples: list[SequenceSample], path: Path, meta: dict[str, Any]) -> None:
    """Write a suite of equally-shaped sequences to ``<path>.npz`` + ``<path>.meta.json``."""
    lengths = {s.tokens.shape[0] for s in samples}
    if len(lengths) != 1:
        raise ValueError(f"suite sequences must share a length, got {sorted(lengths)}")
    arrays: dict[str, np.ndarray] = {
        "tokens": np.stack([s.tokens for s in samples]),
        "token_type": np.stack([s.token_type for s in samples]),
        "targets": np.stack([s.targets for s in samples]),
        "loss_mask": np.stack([s.loss_mask for s in samples]),
        "latents": _stack_info(samples, "latents"),
        "demo_counts": _stack_info(samples, "demo_counts"),
        "boundaries": _stack_info(samples, "boundaries"),
        "task_spans": _stack_info(samples, "task_spans"),
        "base_mse": _stack_info(samples, "base_mse"),
    }
    if "world" in samples[0].info:
        pools = [s.info["world"] for s in samples]
        for layer in range(len(pools[0].modules)):
            arrays[f"world_modules_{layer}"] = np.stack([p.modules[layer] for p in pools])
            arrays[f"world_biases_{layer}"] = np.stack([p.biases[layer] for p in pools])
        arrays["world_readout"] = np.stack([p.readout for p in pools])

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), allow_pickle=False, **arrays)
    meta = dict(meta, num_sequences=len(samples), git_commit=_git_commit())
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_suite(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.with_suffix(".npz")) as data:
        return dict(data)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
