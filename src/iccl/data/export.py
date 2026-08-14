"""Export of frozen evaluation and analysis sets.

Each suite is written as one ``.npz`` of stacked arrays plus a ``meta.json``
recording the resolved config, git commit, and generation parameters, so
reported numbers and interp analyses always run on byte-identical sequences.
Worlds (module pools) are included as probe targets for the interp stage.

The suites: "in_dist" (standard curriculum sequences), "composite" (curriculum
plus a few-shot composite final task) with its paired "composite_control" (same
world and final task, no history), one "structural_<graph>" suite per
overlap-graph family in eval_sets.structural_graphs, "retention" (curriculum
sequences that re-demonstrate the first task at the end), and one
position-matched control per eval_sets.retention.controls mode, each repeating
its retention sequence with a different task in that final block.

Suites take consecutive blocks of the eval sequence-index space, so appending a
suite leaves the earlier ones' sequences untouched.
"""

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from iccl.data.dataset import make_family, sequence_config_from, sequence_rng
from iccl.data.sequences import (
    FinalTaskConfig,
    SequenceSample,
    build_paired_control,
    build_paired_retention_control,
    build_sequence,
)

EVAL_SEED_OFFSET = 1_000_000  # keeps eval streams disjoint from training indices

# Suite name per retention-control mode, shared by the exporter and the metrics
# so the two agree on what is on disk.
RETENTION_CONTROL_SUITES = {"novel": "retention_control", "shared": "retention_control_shared"}


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
        "num_curriculum_tasks": _stack_info(samples, "num_curriculum_tasks"),
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


def export_eval_sets(cfg: DictConfig) -> int:
    """Write every frozen eval suite to ``cfg.data.eval_sets.out_dir``.

    Returns the number of suites written. The shared-module retention control is
    skipped under a binary weighting, where a task is its support and the
    control would duplicate the retention suite.
    """
    data_cfg = cfg.data
    composite_cfg = data_cfg.eval_sets.composite
    family = make_family(data_cfg, extra_hotness=composite_cfg.hotness)
    seq_cfg = sequence_config_from(data_cfg)
    final = FinalTaskConfig(
        mode="composite", hotness=composite_cfg.hotness, num_demos=composite_cfg.num_demos
    )
    out_dir = Path(data_cfg.eval_sets.out_dir)
    num = data_cfg.eval_sets.num_sequences
    meta = {"config": OmegaConf.to_container(data_cfg, resolve=True), "seed": cfg.seed}
    base_seed = cfg.seed + EVAL_SEED_OFFSET

    in_dist, composite, control = [], [], []
    for i in range(num):
        rng = sequence_rng(base_seed, i)
        in_dist.append(build_sequence(family, seq_cfg, rng, include_world=True))
        rng = sequence_rng(base_seed, num + i)
        seq = build_sequence(family, seq_cfg, rng, final_task=final, include_world=True)
        composite.append(seq)
        control.append(build_paired_control(family, seq_cfg, seq, final, rng))

    export_suite(in_dist, out_dir / "in_dist", dict(meta, suite="in_dist"))
    export_suite(composite, out_dir / "composite", dict(meta, suite="composite"))
    export_suite(control, out_dir / "composite_control", dict(meta, suite="composite_control"))

    num_suites = 3
    for graph_offset, graph in enumerate(data_cfg.eval_sets.structural_graphs):
        graph_cfg = replace(seq_cfg, task_graph=graph)
        suite = [
            build_sequence(
                family,
                graph_cfg,
                sequence_rng(base_seed, (2 + graph_offset) * num + i),
                include_world=True,
            )
            for i in range(num)
        ]
        export_suite(
            suite, out_dir / f"structural_{graph}", dict(meta, suite=f"structural_{graph}")
        )
        num_suites += 1

    retention_base = (2 + len(data_cfg.eval_sets.structural_graphs)) * num
    retention = [
        build_sequence(
            family,
            seq_cfg,
            sequence_rng(base_seed, retention_base + i),
            revisit_demos=data_cfg.eval_sets.retention.revisit_demos,
            include_world=True,
        )
        for i in range(num)
    ]
    export_suite(retention, out_dir / "retention", dict(meta, suite="retention"))
    num_suites += 1

    modes = list(data_cfg.eval_sets.retention.controls)
    if "shared" in modes and data_cfg.weighting == "binary":
        print(
            "skipping the shared-module retention control: under weighting=binary a task is "
            "its support, so the control would duplicate the retention suite"
        )
        modes.remove("shared")
    for offset, mode in enumerate(modes):
        base = retention_base + (1 + offset) * num
        suite = [
            build_paired_retention_control(
                family, retention[i], sequence_rng(base_seed, base + i), mode=mode
            )
            for i in range(num)
        ]
        name = RETENTION_CONTROL_SUITES[mode]
        suite_meta: dict[str, Any] = dict(meta, suite=name, control_mode=mode)
        # The shared control reuses the revisited task's modules with different
        # weights; how different bounds how much episodic help leaks into it, so
        # the frozen set records the spread.
        if mode == "shared":
            distances = [
                float(np.linalg.norm(c.info["latents"][-1] - r.info["latents"][-1]))
                for c, r in zip(suite, retention, strict=True)
            ]
            suite_meta["latent_distance_to_revisited"] = {
                "mean": float(np.mean(distances)),
                "min": float(np.min(distances)),
            }
        export_suite(suite, out_dir / name, suite_meta)
        num_suites += 1

    print(f"exported {num_suites} suites x {num} sequences to {out_dir}/")
    return num_suites


def load_suite(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.with_suffix(".npz")) as data:
        return dict(data)


def suite_paths(
    eval_dir: Path,
    suite_name: str,
    explicit_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve and validate the array and metadata paths for one frozen suite."""
    suite_path = Path(explicit_path) if explicit_path is not None else eval_dir / suite_name
    suite_path = suite_path.with_suffix(".npz")
    metadata_path = suite_path.with_suffix(".meta.json")
    if not suite_path.exists():
        raise FileNotFoundError(f"frozen suite not found: {suite_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"frozen suite metadata not found: {metadata_path}")
    return suite_path, metadata_path


def load_suite_metadata(path: Path) -> dict[str, Any]:
    """Load a frozen suite's JSON-object metadata sidecar."""
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"suite metadata must be a JSON object: {path}")
    return metadata


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
