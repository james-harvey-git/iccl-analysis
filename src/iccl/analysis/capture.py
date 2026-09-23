"""Checkpoint-driven capture of terminal fast-weight states on independent episodes."""

from __future__ import annotations

import importlib.metadata
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from iccl.analysis.probe_config import (
    SPLITS,
    configure_runtime,
    resolved_config,
    stream_seed,
    validate_probe_config,
)
from iccl.analysis.probe_dataset import DatasetWriter, array_schema, file_digest, write_json
from iccl.analysis.probe_targets import PROTOCOL, episode_targets
from iccl.checkpoints import (
    WANDB_SCHEME,
    checkpoint_model_config,
    checkpoint_model_digest,
    resolve_checkpoint_path,
    source_from_checkpoint,
)
from iccl.data.dataset import sequence_dataset_from_config
from iccl.data.sequences import TOKEN_BOUNDARY, TOKEN_Y, SequenceSample
from iccl.models.blocks import GDNBlock
from iccl.models.model import GDNModel, model_from_config
from iccl.models.ops import Backend, resolve_backend
from iccl.reporting.logger import RunLogger
from iccl.training.trainer import resolve_autocast_dtype


def append_terminal_boundary(sample: SequenceSample) -> tuple[np.ndarray, np.ndarray]:
    """Append one ordinary boundary without mutating production sequence serialization."""
    if len(sample.tokens) != 520 or sample.token_type[-1] != TOKEN_Y:
        raise ValueError("expected eight 32-demo tasks ending with a y-token")
    if np.count_nonzero(sample.token_type == TOKEN_BOUNDARY) != 8:
        raise ValueError("expected exactly eight task-start boundaries")
    tokens = np.concatenate((sample.tokens, np.zeros_like(sample.tokens[:1])), axis=0)
    types = np.concatenate(
        (sample.token_type, np.asarray([TOKEN_BOUNDARY], sample.token_type.dtype))
    )
    return tokens, types


class CaptureEpisodes(Dataset):
    """Map-style CPU generation preserves indexed world identity across worker counts."""

    def __init__(self, data: DictConfig, seed: int, start: int, stop: int) -> None:
        self.stream = sequence_dataset_from_config(data, base_seed=seed)
        self.start, self.stop = start, stop

    def __len__(self) -> int:
        return self.stop - self.start

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        index += self.start
        sample = self.stream.build(index, include_world=True)
        tokens, types = append_terminal_boundary(sample)
        return {
            **episode_targets(sample.info["world"], sample.info["latents"]),
            "episode_index": np.asarray(index, dtype=np.int64),
            "tokens": tokens,
            "token_type": types,
        }


def collate_capture(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {name: np.stack([row[name] for row in rows]) for name in rows[0]}


def state_layout(model: GDNModel) -> dict[str, Any]:
    shapes = [
        [block.mixer.n_heads, block.mixer.head_v_dim, block.mixer.head_k_dim]
        for block in cast(list[GDNBlock], list(model.blocks))
    ]
    if not shapes:
        raise ValueError("capture requires at least one GDN layer")
    return {
        "order": ["layer", "head", "value", "key"],
        "layer_shapes": shapes,
        "input_features": sum(int(np.prod(shape)) for shape in shapes),
    }


def flatten_final_states(states: list[torch.Tensor], layout: dict[str, Any]) -> torch.Tensor:
    if len(states) != len(layout["layer_shapes"]):
        raise ValueError("captured layer count does not match the checkpoint")
    if any(
        list(value.shape[1:]) != shape
        for value, shape in zip(states, layout["layer_shapes"], strict=True)
    ):
        raise ValueError("captured state axes do not match the checkpoint layout")
    return torch.cat([value.reshape(value.shape[0], -1) for value in states], dim=1).float()


def _implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    files = [
        "analysis/capture.py",
        "analysis/probe_targets.py",
        "analysis/probe_config.py",
        "data/dataset.py",
        "data/sequences.py",
        "data/curriculum.py",
        "data/teacher.py",
        "models/model.py",
        "models/blocks.py",
        "models/ops.py",
        "models/reference.py",
    ]
    return {name: file_digest(package / name) for name in files}


@torch.inference_mode()
def capture_dataset(cfg: DictConfig, out_dir: Path | str) -> dict[str, Any]:
    """Generate or safely resume immutable final-state shards for one frozen checkpoint."""
    started = time.perf_counter()
    validate_probe_config(cfg, "capture")
    device = configure_runtime(cfg)
    p = cfg.probe
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path, artifact = resolve_checkpoint_path(str(p.capture.checkpoint))
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    architecture = checkpoint_model_config(checkpoint)
    if architecture["data"] != {"input_dim": 16, "output_dim": 16}:
        raise ValueError("the source GDN checkpoint must use 16-dimensional inputs and outputs")
    backend = resolve_backend(cast(Backend, p.capture.backend), device)
    if backend == "fla" and device.type != "cuda":
        raise ValueError("FLA capture requires a CUDA device")
    model_cfg = OmegaConf.create(architecture)
    model_cfg.model.backend = backend
    model = model_from_config(model_cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval().requires_grad_(False)
    layout = state_layout(model)
    dtype = resolve_autocast_dtype(p.capture.precision, device)
    source = source_from_checkpoint(checkpoint)
    model_digest = checkpoint_model_digest(checkpoint)
    seeds = {split: stream_seed(p.dataset.seed, f"episodes/{split}") for split in SPLITS}
    if len(set(seeds.values())) != len(SPLITS):
        raise ValueError("split-seed collision; choose a different dataset seed")
    known_seed = checkpoint["config"].get("seed")
    if known_seed is not None and any(
        int(known_seed) <= seed <= int(known_seed) + 2**32 + 5_000_000 for seed in seeds.values()
    ):
        raise ValueError("probe split seed overlaps a known source data stream")
    identity = {
        "protocol": PROTOCOL,
        "source_model_digest": model_digest,
        "architecture": architecture,
        "source_step": int(checkpoint["step"]),
        "data": {
            key: resolved_config(cfg)["data"][key]
            for key in (
                "input_dim",
                "output_dim",
                "hidden_dims",
                "use_bias",
                "num_modules",
                "scale",
                "weighting",
                "sequence",
            )
        },
        "dataset_seed": int(p.dataset.seed),
        "split_seeds": seeds,
        "state_layout": layout,
        "input_features": layout["input_features"],
        "token_count": 521,
        "storage_dtype": "float32",
        "state_compute_dtype": "float32",
        "canonicalization": "unit-readout-row-norm-f64-to-f32-v1",
        "backend": backend,
        "precision": str(dtype or torch.float32),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "capture_batch_size": int(p.capture.batch_size),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "fla_core": importlib.metadata.version("fla-core") if backend == "fla" else None,
        "implementation": _implementation_hashes(),
    }
    provenance = {
        "checkpoint_reference": str(p.capture.checkpoint),
        "checkpoint_path": str(source_path.resolve()),
        "checkpoint_file_sha256": file_digest(source_path),
        "source_run": None if source is None else asdict(source),
        "known_source_seed": known_seed,
        "resolved_config": resolved_config(cfg),
    }
    load_seconds = time.perf_counter() - started
    counts = {split: int(p.dataset.counts[split]) for split in SPLITS}
    schema = array_schema(layout["input_features"])
    bytes_per_episode = sum(
        int(np.prod(value["shape"])) * np.dtype(value["dtype"]).itemsize
        for value in schema.values()
    )
    estimated_gib = sum(counts.values()) * bytes_per_episode / 2**30
    print(
        f"capture: {layout['input_features']:,} features; estimated arrays {estimated_gib:.3f} GiB"
    )
    logger = RunLogger(cfg, out_dir, job_type="probe-capture", source=source, protocol=PROTOCOL)
    generated, forward_seconds = 0, 0.0
    try:
        with DatasetWriter(
            p.dataset.path, identity, counts, p.dataset.shard_size, resume=p.capture.resume
        ) as writer:
            writer.record_provenance(provenance)
            logger.start()
            if artifact:
                logger.use_artifact(str(p.capture.checkpoint).removeprefix(WANDB_SCHEME))
            for split in SPLITS:
                episodes = CaptureEpisodes(
                    cfg.data, seeds[split], writer.completed(split), counts[split]
                )
                if len(episodes) == 0:
                    continue
                loader = DataLoader(
                    episodes,
                    batch_size=p.capture.batch_size,
                    num_workers=p.capture.num_workers,
                    collate_fn=collate_capture,
                    multiprocessing_context="spawn" if p.capture.num_workers else None,
                )
                buffered = 0
                buffers: dict[str, list[np.ndarray]] = {name: [] for name in schema}
                for batch in loader:
                    tick = time.perf_counter()
                    tokens = torch.from_numpy(batch.pop("tokens")).to(device)
                    types = torch.from_numpy(batch.pop("token_type")).to(device)
                    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                        output = model(tokens, types, capture_final=True)
                    assert output.final_states is not None
                    if any(value.dtype != torch.float32 for value in output.final_states):
                        raise ValueError("unexpected recurrence-state compute dtype")
                    batch["states"] = (
                        flatten_final_states(output.final_states, layout).cpu().numpy()
                    )
                    forward_seconds += time.perf_counter() - tick
                    generated += len(tokens)
                    offset = 0
                    while offset < len(tokens):
                        take = min(p.dataset.shard_size - buffered, len(tokens) - offset)
                        for name in schema:
                            buffers[name].append(batch[name][offset : offset + take])
                        offset += take
                        buffered += take
                        if buffered == p.dataset.shard_size:
                            writer.append(
                                split,
                                {name: np.concatenate(parts) for name, parts in buffers.items()},
                            )
                            buffers = {name: [] for name in schema}
                            buffered = 0
                            logger.log({"capture/new_episodes": float(generated)}, generated)
                if buffered:
                    writer.append(
                        split, {name: np.concatenate(parts) for name, parts in buffers.items()}
                    )
            dataset_id = writer.manifest["dataset_id"]
        report = {
            "dataset_path": str(Path(p.dataset.path).resolve()),
            "dataset_id": dataset_id,
            "new_episodes": generated,
            "checkpoint_load_seconds": load_seconds,
            "forward_and_transfer_seconds": forward_seconds,
            "total_seconds": time.perf_counter() - started,
            "estimated_array_bytes": sum(counts.values()) * bytes_per_episode,
            "timing_scope": (
                "total includes checkpoint resolution, validation, generation, inference and writes"
            ),
        }
        write_json(out_dir / "capture.json", report)
        return report
    finally:
        logger.finish()
