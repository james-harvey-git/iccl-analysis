"""Fixed-world dataset creation with segmented, final-token capture and disposable queries."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from iccl.analysis.gdn_inference import continue_gdn
from iccl.analysis.probe_config import configure_runtime, resolved_config, validate_probe_config
from iccl.analysis.probe_execution import run_stage
from iccl.analysis.probe_features import feature_selections
from iccl.analysis.probe_storage import (
    file_digest,
    initialize_dataset,
    lock,
    read_json,
    read_manifest,
    save_npz,
    shard_path,
    validate_shard,
    write_json,
    write_shard,
)
from iccl.analysis.probe_tracking import WorkerEvents
from iccl.analysis.probe_worlds import CAPTURES, SPLITS, ProbeTrial, make_trial, make_world
from iccl.checkpoints import (
    checkpoint_model_config,
    checkpoint_model_digest,
    resolve_checkpoint_path,
    source_from_checkpoint,
)
from iccl.data.dataset import teacher_config_from
from iccl.data.sequences import TOKEN_X
from iccl.models.blocks import GDNBlock
from iccl.models.model import GDNModel, model_from_config
from iccl.models.ops import Backend, resolve_backend
from iccl.training.trainer import resolve_autocast_dtype


def representation_shapes(model: GDNModel) -> dict[str, list[int]]:
    blocks = cast(list[GDNBlock], list(model.blocks))
    mixer = blocks[0].mixer
    return {
        "residual_after_mixer": [len(blocks), model.head.in_features],
        "residual_after_mlp": [len(blocks), model.head.in_features],
        "mlp_hidden": [len(blocks), blocks[0].mlp.down_proj.in_features],
        "state": [len(blocks), mixer.n_heads, mixer.head_v_dim, mixer.head_k_dim],
        "head_read": [len(blocks), mixer.n_heads, mixer.head_v_dim],
    }


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    names = (
        "analysis/capture.py",
        "analysis/gdn_inference.py",
        "analysis/probe_worlds.py",
        "analysis/probe_features.py",
        "data/teacher.py",
        "data/sequences.py",
        "data/curriculum.py",
        "models/model.py",
        "models/blocks.py",
        "models/ops.py",
        "models/reference.py",
    )
    return {name: file_digest(package / name) for name in names}


@torch.inference_mode()
def capture_trials(
    model: GDNModel,
    trials: list[ProbeTrial],
    *,
    backend: Backend,
    device: torch.device,
    dtype: torch.dtype | None,
    families: list[str],
) -> dict[str, np.ndarray]:
    """All points share one prefix; the fresh query receives a complete cloned cache."""
    stop = trials[0].answer_stop
    if any(trial.answer_stop != stop for trial in trials):
        raise ValueError("a capture batch must share target position and task length")
    tokens = torch.from_numpy(np.stack([trial.sample.tokens[:stop] for trial in trials])).to(device)
    types = torch.from_numpy(np.stack([trial.sample.token_type[:stop] for trial in trials])).to(
        device
    )
    output: dict[str, np.ndarray] = {}
    features: dict[str, torch.Tensor] = {}
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
        predictions, cache = continue_gdn(
            model, tokens[:, :-1], types[:, :-1], backend=backend, capture=features
        )
        for family in families:
            output[f"final_input__{family}"] = features[family].float().cpu().numpy()
        output["final_input__prediction"] = predictions[:, -1].float().cpu().numpy()
        features = {}
        _, cache = continue_gdn(
            model, tokens[:, -1:], types[:, -1:], cache, backend=backend, capture=features
        )
        for family in families:
            output[f"final_answer__{family}"] = features[family].float().cpu().numpy()
        queries = torch.zeros(len(trials), 1, tokens.shape[-1], device=device)
        query_array = np.stack([trial.branch_input for trial in trials])
        queries[:, :, : query_array.shape[-1]] = torch.from_numpy(query_array).to(device)
        query_types = torch.full((len(trials), 1), TOKEN_X, device=device, dtype=torch.long)
        features = {}
        copied = cache.select(torch.arange(len(trials), device=device))
        predictions, _ = continue_gdn(
            model, queries, query_types, copied, backend=backend, capture=features
        )
        for family in families:
            output[f"branch_input__{family}"] = features[family].float().cpu().numpy()
        output["branch_input__prediction"] = predictions[:, -1].float().cpu().numpy()
    return output


def _trial_arrays(
    trials: list[ProbeTrial], input_dim: int, output_dim: int, observations: bool
) -> dict[str, np.ndarray]:
    arrays = {
        "target": np.stack([t.target for t in trials]),
        "support": np.stack([t.support for t in trials]),
        "exposure": np.stack([t.prior_exposures for t in trials]),
        "latents": np.stack([t.sample.info["latents"] for t in trials]),
        "trial": np.asarray([t.index for t in trials], dtype=np.int64),
        "ordinary_answer": np.stack(
            [t.sample.tokens[t.answer_stop - 1, :output_dim] for t in trials]
        ),
        "branch_answer": np.concatenate([t.branch_answer for t in trials]),
        "target_variance": np.stack(
            [
                t.sample.tokens[t.task_start + 1 : t.answer_stop : 2, :output_dim].var(axis=0)
                for t in trials
            ]
        ),
    }
    if observations:
        for capture in CAPTURES:
            arrays[f"observations__{capture}"] = np.stack(
                [t.observations(capture, input_dim, output_dim) for t in trials]
            )
    return arrays


@torch.inference_mode()
def capture_dataset(cfg: DictConfig, out_dir: Path | str) -> dict[str, Any]:
    validate_probe_config(cfg, "capture")
    device = configure_runtime(cfg)
    d = cfg.probe.dataset
    checkpoint_path, _ = resolve_checkpoint_path(str(d.checkpoint))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = checkpoint_model_config(checkpoint)
    source = source_from_checkpoint(checkpoint)
    teacher_data = cast(DictConfig, OmegaConf.create(checkpoint["config"]["data"]))
    teacher_data.num_modules = int(d.modules_per_world)
    teacher_data.weighting = d.weighting
    teacher = teacher_config_from(teacher_data, num_modules=d.modules_per_world)
    signal_boundaries = bool(teacher_data.sequence.signal_boundaries)
    backend = resolve_backend(cast(Backend, d.capture.backend), device)
    dtype = resolve_autocast_dtype(d.capture.precision, device)
    model_cfg = OmegaConf.create(architecture)
    model_cfg.model.backend = backend
    models = {}
    for name in d.capture.models:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(d.capture.untrained_seed))
            model = model_from_config(model_cfg)
        if name == "trained":
            model.load_state_dict(checkpoint["model"])
        models[name] = model.to(device).eval().requires_grad_(False)
    shapes = representation_shapes(models["trained"])
    specifications = cast(dict[str, Any], OmegaConf.to_container(d.capture.representations))
    head_groups = resolved_config(cfg)["probe"]["dataset"]["capture"]["head_groups"]
    feature_selections(shapes, specifications, head_groups)
    families = list(specifications)
    dataset_spec = {
        key: resolved_config(cfg)["probe"]["dataset"][key]
        for key in (
            "num_worlds",
            "modules_per_world",
            "active_modules_per_task",
            "tasks_per_episode",
            "demos_per_task",
            "weighting",
            "target_positions",
            "constituent_pair_split",
            "trajectories",
            "shard_size",
        )
    }
    identity = {
        "phase": cfg.experiment.phase,
        "seed": int(cfg.experiment.seed),
        "dataset": dataset_spec,
        "teacher": asdict(teacher),
        "signal_boundaries": signal_boundaries,
        "source_model_digest": checkpoint_model_digest(checkpoint),
        "architecture": architecture,
        "implementation": _implementation(),
        "capture": {
            "models": list(d.capture.models),
            "untrained_seed": int(d.capture.untrained_seed),
            "representations": specifications,
            "head_groups": head_groups,
            "shapes": {key: shapes[key] for key in families},
            "points": list(CAPTURES),
            "observations": bool(d.capture.store_observations),
            "backend": backend,
            "precision": str(dtype or torch.float32),
            "storage_dtype": "float32",
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
    root = Path(d.path)
    manifest = initialize_dataset(
        root,
        identity,
        {
            "checkpoint_reference": str(d.checkpoint),
            "checkpoint_file_sha256": file_digest(checkpoint_path),
            "source_run": None if source is None else asdict(source),
            "source_step": int(checkpoint.get("step", -1)),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
            ).strip(),
            "config": resolved_config(cfg),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    )
    dataset_id = manifest["dataset_id"]
    feature_bytes = 4 * sum(int(np.prod(shapes[key])) for key in families)
    total_trials = (
        sum(dataset_spec["trajectories"].values()) * d.num_worlds * len(d.target_positions)
    )
    estimated = total_trials * feature_bytes * len(CAPTURES) * len(models)
    if int(os.environ.get("SLURM_PROCID", 0)) == 0:
        print(
            f"capture feature storage lower bound: {estimated / 2**30:.3f} GiB; "
            f"{total_trials} independent trials",
            flush=True,
        )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def worker(world_id: int, events: WorkerEvents) -> None:
        folder = root / f"world-{world_id:04d}"
        with lock(folder / ".capture.lock", blocking=False):
            world = make_world(
                teacher,
                seed=cfg.experiment.seed,
                phase=cfg.experiment.phase,
                world_id=world_id,
                pair_counts=dict(d.constituent_pair_split),
            )
            world_meta = world.metadata()
            pool_arrays = {"readout": world.pool.readout}
            pool_arrays.update({f"modules_{i}": x for i, x in enumerate(world.pool.modules)})
            pool_arrays.update({f"biases_{i}": x for i, x in enumerate(world.pool.biases)})
            if (folder / "world.json").exists():
                if read_json(folder / "world.json") != json_normalized(world_meta):
                    raise ValueError("saved teacher world differs from the requested identity")
            else:
                write_json(folder / "world.json", world_meta)
            if (folder / "world.npz").exists():
                with np.load(folder / "world.npz", allow_pickle=False) as saved_pool:
                    if set(saved_pool.files) != set(pool_arrays) or any(
                        not np.array_equal(saved_pool[key], value)
                        for key, value in pool_arrays.items()
                    ):
                        raise ValueError("saved teacher arrays differ from the deterministic world")
            else:
                save_npz(folder / "world.npz", pool_arrays)
            splits, coverage = {}, []
            started = time.perf_counter()
            new_trials = 0
            for position in d.target_positions:
                for split in SPLITS:
                    entries, all_targets, all_exposures = [], [], []
                    count = int(d.trajectories[split])
                    for start in range(0, count, d.shard_size):
                        stop = min(start + d.shard_size, count)
                        path = shard_path(root, world_id, position, split, start)
                        context = {
                            "dataset_id": dataset_id,
                            "world": world_id,
                            "position": position,
                            "split": split,
                            "start": start,
                            "stop": stop,
                        }
                        if path.exists():
                            if not d.capture.resume:
                                raise ValueError("capture exists and resume is disabled")
                            entry = validate_shard(path, dataset_id)
                            if any(entry[key] != value for key, value in context.items()):
                                raise ValueError(
                                    "shard belongs to a different world/position/population"
                                )
                        else:
                            batches: dict[str, list[np.ndarray]] = {}
                            for offset in range(start, stop, d.capture.batch_size):
                                trials = [
                                    make_trial(
                                        world,
                                        split=split,
                                        position=position,
                                        index=index,
                                        tasks=d.tasks_per_episode,
                                        demos=d.demos_per_task,
                                        signal_boundaries=signal_boundaries,
                                    )
                                    for index in range(
                                        offset, min(offset + d.capture.batch_size, stop)
                                    )
                                ]
                                arrays = _trial_arrays(
                                    trials,
                                    teacher.input_dim,
                                    teacher.output_dim,
                                    d.capture.store_observations,
                                )
                                for model_name, model in models.items():
                                    arrays.update(
                                        {
                                            f"{model_name}__{key}": value
                                            for key, value in capture_trials(
                                                model,
                                                trials,
                                                backend=backend,
                                                device=device,
                                                dtype=dtype,
                                                families=families,
                                            ).items()
                                        }
                                    )
                                for name, value in arrays.items():
                                    batches.setdefault(name, []).append(value)
                            arrays = {
                                name: np.concatenate(value) for name, value in batches.items()
                            }
                            entry = write_shard(
                                path,
                                arrays,
                                context
                                | {
                                    "batch_size": int(d.capture.batch_size),
                                    "device": str(device),
                                },
                            )
                            new_trials += stop - start
                            events.emit(
                                "capture_shard",
                                world=world_id,
                                position=position,
                                split=split,
                                trials=stop - start,
                                seconds=time.perf_counter() - started,
                                gpu_memory_bytes=torch.cuda.max_memory_allocated(device)
                                if device.type == "cuda"
                                else 0,
                            )
                        entries.append(entry)
                        all_targets.append(np.load(path / "target.npy", allow_pickle=False))
                        all_exposures.append(np.load(path / "exposure.npy", allow_pickle=False))
                    splits[f"{position}/{split}"] = entries
                    targets, exposures = (
                        np.concatenate(all_targets),
                        np.sort(np.concatenate(all_exposures), axis=1),
                    )
                    if (targets.var(axis=0) <= 0).any():
                        raise ValueError(
                            f"constant target coordinate in world {world_id}/{position}/{split}"
                        )
                    if (
                        split == "train"
                        and np.linalg.matrix_rank(targets - targets.mean(axis=0))
                        < teacher.num_modules
                    ):
                        raise ValueError(
                            "training targets lack full affine variation; increase the "
                            "trajectory budget"
                        )
                    for pair in np.unique(exposures, axis=0):
                        selected = np.all(exposures == pair, axis=1)
                        z = targets[selected]
                        coverage.append(
                            {
                                "world": world_id,
                                "position": position,
                                "split": split,
                                "e_min": int(pair[0]),
                                "e_max": int(pair[1]),
                                "trials": int(selected.sum()),
                                "target_variances": z.var(axis=0).tolist(),
                                "target_pairs": np.unique(
                                    np.stack([np.flatnonzero(row) for row in z]), axis=0
                                ).tolist(),
                            }
                        )
            complete = {
                "dataset_id": dataset_id,
                "world": world_id,
                "splits": splits,
                "world_sha256": file_digest(folder / "world.npz"),
                "world_metadata_sha256": file_digest(folder / "world.json"),
                "coverage": coverage,
            }
            write_json(folder / "complete.json", complete)
            events.emit(
                "capture_complete",
                world=world_id,
                trials=new_trials,
                seconds=time.perf_counter() - started,
            )

    def finalize() -> dict[str, Any]:
        with lock(root / ".manifest.lock"):
            result = read_manifest(root, complete=False)
            result["worlds"] = {
                str(w): file_digest(root / f"world-{w:04d}" / "complete.json")
                for w in range(d.num_worlds)
            }
            result["complete"] = True
            result["estimated_feature_bytes"] = estimated
            write_json(root / "manifest.json", result)
        return {
            "dataset_path": str(root.resolve()),
            "dataset_id": dataset_id,
            "trials": total_trials,
        }

    return run_stage(cfg, "capture", worker, finalize)


def json_normalized(value: Any) -> Any:
    import json

    return json.loads(json.dumps(value))
