"""Full affine probe optimization, detached exact matching and resumable training."""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from iccl.analysis.probe_config import (
    configure_runtime,
    resolved_config,
    stream_seed,
    validate_probe_config,
)
from iccl.analysis.probe_dataset import (
    CapturedDataset,
    EpisodeBatchSampler,
    shuffled_pairing,
    write_json,
)
from iccl.analysis.probe_loss import aligned_targets, parameter_errors
from iccl.analysis.probe_matching import Assignment, AssignmentSolver
from iccl.analysis.probe_results import (
    CHECKPOINT_VERSION,
    TARGET_LAYOUT,
    load_checkpoint,
    save_checkpoint,
    source_run,
)
from iccl.analysis.probe_targets import PROTOCOL
from iccl.analysis.probes import make_decoder
from iccl.reporting.logger import RunLogger
from iccl.training.trainer import build_optimizer, build_scheduler, resolve_autocast_dtype


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def probe_loader(
    dataset: CapturedDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
    *,
    sampler: EpisodeBatchSampler | None = None,
    seed: int = 0,
) -> DataLoader:
    """Spawn workers independently of the native pool; prefetch never owns the resume cursor."""
    return DataLoader(
        dataset,
        batch_size=1 if sampler is not None else batch_size,
        batch_sampler=sampler,
        num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(stream_seed(seed, "loader-workers")),
    )


@dataclass
class UpdateResult:
    loss: float
    grad_norm: float
    episodes: int
    assignment: Assignment
    stages: dict[str, float]
    episode_indices: list[int]


def optimizer_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: dict[str, torch.Tensor],
    solver: AssignmentSolver,
    device: torch.device,
    dtype: torch.dtype | None,
    readout_weight: float,
    grad_clip: float | None,
    *,
    profile: bool = False,
) -> UpdateResult:
    """The shared training/benchmark update; only the detached output crosses back to CPU."""
    stages: dict[str, float] = {}
    if profile:
        synchronize(device)
    tick = time.perf_counter()

    def mark(name: str) -> None:
        nonlocal tick
        if profile:
            synchronize(device)
            now = time.perf_counter()
            stages[name] = now - tick
            tick = now

    optimizer.zero_grad(set_to_none=True)
    states = batch["states"].to(device, non_blocking=True)
    targets = batch["target"].to(device, non_blocking=True)
    mark("host_to_device")
    with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
        predictions = model(states)
    predictions = predictions.float()
    mark("decoder_forward")
    detached = predictions.detach().cpu().numpy()
    mark("prediction_to_cpu")
    assignment = solver.match(detached, batch["target"].numpy(), readout_weight, profile=profile)
    mark("native_matching_wall")
    loss = parameter_errors(
        predictions, aligned_targets(targets, assignment), readout_weight
    ).joint.mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("nonfinite aligned probe loss")
    mark("assignment_return_and_loss")
    loss.backward()
    mark("backward")
    # This verifies every gradient and optionally clips it before any parameter update.
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        float("inf") if grad_clip is None else grad_clip,
        error_if_nonfinite=True,
    )
    mark("gradient_check_and_clip")
    optimizer.step()
    scheduler.step()
    mark("optimizer_and_scheduler")
    return UpdateResult(
        float(loss.detach()),
        float(norm),
        len(states),
        assignment,
        stages,
        batch["episode_index"].tolist(),
    )


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"])


class ProbeTrainer:
    """Own one decoder and native pool; validation and testing never share a sampler."""

    def __init__(self, cfg: DictConfig, out_dir: Path | str, *, stage: str = "train") -> None:
        validate_probe_config(cfg, stage)
        self.cfg, self.p = cfg, cfg.probe
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if (
            stage == "train"
            and not self.p.training.resume
            and (self.out_dir / "checkpoints").exists()
        ):
            raise FileExistsError("choose a fresh training run directory or explicitly resume")
        self.device = configure_runtime(cfg)
        self.dtype = resolve_autocast_dtype(self.p.training.precision, self.device)
        self.timings: dict[str, float] = {}
        tick = time.perf_counter()
        self.train = CapturedDataset(self.p.dataset.path, "train")
        self.validation = CapturedDataset(self.p.dataset.path, "validation")
        self.timings["dataset_validation"] = time.perf_counter() - tick
        self.control = str(self.p.training.control)
        self.mapping = (
            shuffled_pairing(len(self.train), cfg.seed)
            if self.control == "shuffled_targets"
            else None
        )
        self.train.target_permutation = self.mapping
        features = self.train.manifest["identity"]["input_features"]
        tick = time.perf_counter()
        self.model = make_decoder(features, self.control).to(self.device)
        self.optimizer = build_optimizer(self.model, self.p.training)
        self.scheduler = build_scheduler(self.optimizer, self.p.training)
        self.timings["model_optimizer_initialization"] = time.perf_counter() - tick
        tick = time.perf_counter()
        self.solver = AssignmentSolver(self.p.solver.num_threads, self.p.solver.cache_dir)
        self.timings["native_build_load_pool"] = time.perf_counter() - tick
        self.step, self.consumed = 0, 0
        self.best_loss: float | None = None
        self.best_step: int | None = None
        self.history: list[dict[str, Any]] = []
        self.validation_history: list[dict[str, Any]] = []
        self.contract = {
            "seed": int(cfg.seed),
            "control": self.control,
            "batch_size": int(self.p.training.batch_size),
            "optimizer": str(self.p.training.optimizer),
            "lr": float(self.p.training.lr),
            "weight_decay": float(self.p.training.weight_decay),
            "schedule": str(self.p.training.schedule),
            "warmup_steps": int(self.p.training.warmup_steps),
            "num_steps": int(self.p.training.num_steps),
            "grad_clip": self.p.training.grad_clip,
            "readout_weight": float(self.p.loss.readout_weight),
            "dtype": str(self.dtype or torch.float32),
            "device_type": self.device.type,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "solver_source": self.solver.identity["source_sha256"],
            "validation_every": int(self.p.training.validation_every),
        }
        self.logger = RunLogger(
            cfg,
            self.out_dir,
            job_type=f"probe-{stage}",
            source=source_run(self.train.manifest),
            protocol=PROTOCOL,
        )
        self._iterator: Iterator[dict[str, torch.Tensor]] | None = None
        if self.p.training.resume:
            tick = time.perf_counter()
            self._resume(Path(self.p.training.resume))
            self.timings["checkpoint_load"] = time.perf_counter() - tick
        self.loader = probe_loader(
            self.train,
            self.p.training.batch_size,
            self.p.training.num_workers,
            self.device,
            sampler=EpisodeBatchSampler(
                len(self.train), self.p.training.batch_size, cfg.seed, self.consumed
            ),
            seed=cfg.seed,
        )
        self.val_loader = probe_loader(
            self.validation,
            self.p.training.batch_size,
            self.p.training.num_workers,
            self.device,
            seed=cfg.seed,
        )
        parameters = sum(value.numel() for value in self.model.parameters())
        print(
            f"probe: {parameters:,} parameters; {features:,} inputs; "
            f"device={self.device}; control={self.control}"
        )

    def _resume(self, path: Path) -> None:
        state = load_checkpoint(path, self.train)
        if state.get("kind") != "resumable" or state.get("training_contract") != self.contract:
            raise ValueError(
                "resume requires the same training/numerical settings and a full-state checkpoint"
            )
        if state["split_signatures"]["validation"] != self.validation.signature:
            raise ValueError("validation population changed since the checkpoint")
        saved_mapping = state["target_permutation"]
        if (saved_mapping is None) != (self.mapping is None) or (
            self.mapping is not None and not np.array_equal(saved_mapping, self.mapping)
        ):
            raise ValueError("shuffled-target pairing differs from the checkpoint")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step, self.consumed = int(state["step"]), int(state["samples_consumed"])
        self.best_loss, self.best_step = state["best_loss"], state["best_step"]
        self.history, self.validation_history = state["history"], state["validation_history"]
        self._previous_best = state.get("best_checkpoint") or str(path.parent / "best.pt")
        _restore_rng(state["rng"])

    def next_batch(self) -> dict[str, torch.Tensor]:
        if self._iterator is None:
            self._iterator = iter(self.loader)
        return next(self._iterator)

    def update(self, batch: dict[str, torch.Tensor], *, profile: bool = False) -> UpdateResult:
        self.model.train()
        try:
            result = optimizer_update(
                self.model,
                self.optimizer,
                self.scheduler,
                batch,
                self.solver,
                self.device,
                self.dtype,
                self.p.loss.readout_weight,
                self.p.training.grad_clip,
                profile=profile,
            )
        except torch.OutOfMemoryError as error:
            raise RuntimeError(
                "The full decoder exhausted device memory. "
                "Reduce probe.training.batch_size explicitly "
                "or use a larger GPU; the decoder is never replaced by a smaller map."
            ) from error
        self.step += 1
        self.consumed += result.episodes
        return result

    @torch.inference_mode()
    def validate(self) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.val_loader:
            states = batch["states"].to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype is not None):
                predictions = self.model(states)
            predictions = predictions.float()
            assignment = self.solver.match(
                predictions.cpu().numpy(), batch["target"].numpy(), self.p.loss.readout_weight
            )
            targets = aligned_targets(batch["target"].to(self.device), assignment)
            errors = parameter_errors(predictions, targets, self.p.loss.readout_weight).joint
            total += errors.double().sum().item()
            count += len(errors)
        value = total / count
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite validation joint MSE")
        return value

    def checkpoint(self, *, weights_only: bool = False) -> dict[str, Any]:
        identity = self.train.manifest["identity"]
        state = {
            "version": CHECKPOINT_VERSION,
            "kind": "weights" if weights_only else "resumable",
            "model": self.model.state_dict(),
            "step": self.step,
            "epoch": self.consumed // len(self.train),
            "samples_consumed": self.consumed,
            "control": self.control,
            "readout_weight": float(self.p.loss.readout_weight),
            "dataset_id": self.train.manifest["dataset_id"],
            "split_signatures": {
                "train": self.train.signature,
                "validation": self.validation.signature,
            },
            "state_layout": identity["state_layout"],
            "target_layout": TARGET_LAYOUT,
            "source_model_digest": identity["source_model_digest"],
            "source_provenance": self.train.manifest.get("source_provenance"),
            "config": resolved_config(self.cfg),
            "solver": self.solver.identity,
            "training_contract": self.contract,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "best_checkpoint": str((self.out_dir / "checkpoints/best.pt").resolve())
            if self.best_step is not None
            else None,
            "wandb_run": self.logger.run_reference(),
        }
        if not weights_only:
            state.update(
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                rng=_rng_state(),
                target_permutation=self.mapping,
                history=self.history,
                validation_history=self.validation_history,
            )
        return state

    def save(self, name: str) -> Path:
        path = self.out_dir / "checkpoints" / f"{name}.pt"
        tick = time.perf_counter()
        save_checkpoint(path, self.checkpoint())
        self.timings["checkpoint_write"] = (
            self.timings.get("checkpoint_write", 0) + time.perf_counter() - tick
        )
        return path

    def fit(self) -> dict[str, Any]:
        self.logger.start()
        if self.p.training.resume and self.best_step is not None:
            if self.best_step == self.step:
                self.save("best")
            else:
                previous = Path(self._previous_best)
                best = load_checkpoint(previous, self.validation)
                if best["step"] != self.best_step or best["best_loss"] != self.best_loss:
                    raise ValueError(
                        "the previous best checkpoint was replaced; resume that best checkpoint "
                        "or restore its matching last/best pair"
                    )
                if previous.resolve() != (self.out_dir / "checkpoints/best.pt").resolve():
                    tick = time.perf_counter()
                    save_checkpoint(self.out_dir / "checkpoints/best.pt", best)
                    self.timings["carry_best_checkpoint"] = time.perf_counter() - tick
                del best
        write_json(self.out_dir / "config.json", resolved_config(self.cfg))
        if self.mapping is not None:
            np.save(self.out_dir / "target-permutation.npy", self.mapping, allow_pickle=False)
        started = time.perf_counter()
        while self.step < self.p.training.num_steps:
            tick = time.perf_counter()
            result = self.update(self.next_batch())
            record = {
                "step": self.step,
                "loss": result.loss,
                "grad_norm": result.grad_norm,
                "lr": self.optimizer.param_groups[0]["lr"],
                "samples_consumed": self.consumed,
                "episodes": result.episodes,
                "seconds": time.perf_counter() - tick,
                "matching_nodes_mean": float(result.assignment.counters[:, 1].mean()),
            }
            self.history.append(record)
            if self.step % self.p.training.log_every == 0 or self.step == self.p.training.num_steps:
                self.logger.log(
                    {f"probe/train/{k}": float(record[k]) for k in ("loss", "grad_norm", "lr")},
                    self.step,
                )
            if (
                self.step % self.p.training.validation_every == 0
                or self.step == self.p.training.num_steps
            ):
                validation = self.validate()
                self.validation_history.append({"step": self.step, "joint_mse": validation})
                self.logger.log({"probe/validation/joint_mse": validation}, self.step)
                if self.best_loss is None or validation < self.best_loss:
                    self.best_loss, self.best_step = validation, self.step
                    self.save("best")
            if (
                self.step % self.p.training.checkpoint_every == 0
                or self.step == self.p.training.num_steps
            ):
                self.save("last")
                write_json(
                    self.out_dir / "history.json",
                    {"training": self.history, "validation": self.validation_history},
                )
        if not (self.out_dir / "checkpoints/last.pt").exists():
            self.save("last")
        write_json(
            self.out_dir / "history.json",
            {"training": self.history, "validation": self.validation_history},
        )
        from iccl.analysis.plotting import plot_probe_history

        plot_probe_history([self.out_dir / "history.json"], self.out_dir / "plots")
        summary = {
            "step": self.step,
            "samples_consumed": self.consumed,
            "best_step": self.best_step,
            "best_validation_joint_mse": self.best_loss,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "control": self.control,
            "dataset_id": self.train.manifest["dataset_id"],
            "fit_seconds": time.perf_counter() - started,
            "setup_and_checkpoint_seconds": self.timings,
        }
        write_json(self.out_dir / "training.json", summary)
        if self.cfg.wandb.get("upload_weights"):
            path = self.out_dir / "weights.pt"
            save_checkpoint(path, self.checkpoint(weights_only=True))
            self.logger.upload_probe_artifact(path, kind="weights")
        self.logger.upload_probe_artifact(self.out_dir / "history.json", kind="results")
        return summary

    def close(self) -> None:
        # Dropping the iterator joins its multiprocessing workers before native pool teardown.
        self._iterator = None
        self.solver.close()
        self.train.close()
        self.validation.close()
        self.logger.finish()

    def __enter__(self) -> ProbeTrainer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def train_probe(cfg: DictConfig, out_dir: Path | str) -> dict[str, Any]:
    with ProbeTrainer(cfg, out_dir) as trainer:
        return trainer.fit()
