"""Meta-training loop: optimization, evaluation, and checkpointing.

Trains the sequence model on the on-the-fly ICCL stream with a masked MSE loss
at every demonstration position. Single-device by design (the pilot model fits
one GPU with room to spare; sweeps parallelize across GPUs via hydra multirun).
Metric reporting belongs to ``iccl.reporting.logger``, which the standalone
evaluation entry point shares.
"""

import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from iccl.data.dataset import (
    collate_sequences,
    sequence_dataset_from_config,
)
from iccl.data.sequences import TOKEN_PAD
from iccl.evaluation.metrics import EvaluationReport, evaluate_suites, load_eval_suites
from iccl.models.model import model_from_config
from iccl.reporting.logger import RunLogger
from iccl.utils import resolve_device


def masked_mse(
    preds: Float[torch.Tensor, "batch seq d_out"],
    targets: Float[torch.Tensor, "batch seq d_out"],
    mask: Float[torch.Tensor, "batch seq"],
) -> Float[torch.Tensor, ""]:
    """Token-weighted MSE over the masked (x-token) positions."""
    return ((preds - targets).pow(2).mean(dim=-1) * mask).sum() / mask.sum()


def split_decay_params(
    model: nn.Module,
) -> tuple[dict[str, nn.Parameter], dict[str, nn.Parameter]]:
    """Split parameters into (decay, no_decay) by name. Weight decay applies
    only to matrices without the ``_no_weight_decay`` flag; vectors and scalars
    (norm scales, biases, ``A_log``/``dt_bias``) are exempt."""
    decay: dict[str, nn.Parameter] = {}
    no_decay: dict[str, nn.Parameter] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        exempt = param.ndim < 2 or getattr(param, "_no_weight_decay", False)
        (no_decay if exempt else decay)[name] = param
    return decay, no_decay


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    if cfg.optimizer != "adamw":
        raise ValueError(f"unknown optimizer: {cfg.optimizer}")
    decay, no_decay = split_decay_params(model)
    groups = [
        {"params": list(decay.values()), "weight_decay": cfg.weight_decay},
        {"params": list(no_decay.values()), "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: DictConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup, then constant or cosine over the remaining steps."""
    if cfg.schedule not in ("constant", "cosine"):
        raise ValueError(f"unknown schedule: {cfg.schedule}")

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.schedule == "constant":
            return 1.0
        progress = (step - cfg.warmup_steps) / max(1, cfg.num_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def resolve_snapshot_steps(cfg: DictConfig) -> list[int]:
    """Steps at which to keep a step-tagged copy of the weights, from
    ``cfg.snapshots``.

    The union of three generators — an explicit list, a linear cadence, and a
    log-spaced sequence — so a run can have both a dense early sample and a
    steady later one without choosing. ``num_steps`` is always included, so the
    terminal weights are always in the series and analysis code can glob the
    directory without special-casing the end of training.
    """
    num_steps = int(cfg.num_steps)
    steps = {num_steps}
    snapshots = cfg.get("snapshots")
    if snapshots is not None:
        for step in snapshots.get("at") or []:
            steps.add(int(step))
        every = snapshots.get("every")
        if every is not None:
            if every <= 0:
                raise ValueError(f"snapshots.every must be positive, got {every}")
            steps.update(range(int(every), num_steps + 1, int(every)))
        log = snapshots.get("log")
        if log is not None:
            start, count = int(log.start), int(log.count)
            if start < 1 or count < 1:
                raise ValueError(f"snapshots.log needs start >= 1 and count >= 1, got {log}")
            # Rounding collides at the dense low end; the set absorbs the repeats.
            steps.update(int(round(step)) for step in np.geomspace(start, num_steps, count))
    return sorted(step for step in steps if 0 < step <= num_steps)


def resolve_autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """None means full fp32 (no autocast); ``auto`` picks bf16 on CUDA only."""
    match precision:
        case "auto":
            return torch.bfloat16 if device.type == "cuda" else None
        case "bf16":
            return torch.bfloat16
        case "fp32":
            return None
        case _:
            raise ValueError(f"unknown precision: {precision}")


def is_monitor_suite(metadata: dict[str, Any]) -> bool:
    """Whether frozen-suite metadata belongs to the canonical live monitor."""
    return metadata.get("capability") in {
        "icl",
        "composition",
        "retention",
    } and "canonical" in metadata.get("family_memberships", ())


class Trainer:
    """Owns the model, optimizer, eval harness, and checkpoint state for one run.

    ``out_dir`` receives ``checkpoints/``, ``snapshots/`` and ``monitor/``
    subdirectories; pass the hydra run dir from scripts. ``cfg.training.resume``
    restores a checkpoint and continues the data stream at the exact sample
    offset (bit-exact given the same batch size and worker count). Snapshot
    steps already behind a resumed run are not retaken, so a resumed run's
    series has a gap wherever the original run's did.
    """

    def __init__(self, cfg: DictConfig, out_dir: Path | str) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.device = resolve_device(cfg.device)
        self.autocast_dtype = resolve_autocast_dtype(cfg.training.precision, self.device)
        self.model = model_from_config(cfg).to(self.device)
        self.optimizer = build_optimizer(self.model, cfg.training)
        self.scheduler = build_scheduler(self.optimizer, cfg.training)
        self.step = 0
        self.best_metric = float("inf")
        self.best_metric_name = str(cfg.data.eval_sets.get("best_metric", ""))
        self.last_loss = float("nan")
        self.snapshot_steps = resolve_snapshot_steps(cfg.training)
        self.logger = RunLogger(cfg, self.out_dir, job_type="train")
        if cfg.training.resume is not None:
            self._load_checkpoint(Path(cfg.training.resume))

    # ------------------------------------------------------------------ setup

    def _build_loader(self) -> DataLoader:
        train_cfg = self.cfg.training
        dataset = sequence_dataset_from_config(
            self.cfg.data,
            base_seed=self.cfg.seed,
            start_index=self.step * train_cfg.batch_size,
        )
        return DataLoader(
            dataset,
            batch_size=train_cfg.batch_size,
            collate_fn=collate_sequences,
            num_workers=train_cfg.num_workers,
            persistent_workers=train_cfg.num_workers > 0,
            pin_memory=self.device.type == "cuda",
        )

    # ------------------------------------------------------------------- loop

    def fit(self) -> None:
        train_cfg = self.cfg.training
        validation_enabled = train_cfg.validation_every is not None
        monitor_enabled = train_cfg.monitor_every is not None
        eval_dir = Path(self.cfg.data.eval_sets.out_dir)
        validation_suites = (
            load_eval_suites(eval_dir, select=lambda meta: meta.get("capability") == "validation")
            if validation_enabled
            else {}
        )
        monitor_suites = (
            load_eval_suites(
                eval_dir,
                select=is_monitor_suite,
            )
            if monitor_enabled
            else {}
        )
        self.logger.start()
        self._report_snapshot_plan()
        loader = self._build_loader()
        batches = iter(loader)
        grad_clip = train_cfg.grad_clip if train_cfg.grad_clip is not None else float("inf")

        self.model.train()
        window_start = time.perf_counter()
        window_padded_tokens = 0
        window_real_tokens = 0
        window_sequences = 0
        while self.step < train_cfg.num_steps:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in next(batches).items()}
            ctx = (
                torch.autocast(self.device.type, self.autocast_dtype)
                if self.autocast_dtype is not None
                else nullcontext()
            )
            with ctx:
                out = self.model(batch["tokens"], batch["token_type"])
            loss = masked_mse(out.preds.float(), batch["targets"], batch["loss_mask"])
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
            self.last_loss = loss.item()
            batch_sequences = batch["tokens"].shape[0]
            window_sequences += batch_sequences
            window_padded_tokens += int(batch["token_type"].numel())
            window_real_tokens += int((batch["token_type"] != TOKEN_PAD).sum().item())

            if self.step % train_cfg.log_every == 0:
                elapsed = time.perf_counter() - window_start
                self.logger.log(
                    {
                        "train/token_mse": self.last_loss,
                        "train/lr": float(self.scheduler.get_last_lr()[0]),
                        "train/grad_norm": float(grad_norm),
                        "train/sequences_per_s": train_cfg.log_every
                        * train_cfg.batch_size
                        / elapsed,
                        "train/padding_fraction": 1.0 - window_real_tokens / window_padded_tokens,
                        "train/mean_serialized_length": window_real_tokens / window_sequences,
                    },
                    self.step,
                )
                window_start = time.perf_counter()
                window_padded_tokens = 0
                window_real_tokens = 0
                window_sequences = 0
            if validation_enabled and self.step % train_cfg.validation_every == 0:
                self._validate(validation_suites)
            if monitor_enabled and self.step % train_cfg.monitor_every == 0:
                self._monitor(monitor_suites)
            if self.step % train_cfg.checkpoint_every == 0:
                self._save_checkpoint("last.pt")
            if self.step in self.snapshot_steps:
                self._save_snapshot()
            if self.step < train_cfg.num_steps:
                self.logger.flush()

        if validation_enabled and self.step % train_cfg.validation_every != 0:
            self._validate(validation_suites)
        if monitor_enabled and self.step % train_cfg.monitor_every != 0:
            self._monitor(monitor_suites)
        self._save_checkpoint("last.pt")
        final_snapshot = self._snapshot_path(self.step)
        if not final_snapshot.exists():
            # A run resumed at or past num_steps never entered the loop, so the
            # terminal snapshot the series guarantees has not been written yet.
            self._save_snapshot()
        self.logger.upload_weights(final_snapshot)
        self.logger.finish()

    def _evaluation_report(self, suites: dict[str, dict[str, np.ndarray]]) -> EvaluationReport:
        self.model.eval()
        report = evaluate_suites(
            self.model,
            suites,
            self.device,
            autocast_dtype=self.autocast_dtype,
            bootstrap_seed=int(self.cfg.data.eval_sets.get("bootstrap_seed", 0)),
            bootstrap_replicates=int(self.cfg.data.eval_sets.get("bootstrap_replicates", 1000)),
        )
        self.model.train()
        return report

    def _validate(self, suites: dict[str, dict[str, np.ndarray]]) -> None:
        report = self._evaluation_report(suites)
        if self.best_metric_name not in report.scalars:
            raise KeyError(
                f"best metric {self.best_metric_name!r} is absent from the frozen evaluation "
                "sets; regenerate them with scripts/make_eval_sets.py"
            )
        metric = report.scalars[self.best_metric_name]
        self.logger.log_validation(metric, self.step)
        if metric < self.best_metric:
            self.best_metric = metric
            self._save_checkpoint("best.pt")

    def _monitor(self, suites: dict[str, dict[str, np.ndarray]]) -> None:
        self.logger.log_monitor(self._evaluation_report(suites), self.step)

    def _report_snapshot_plan(self) -> None:
        """The resolved schedule and its disk cost, once at startup, so a
        misconfigured cadence is obvious before it fills a filesystem."""
        per_snapshot = sum(p.numel() for p in self.model.parameters()) * 4
        if self._keeps_optimizer_state():
            per_snapshot *= 3  # weights plus AdamW's two moments
        listed = ", ".join(str(step) for step in self.snapshot_steps[:6])
        tail = ", ..." if len(self.snapshot_steps) > 6 else ""
        total_mb = per_snapshot * len(self.snapshot_steps) / 1e6
        print(f"snapshots: {len(self.snapshot_steps)} steps [{listed}{tail}], ~{total_mb:.0f} MB")

    # ----------------------------------------------------------- persistence

    def _keeps_optimizer_state(self) -> bool:
        snapshots = self.cfg.training.get("snapshots")
        return snapshots is not None and bool(snapshots.get("include_optimizer_state"))

    def _snapshot_path(self, step: int) -> Path:
        return self.out_dir / "snapshots" / f"step_{step:07d}.pt"

    def _save_snapshot(self) -> None:
        """Step-tagged weights, kept rather than overwritten: the rolling
        checkpoint serves resumption, this serves interpretability and
        re-scoring under metrics defined after the run.

        Carries the step and run reference ``scripts/eval.py`` needs to score a
        snapshot and link it back to the run that produced it.
        """
        path = self._snapshot_path(self.step)
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "model": self.model.state_dict(),
            "step": self.step,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "wandb_run": self.logger.run_reference(),
        }
        if self._keeps_optimizer_state():
            state["optimizer"] = self.optimizer.state_dict()
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, path)

    def _save_checkpoint(self, name: str) -> None:
        ckpt_dir = self.out_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "samples_consumed": self.step * self.cfg.training.batch_size,
            "best_metric": self.best_metric,
            "best_metric_name": self.best_metric_name,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            # Lets an evaluation of this checkpoint name and link back to the
            # run that produced it.
            "wandb_run": self.logger.run_reference(),
        }
        if self.device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(state, ckpt_dir / name)

    def _load_checkpoint(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = ckpt["step"]
        self.best_metric = (
            ckpt["best_metric"]
            if ckpt.get("best_metric_name") == self.best_metric_name
            else float("inf")
        )
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        if self.device.type == "cuda" and "cuda_rng" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
