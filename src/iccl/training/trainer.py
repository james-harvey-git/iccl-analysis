"""Meta-training loop: optimization, evaluation, and checkpointing.

Trains the sequence model on the on-the-fly ICCL stream with a masked MSE loss
at every demonstration position. Single-device by design (the pilot model fits
one GPU with room to spare; sweeps parallelize across GPUs via hydra multirun).
Metric reporting belongs to ``iccl.training.logger``, which the standalone
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
    SequenceDataset,
    collate_sequences,
    make_family,
    sequence_config_from,
)
from iccl.models.model import model_from_config
from iccl.training.logger import RunLogger
from iccl.training.metrics import evaluate_suites, load_eval_suites
from iccl.utils import resolve_device

BEST_METRIC = "in_dist/nmse_last_demo"


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


class Trainer:
    """Owns the model, optimizer, eval harness, and checkpoint state for one run.

    ``out_dir`` receives ``checkpoints/`` and ``eval/`` subdirectories; pass the
    hydra run dir from scripts. ``cfg.training.resume`` restores a checkpoint
    and continues the data stream at the exact sample offset (bit-exact given
    the same batch size and worker count).
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
        self.last_loss = float("nan")
        self.logger = RunLogger(cfg, self.out_dir, job_type="train")
        if cfg.training.resume is not None:
            self._load_checkpoint(Path(cfg.training.resume))

    # ------------------------------------------------------------------ setup

    def _build_loader(self) -> DataLoader:
        train_cfg = self.cfg.training
        family = make_family(self.cfg.data)
        dataset = SequenceDataset(
            family,
            sequence_config_from(self.cfg.data),
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
        eval_enabled = train_cfg.eval_every is not None
        suites = load_eval_suites(Path(self.cfg.data.eval_sets.out_dir)) if eval_enabled else {}
        self.logger.start()
        loader = self._build_loader()
        batches = iter(loader)
        grad_clip = train_cfg.grad_clip if train_cfg.grad_clip is not None else float("inf")

        self.model.train()
        window_start, window_tokens = time.perf_counter(), 0
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
            window_tokens += batch["tokens"].shape[0] * batch["tokens"].shape[1]

            if self.step % train_cfg.log_every == 0:
                elapsed = time.perf_counter() - window_start
                self.logger.log(
                    {
                        "train/loss": self.last_loss,
                        "train/lr": float(self.scheduler.get_last_lr()[0]),
                        "train/grad_norm": float(grad_norm),
                        "train/sequences_per_s": train_cfg.log_every
                        * train_cfg.batch_size
                        / elapsed,
                        "train/tokens_per_s": window_tokens / elapsed,
                    },
                    self.step,
                )
                window_start, window_tokens = time.perf_counter(), 0
            if eval_enabled and self.step % train_cfg.eval_every == 0:
                self._evaluate(suites)
            if self.step % train_cfg.checkpoint_every == 0:
                self._save_checkpoint("last.pt")

        if eval_enabled and self.step % train_cfg.eval_every != 0:
            self._evaluate(suites)
        self._save_checkpoint("last.pt")
        self.logger.finish()

    def _evaluate(self, suites: dict[str, dict[str, np.ndarray]]) -> None:
        self.model.eval()
        scalars, curves = evaluate_suites(
            self.model, suites, self.device, autocast_dtype=self.autocast_dtype
        )
        self.model.train()
        self.logger.log_eval(scalars, curves, self.step)
        if BEST_METRIC in scalars and scalars[BEST_METRIC] < self.best_metric:
            self.best_metric = scalars[BEST_METRIC]
            self._save_checkpoint("best.pt")

    # ----------------------------------------------------------- persistence

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
        self.best_metric = ckpt["best_metric"]
        torch.set_rng_state(ckpt["torch_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        if self.device.type == "cuda" and "cuda_rng" in ckpt:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
