"""Controlled training-performance experiments; runtime training has no baseline switch."""

import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import statistics
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from iccl.data.dataset import collate_sequences, sequence_dataset_from_config, to_tensors
from iccl.data.eval_bundle import validate_eval_bundle
from iccl.data.sequences import TOKEN_PAD
from iccl.evaluation.metrics import evaluate_suites, load_eval_suites, predict_suite
from iccl.evaluation.results import write_evaluation_results
from iccl.models.model import GDNModel, ModelOutput, TokenEmbedding, model_from_config
from iccl.reporting.logger import RunLogger
from iccl.training.trainer import (
    Trainer,
    build_optimizer,
    build_scheduler,
    is_monitor_suite,
    masked_mse,
    resolve_autocast_dtype,
)
from iccl.utils import resolve_device, seed_everything

VARIANTS = (
    "baseline",
    "optimized",
    "group2",
    "group4",
    "bias",
    "all_types",
    "bias_group2",
    "all_types_group2",
)


def baseline_embedding(self: TokenEmbedding, tokens: torch.Tensor, kinds: torch.Tensor):
    """The indexed affine embedding at main revision 4f49da4, for the experimental control."""
    kinds = kinds.long()
    return torch.einsum("btoi,bti->bto", self.weight[kinds], tokens) + self.bias[kinds]


def bias_embedding(self: TokenEmbedding, tokens: torch.Tensor, kinds: torch.Tensor):
    kinds = kinds.long()
    return torch.einsum("btoi,bti->bto", self.weight[kinds], tokens) + F.embedding(kinds, self.bias)


def all_types_embedding(self: TokenEmbedding, tokens: torch.Tensor, kinds: torch.Tensor):
    kinds = kinds.long()
    values = F.linear(tokens, self.weight.flatten(0, 1)).unflatten(-1, self.weight.shape[:2])
    indices = kinds[..., None, None].expand(-1, -1, 1, self.weight.shape[1])
    return values.gather(-2, indices).squeeze(-2) + F.embedding(kinds, self.bias)


def install_variant(model: GDNModel, variant: str) -> None:
    """Change computation on this experimental model without changing its parameter schema."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown performance variant: {variant}")
    if variant == "baseline":
        model.embed.forward = MethodType(baseline_embedding, model.embed)
    if variant in {"bias", "bias_group2"}:
        model.embed.forward = MethodType(bias_embedding, model.embed)
    if variant in {"all_types", "all_types_group2"}:
        model.embed.forward = MethodType(all_types_embedding, model.embed)
    groups = {"group2": 2, "group4": 4, "bias_group2": 2, "all_types_group2": 2}.get(variant)
    if groups is None:
        return
    ordinary_forward = model.forward

    def grouped_forward(self, tokens, token_type, **kwargs):
        if kwargs.get("capture", False):
            raise ValueError("the length-group experiment does not capture state trajectories")
        lengths = (token_type != TOKEN_PAD).sum(dim=1)
        order = torch.argsort(lengths, stable=True)
        predictions = []
        for indices in torch.tensor_split(order, min(groups, len(order))):
            length = int(lengths[indices].max().item())
            output = ordinary_forward(
                tokens[indices, :length], token_type[indices, :length], **kwargs
            )
            predictions.append(F.pad(output.preds, (0, 0, 0, tokens.shape[1] - length)))
        return ModelOutput(preds=torch.cat(predictions)[torch.argsort(order)])

    model.forward = MethodType(grouped_forward, model)


def write_result(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def provenance(cfg: DictConfig, device: torch.device) -> dict[str, Any]:
    packages = {}
    for name in ("torch", "triton", "fla-core", "tilelang"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    checkpoint_path = cfg.performance.checkpoint or cfg.training.resume
    checkpoint_digest = None
    if checkpoint_path:
        with Path(checkpoint_path).open("rb") as stream:
            checkpoint_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__,
        "packages": packages,
        "source_file": inspect.getfile(model_from_config),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "checkpoint_sha256": checkpoint_digest,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def autocast(cfg: DictConfig, device: torch.device):
    dtype = resolve_autocast_dtype(cfg.training.precision, device)
    return torch.autocast(device.type, dtype) if dtype is not None else nullcontext()


def checkpoint_from(cfg: DictConfig) -> dict[str, Any] | None:
    path = cfg.performance.checkpoint
    return torch.load(path, map_location="cpu", weights_only=False) if path else None


def fixed_batches(cfg: DictConfig, count: int, start_step: int = 0):
    dataset = sequence_dataset_from_config(cfg.data, base_seed=int(cfg.seed))
    size, workers = int(cfg.training.batch_size), max(1, int(cfg.training.num_workers))
    batches = []
    digest = hashlib.sha256()
    for batch_index in range(count):
        first = (
            start_step * size + (batch_index // workers) * size * workers + batch_index % workers
        )
        batch = collate_sequences(
            [to_tensors(dataset.build(first + row * workers)) for row in range(size)]
        )
        for tensor in batch.values():
            digest.update(tensor.numpy().tobytes())
        batches.append(batch)
    return batches, digest.hexdigest()


def make_training_state(cfg, device, state, checkpoint, variant):
    model = model_from_config(cfg).to(device)
    model.load_state_dict(state, strict=True)
    install_variant(model, variant)
    optimizer = build_optimizer(model, cfg.training)
    scheduler = build_scheduler(optimizer, cfg.training)
    if checkpoint is not None:
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
        scheduler.load_state_dict(checkpoint["scheduler"])
    return model, optimizer, scheduler


def initial_state(cfg, checkpoint):
    seed_everything(int(cfg.performance.initialization_seed))
    return checkpoint["model"] if checkpoint is not None else model_from_config(cfg).state_dict()


def errors(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual, expected = actual.double(), expected.double()
    difference = actual - expected
    return {
        "relative_rms": float(difference.norm() / expected.norm().clamp_min(1e-12)),
        "max_abs": float(difference.abs().max()),
    }


def numerical(cfg: DictConfig, out_dir: Path, device: torch.device) -> None:
    checkpoint = checkpoint_from(cfg)
    state = initial_state(cfg, checkpoint)
    cpu_batches, digest = fixed_batches(cfg, 1, checkpoint["step"] if checkpoint else 0)
    batch = {key: value.to(device) for key, value in cpu_batches[0].items()}
    real = cpu_batches[0]["token_type"] != TOKEN_PAD
    rows = []
    for precision in ("fp32", "bf16"):
        cfg.training.precision = precision
        reference = None
        for variant in (*VARIANTS, "baseline"):
            model, optimizer, scheduler = make_training_state(
                cfg, device, state, checkpoint, variant
            )
            with autocast(cfg, device):
                predictions = model(batch["tokens"], batch["token_type"]).preds
            loss = masked_mse(predictions.float(), batch["targets"], batch["loss_mask"])
            loss.backward()
            gradients = {}
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                gradients[name] = parameter.grad.detach().cpu().clone()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
            updates = {name: p.detach().cpu() - state[name] for name, p in model.named_parameters()}
            observed = {
                "predictions": predictions.detach().float().cpu()[real],
                "gradients": gradients,
                "updates": updates,
                "loss": float(loss),
            }
            assert all(torch.isfinite(g).all() for g in gradients.values())
            if reference is None:
                reference = observed
            else:
                grad_errors = {
                    name: errors(g, reference["gradients"][name]) for name, g in gradients.items()
                }
                global_grad = errors(
                    torch.cat([g.flatten() for g in gradients.values()]),
                    torch.cat([g.flatten() for g in reference["gradients"].values()]),
                )
                global_update = errors(
                    torch.cat([g.flatten() for g in updates.values()]),
                    torch.cat([g.flatten() for g in reference["updates"].values()]),
                )
                prediction_error = errors(observed["predictions"], reference["predictions"])
                loss_error = abs(observed["loss"] - reference["loss"]) / max(
                    abs(reference["loss"]), 1e-12
                )
                limits = (
                    {
                        "prediction": 5e-5,
                        "gradient": 1e-4,
                        "parameter": 1e-3,
                        "update": 1e-3,
                        "loss": 5e-5,
                    }
                    if precision == "fp32"
                    else {
                        "prediction": 5e-3,
                        "gradient": 1e-2,
                        "parameter": 2e-2,
                        "update": 2e-2,
                        "loss": 2e-3,
                    }
                )
                maximum = max(value["relative_rms"] for value in grad_errors.values())
                passed = (
                    prediction_error["relative_rms"] <= limits["prediction"]
                    and global_grad["relative_rms"] <= limits["gradient"]
                    and maximum <= limits["parameter"]
                    and global_update["relative_rms"] <= limits["update"]
                    and loss_error <= limits["loss"]
                )
                row = {
                    "variant": variant,
                    "precision": precision,
                    "optimizer_learning_rate": learning_rate,
                    "passed": passed,
                    "loss": observed["loss"],
                    "loss_relative_error": loss_error,
                    "predictions": prediction_error,
                    "global_gradient": global_grad,
                    "global_update": global_update,
                    "max_parameter_gradient_relative_rms": maximum,
                    "gradient_errors": grad_errors,
                    "limits": limits,
                }
                rows.append(row)
                print(
                    json.dumps({k: v for k, v in row.items() if k != "gradient_errors"}), flush=True
                )
            del model, optimizer, scheduler, predictions, loss, observed
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_result(
        out_dir / "numerical.json",
        provenance(cfg, device) | {"batch_sha256": digest, "comparisons": rows},
    )


def precision_probe(cfg: DictConfig, out_dir: Path, device: torch.device) -> None:
    """Locate FP32 differences without changing any numerical acceptance threshold."""
    checkpoint = checkpoint_from(cfg)
    state = initial_state(cfg, checkpoint)
    batches, digest = fixed_batches(cfg, 1, checkpoint["step"] if checkpoint else 0)
    batch = {key: value.to(device) for key, value in batches[0].items()}
    real = batch["token_type"] != TOKEN_PAD
    original_flag = torch.backends.cudnn.allow_tf32
    rows = []
    try:
        for flag in (True, False):
            torch.backends.cudnn.allow_tf32 = flag
            reference = None
            for variant in ("baseline", "optimized", "bias", "all_types"):
                model = model_from_config(cfg).to(device)
                model.load_state_dict(state)
                install_variant(model, variant)
                with torch.no_grad():
                    observed = {
                        "embedding": model.embed(batch["tokens"], batch["token_type"])[real],
                        "predictions": model(batch["tokens"], batch["token_type"]).preds[real],
                    }
                if reference is None:
                    reference = observed
                else:
                    row = {"variant": variant, "cudnn_allow_tf32": flag} | {
                        name: errors(value, reference[name]) for name, value in observed.items()
                    }
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                del model
    finally:
        torch.backends.cudnn.allow_tf32 = original_flag
    write_result(
        out_dir / "precision-probe.json",
        provenance(cfg, device) | {"batch_sha256": digest, "comparisons": rows},
    )


def training_step(batch, model, optimizer, scheduler, cfg, device):
    with autocast(cfg, device):
        output = model(batch["tokens"], batch["token_type"])
    loss = masked_mse(output.preds.float(), batch["targets"], batch["loss_mask"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip = cfg.training.grad_clip if cfg.training.grad_clip is not None else float("inf")
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    scheduler.step()
    value = loss.item()
    int((batch["token_type"] != TOKEN_PAD).sum().item())
    assert math.isfinite(value)
    return value


def benchmark(cfg: DictConfig, out_dir: Path, device: torch.device) -> None:
    checkpoint = checkpoint_from(cfg)
    state = initial_state(cfg, checkpoint)
    cpu_batches, digest = fixed_batches(
        cfg, int(cfg.performance.batches), checkpoint["step"] if checkpoint else 0
    )
    batches = [{key: value.to(device) for key, value in batch.items()} for batch in cpu_batches]
    rows = []
    for variant in (*VARIANTS, "baseline"):
        model, optimizer, scheduler = make_training_state(cfg, device, state, checkpoint, variant)

        for batch in batches:
            training_step(batch, model, optimizer, scheduler, cfg, device)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = []
        value = float("nan")
        for _ in range(int(cfg.performance.repetitions)):
            sync(device)
            start = time.perf_counter()
            for batch in batches:
                value = training_step(batch, model, optimizer, scheduler, cfg, device)
            sync(device)
            samples.append((time.perf_counter() - start) / len(batches))
        row = {
            "variant": variant,
            "step_seconds_trials": samples,
            "median_step_seconds": statistics.median(samples),
            "last_loss": value,
            "sequences_per_second": int(cfg.training.batch_size) / statistics.median(samples),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        print(json.dumps(row), flush=True)
        rows.append(row)
        del model, optimizer, scheduler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_result(
        out_dir / "benchmark.json",
        provenance(cfg, device) | {"batch_sha256": digest, "measurements": rows},
    )


class RecordingLogger(RunLogger):
    def __init__(self, cfg, out_dir):
        super().__init__(cfg, out_dir, job_type="training-performance")
        self.started = time.perf_counter()

    def log(self, metrics: dict[str, float], step: int) -> None:
        with (self.out_dir / "history.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "step": step,
                        "elapsed_seconds": time.perf_counter() - self.started,
                        "metrics": metrics,
                    }
                )
                + "\n"
            )
        super().log(metrics, step)


def preflight(cfg: DictConfig, out_dir: Path, device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("the cluster preflight requires CUDA")
    checks = subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "--no-sync",
            "pytest",
            "-q",
            "-m",
            "cuda",
            "-p",
            "no:cacheprovider",
            "--basetemp",
            str(out_dir / "pytest"),
        ],
        capture_output=True,
        text=True,
    )
    (out_dir / "cuda-tests.log").write_text(checks.stdout + checks.stderr)
    print(checks.stdout + checks.stderr, flush=True)
    checks.check_returncode()
    bundle = validate_eval_bundle(cfg)
    checkpoint = cfg.performance.checkpoint
    if not checkpoint:
        raise ValueError("preflight needs a full-state checkpoint")
    for label, source in (("fresh", None), ("resumed", checkpoint)):
        variant_cfg = copy.deepcopy(cfg)
        variant_cfg.performance.checkpoint = source
        destination = out_dir / label
        destination.mkdir()
        numerical(variant_cfg, destination, device)
    destination = out_dir / "timing"
    destination.mkdir()
    benchmark(cfg, destination, device)
    write_result(
        out_dir / "preflight.json",
        provenance(cfg, device)
        | {"cuda_tests_passed": True, "evaluation_bundle_suites": bundle["num_suites"]},
    )


def trajectory(cfg: DictConfig, out_dir: Path, device: torch.device) -> None:
    seed_everything(int(cfg.performance.initialization_seed))
    trainer = Trainer(cfg, out_dir)
    install_variant(trainer.model, str(cfg.performance.variant))
    trainer.logger = RecordingLogger(cfg, out_dir)
    start = time.perf_counter()
    trainer.fit()
    elapsed = time.perf_counter() - start
    assert all(torch.isfinite(p).all() for p in trainer.model.parameters())
    trainer.model.eval()
    suites = load_eval_suites(
        Path(cfg.data.eval_sets.out_dir),
        select=lambda meta: is_monitor_suite(meta) or meta.get("capability") == "validation",
        monitor=True,
    )
    report = evaluate_suites(
        trainer.model,
        suites,
        device,
        autocast_dtype=trainer.autocast_dtype,
        bootstrap_seed=int(cfg.data.eval_sets.bootstrap_seed),
        bootstrap_replicates=int(cfg.data.eval_sets.bootstrap_replicates),
    )
    for name, suite in suites.items():
        if suite["__meta__"]["capability"] == "validation":
            predictions = predict_suite(
                trainer.model, suite, device, autocast_dtype=trainer.autocast_dtype
            )
            squared_error = ((predictions - suite["targets"]) ** 2).mean(axis=-1)
            report.raw_errors[f"{name}/episode_error_sum"] = (
                squared_error * suite["loss_mask"]
            ).sum(axis=1)
            report.raw_errors[f"{name}/episode_token_count"] = suite["loss_mask"].sum(axis=1)
    write_evaluation_results(
        report,
        out_dir / "final-evaluation",
        trainer.step,
        provenance(cfg, device)
        | {
            "checkpoint_reference": str(out_dir / "checkpoints/last.pt"),
            "suites": {name: suite["__meta__"] for name, suite in suites.items()},
            "sample_scope": "training validation and canonical monitor episodes",
        },
    )
    result = provenance(cfg, device) | {
        "elapsed_seconds": elapsed,
        "step": trainer.step,
        "last_loss": trainer.last_loss,
        "best_validation": trainer.best_metric if math.isfinite(trainer.best_metric) else None,
        "wandb_run": trainer.logger.run_reference(),
    }
    write_result(out_dir / "trajectory.json", result)


@hydra.main(version_base=None, config_path="../configs", config_name="performance")
def main(cfg: DictConfig) -> None:
    torch.set_num_threads(1)
    device = resolve_device(cfg.device)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    mode = str(cfg.performance.mode)
    if mode == "preflight":
        preflight(cfg, out_dir, device)
    elif mode == "precision_probe":
        precision_probe(cfg, out_dir, device)
    elif mode == "numerical":
        numerical(cfg, out_dir, device)
    elif mode == "benchmark":
        benchmark(cfg, out_dir, device)
    elif mode == "trajectory":
        trajectory(cfg, out_dir, device)
    else:
        raise ValueError(f"unknown performance mode: {mode}")


if __name__ == "__main__":
    main()
