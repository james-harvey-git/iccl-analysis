"""Shared fresh-input and teacher-output contract for probe and GDN function scoring."""

import numpy as np

from iccl.data.dataset import sequence_rng
from iccl.data.teacher import ModulePool, teacher_forward
from iccl.evaluation.metrics import BASE_MSE_FLOOR


def functional_inputs(index: int, tasks: int, count: int, seed: int) -> np.ndarray:
    return sequence_rng(seed, index).uniform(-1, 1, size=(tasks, count, 16)).astype(np.float32)


def functional_targets(batch: dict[str, np.ndarray], inputs: np.ndarray) -> np.ndarray:
    """Evaluate each raw sampled teacher world, before any probe alignment or scaling."""
    outputs = []
    for episode in range(len(inputs)):
        pool = ModulePool(
            [batch["world_modules"][episode]],
            [batch["world_biases"][episode]],
            batch["world_readout"][episode],
        )
        outputs.append(
            np.stack(
                [
                    teacher_forward(pool, latent, task_inputs)
                    for latent, task_inputs in zip(
                        batch["latents"][episode], inputs[episode], strict=True
                    )
                ]
            )
        )
    return np.stack(outputs).astype(np.float64)


def functional_errors(predictions: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Per-task output MSE normalized by the same fresh-input teacher variance."""
    if predictions.shape != targets.shape or not np.isfinite(predictions).all():
        raise ValueError("functional predictions must be finite and match teacher output shapes")
    mse = ((predictions.astype(np.float64) - targets) ** 2).mean(axis=(-2, -1))
    variance = ((targets - targets.mean(axis=-2, keepdims=True)) ** 2).mean(axis=(-2, -1))
    return {
        "functional_mse_by_task": mse,
        "functional_nmse_by_task": mse / np.maximum(variance, BASE_MSE_FLOOR),
        "functional_output_variance": variance,
        "functional_variance_floored": variance < BASE_MSE_FLOOR,
    }
