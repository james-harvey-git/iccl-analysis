"""Teacher parameter layout and the shared positive-scale convention for probes."""

import numpy as np

from iccl.data.teacher import ModulePool

PROTOCOL = "module-set-decoder-v1"
MODULES, WIDTH, AUGMENTED = 4, 16, 17
MODULE_SHAPE = (MODULES, AUGMENTED, WIDTH)
MODULE_FEATURES = int(np.prod(MODULE_SHAPE))
READOUT_FEATURES = WIDTH * WIDTH
OUTPUT_FEATURES = READOUT_FEATURES + MODULE_FEATURES
TARGET_LAYOUT = {
    "order": ["readout", "modules"],
    "readout": [WIDTH, WIDTH],
    "modules": list(MODULE_SHAPE),
    "features": OUTPUT_FEATURES,
    "module_assignment": "prediction-slot-to-target-module",
    "hidden_assignment": "prediction-coordinate-to-target-coordinate",
}


def canonical_pool(pool: ModulePool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Absorb each readout-row norm into every module's corresponding neuron.

    Intermediates are FP64. The resulting FP32 labels preserve the teacher up
    to rounding and fix the standard positive ReLU scaling symmetry.
    """
    if len(pool.modules) != 1 or len(pool.biases) != 1:
        raise ValueError("module decoding requires exactly one teacher hidden layer")
    weights, biases, readout = pool.modules[0], pool.biases[0], pool.readout
    for value, shape in zip(
        (weights, biases, readout), ((MODULES, 16, 16), (MODULES, 16), (16, 16)), strict=True
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid teacher array; expected finite {shape}")
    norms = np.linalg.norm(readout.astype(np.float64), axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("readout row norms must be finite and strictly positive")
    augmented = np.concatenate((weights, biases[:, None]), axis=1).astype(np.float64)
    modules = (augmented * norms[None, None]).astype(np.float32)
    normalized = (readout.astype(np.float64) / norms[:, None]).astype(np.float32)
    if not np.isfinite(modules).all() or not np.isfinite(normalized).all():
        raise ValueError("canonical parameters cannot be represented in FP32")
    return modules, normalized, norms


def episode_targets(pool: ModulePool, latents: np.ndarray) -> dict[str, np.ndarray]:
    """Store each observed module once, retaining the world and task metadata for scoring."""
    latents = np.asarray(latents, dtype=np.float32)
    if latents.ndim != 2 or latents.shape[1] != MODULES or len(latents) < MODULES - 1:
        raise ValueError("latents must have shape [T,4] with T >= 3")
    if not np.isfinite(latents).all():
        raise ValueError("latents must be finite")
    if not (np.count_nonzero(latents, axis=1) == 2).all() or (latents < 0).any():
        raise ValueError("each task must have exactly two positive module coefficients")
    counts = np.count_nonzero(latents, axis=0).astype(np.int64)
    if (counts == 0).any():
        raise ValueError("all four modules must appear before terminal state capture")
    modules, readout, norms = canonical_pool(pool)
    return {
        "modules": modules,
        "readout": readout,
        "world_modules": pool.modules[0].astype(np.float32),
        "world_biases": pool.biases[0].astype(np.float32),
        "world_readout": pool.readout.astype(np.float32),
        "latents": latents,
        "occurrence_count": counts,
        "latent_rank": np.asarray(np.linalg.matrix_rank(latents), dtype=np.int64),
        "readout_norms": norms,
    }


def flat_targets(modules: np.ndarray, readout: np.ndarray) -> np.ndarray:
    """Flatten any leading batch dimensions in the versioned readout-first layout."""
    if modules.shape[-3:] != MODULE_SHAPE or readout.shape[-2:] != (WIDTH, WIDTH):
        raise ValueError("invalid canonical target shapes")
    leading = modules.shape[:-3]
    if readout.shape[:-2] != leading:
        raise ValueError("module and readout batch dimensions differ")
    return np.concatenate(
        (readout.reshape(*leading, READOUT_FEATURES), modules.reshape(*leading, MODULE_FEATURES)),
        axis=-1,
    ).astype(np.float32)
