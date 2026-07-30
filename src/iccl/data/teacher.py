"""HyperTeacher synthetic task family, ported from smonsays/scale-compositionality.

Source: ``compscale/data/teacher.py`` (JAX/Flax). Primitives are per-layer module
weight tensors of a ReLU teacher MLP; a task is a weighted k-hot latent over
modules, and the task's function composes each layer's weights as the
latent-weighted sum of that layer's modules, followed by a shared linear
readout. Unlike the source, the module pool here is sampled *per sequence*
(see ``iccl.data.sequences``), so this module separates the immutable task
family (dimensions, pattern enumeration) from pool sampling.

The source's global in-distribution/OOD combination split is deliberately not
ported: with per-sequence module re-instantiation, module indices are
exchangeable across sequences, so a globally held-out index pattern is
observationally indistinguishable from its in-distribution relabelings.
Compositional generalization is instead measured within-sequence (composite
final tasks with paired no-history controls, see ``iccl.data.sequences``).
"""

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

# E[|N(0,1)| truncated to (-2, 2)] std correction used by jax's variance_scaling
# initializer, kept identical so weight distributions match the source.
_TRUNC_STD_CORRECTION = 0.8796256610342398

_DISCRETE_WEIGHT_VALUES = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class TeacherConfig:
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    use_bias: bool
    num_modules: int
    scale: float
    weighting: str  # binary | discrete | continuous


@dataclass
class ModulePool:
    """One sequence's freshly sampled world: per-layer module weights and biases
    plus the shared readout."""

    modules: list[np.ndarray]  # layer l: [num_modules, d_{l-1}, d_l]
    biases: list[np.ndarray]  # layer l: [num_modules, d_l]
    readout: np.ndarray  # [hidden_dims[-1], output_dim]


def _truncated_normal(rng: np.random.Generator, shape: tuple[int, ...], std: float) -> np.ndarray:
    """Normal truncated to +-2 sigma with corrected std, matching jax's
    variance_scaling(..., distribution='truncated_normal')."""
    samples = rng.standard_normal(size=shape)
    out_of_bounds = np.abs(samples) > 2.0
    while out_of_bounds.any():
        samples[out_of_bounds] = rng.standard_normal(size=int(out_of_bounds.sum()))
        out_of_bounds = np.abs(samples) > 2.0
    return (samples * (std / _TRUNC_STD_CORRECTION)).astype(np.float32)


def sample_module_pool(cfg: TeacherConfig, rng: np.random.Generator) -> ModulePool:
    """Sample a fresh world: fan-in variance-scaled module weights, Uniform[0, 0.5)
    biases (zeros when use_bias is false), and a shared variance-scaled readout."""
    dims = (cfg.input_dim, *cfg.hidden_dims)
    modules: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for d_in, d_out in zip(dims[:-1], dims[1:], strict=True):
        std = math.sqrt(cfg.scale / d_in)
        modules.append(_truncated_normal(rng, (cfg.num_modules, d_in, d_out), std))
        if cfg.use_bias:
            biases.append(rng.uniform(0.0, 0.5, size=(cfg.num_modules, d_out)).astype(np.float32))
        else:
            biases.append(np.zeros((cfg.num_modules, d_out), dtype=np.float32))
    readout_std = math.sqrt(cfg.scale / cfg.hidden_dims[-1])
    readout = _truncated_normal(rng, (cfg.hidden_dims[-1], cfg.output_dim), readout_std)
    return ModulePool(modules=modules, biases=biases, readout=readout)


def teacher_forward(pool: ModulePool, latent: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Forward pass of the composed teacher for inputs ``x`` [n, input_dim].

    The latent is scaled by 1/sqrt(hotness) so composed-weight variance is
    invariant to how many modules are active (the source scales by its global
    num_hot; we use each task's actual hotness since hotness varies per task).
    """
    hotness = int(np.count_nonzero(latent))
    scaled = (latent / math.sqrt(hotness)).astype(np.float32)
    activations = x.astype(np.float32)
    for module_w, module_b in zip(pool.modules, pool.biases, strict=True):
        w = np.einsum("mih,m->ih", module_w, scaled)
        b = np.einsum("mh,m->h", module_b, scaled)
        activations = np.maximum(activations @ w + b, 0.0)
    return activations @ pool.readout


def make_combinations(num_modules: int, hotness_values: list[int]) -> np.ndarray:
    """All binary module-combination patterns [C, num_modules] with the given
    hotness values, in deterministic (hotness, lexicographic) order."""
    patterns = [
        np.array([1 if m in combo else 0 for m in range(num_modules)], dtype=np.int8)
        for k in sorted(hotness_values)
        for combo in combinations(range(num_modules), k)
    ]
    return np.stack(patterns)


class HyperTeacher:
    """Task family: fixed dimensions plus the enumerated combination patterns.

    ``max_hotness`` bounds the enumerated patterns (combinatorial in num_modules).
    """

    def __init__(self, cfg: TeacherConfig, max_hotness: int) -> None:
        self.cfg = cfg
        self.max_hotness = max_hotness
        self.combos: dict[int, np.ndarray] = {
            k: make_combinations(cfg.num_modules, [k]) for k in range(1, max_hotness + 1)
        }

    def sample_pattern(self, rng: np.random.Generator, hotness: int) -> np.ndarray:
        """Sample a binary combination pattern of the given hotness uniformly."""
        candidates = self.combos[hotness]
        return candidates[rng.integers(len(candidates))]

    def apply_weighting(self, rng: np.random.Generator, pattern: np.ndarray) -> np.ndarray:
        """Turn a binary pattern into a weighted task latent per the configured
        weighting mode (semantics identical to the source's task_distribution)."""
        pattern_f = pattern.astype(np.float32)
        match self.cfg.weighting:
            case "binary":
                return pattern_f
            case "discrete":
                weights = rng.choice(_DISCRETE_WEIGHT_VALUES, size=pattern.shape)
                return (weights * pattern_f).astype(np.float32)
            case "continuous":
                # Uniform-on-simplex weights (Willms, 2021), then actives shifted
                # to [0.5, 1.0] to prevent further sparsity.
                weights = rng.exponential(size=pattern.shape).astype(np.float32) * pattern_f
                weights = weights / (weights.sum() + 1.0)
                return ((0.5 * weights + 0.5) * pattern_f).astype(np.float32)
            case _:
                raise ValueError(f"unknown weighting: {self.cfg.weighting}")
