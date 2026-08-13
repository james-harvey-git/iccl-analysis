"""Prior-matched random features for the one-hidden-layer HyperTeacher."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from iccl.data.teacher import sample_truncated_normal


def torch_dtype(name: str) -> torch.dtype:
    """Resolve the floating-point dtypes supported by the observer."""
    match name:
        case "float64" | "double":
            return torch.float64
        case "float32" | "single":
            return torch.float32
        case _:
            raise ValueError(f"unsupported structured-observer dtype: {name}")


def validate_observer_device(
    device: str | torch.device,
    *,
    require_available: bool = True,
) -> torch.device:
    """Validate a CPU/CUDA setting and optionally require local availability."""
    resolved = torch.device(device)
    if resolved.type == "mps":
        raise ValueError("structured GP observers require device=cpu or device=cuda")
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported structured-observer device: {resolved}")
    if require_available and resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device=cuda requested but CUDA is unavailable")
    return resolved


@dataclass(frozen=True)
class FeatureBank:
    """A fixed Monte Carlo draw from the HyperTeacher hidden-unit prior.

    ``module_weights[j, m]`` and ``module_biases[j, m]`` describe module ``m``
    for random hidden unit ``j``. The shared readout has been analytically
    integrated out, leaving the factor ``sqrt(scale / num_features)``.
    """

    module_weights: Tensor  # [features, modules, input_dim]
    module_biases: Tensor  # [features, modules]
    scale: float
    seed: int

    @property
    def num_features(self) -> int:
        return int(self.module_weights.shape[0])

    @property
    def num_modules(self) -> int:
        return int(self.module_weights.shape[1])

    @property
    def input_dim(self) -> int:
        return int(self.module_weights.shape[2])

    @property
    def device(self) -> torch.device:
        return self.module_weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.module_weights.dtype

    def _as_tensor(self, value: np.ndarray | Tensor) -> Tensor:
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def features(self, x: np.ndarray | Tensor, latents: np.ndarray | Tensor) -> Tensor:
        """Evaluate features for one input under one or more task latents.

        Args:
            x: Input with shape ``[input_dim]``.
            latents: Task latents with shape ``[hypotheses, modules]`` or
                ``[modules]``.
        Returns:
            Feature matrix with shape ``[hypotheses, num_features]``.
        """
        x_t = self._as_tensor(x)
        z_t = self._as_tensor(latents)
        if x_t.shape != (self.input_dim,):
            raise ValueError(f"expected x shape {(self.input_dim,)}, got {tuple(x_t.shape)}")
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if z_t.ndim != 2 or z_t.shape[1] != self.num_modules:
            raise ValueError(
                f"expected latents shape [hypotheses, {self.num_modules}], "
                f"got {tuple(z_t.shape)}"
            )
        hotness = torch.count_nonzero(z_t, dim=-1)
        if bool((hotness == 0).any()):
            raise ValueError("task latents must activate at least one module")
        scaled_z = z_t / torch.sqrt(hotness.to(self.dtype)).unsqueeze(-1)
        module_preactivations = torch.einsum("jmd,d->jm", self.module_weights, x_t)
        module_preactivations = module_preactivations + self.module_biases
        hidden = torch.relu(scaled_z @ module_preactivations.T)
        return hidden * math.sqrt(self.scale / self.num_features)

    def features_for_history(
        self,
        inputs: np.ndarray | Tensor,
        latents: np.ndarray | Tensor,
    ) -> Tensor:
        """Evaluate aligned input/latent histories for many hypotheses.

        ``inputs`` is ``[observations, input_dim]`` and ``latents`` is either
        ``[hypotheses, observations, modules]`` or ``[observations, modules]``.
        The result is ``[hypotheses, observations, num_features]``.
        """
        x_t = self._as_tensor(inputs)
        z_t = self._as_tensor(latents)
        if x_t.ndim != 2 or x_t.shape[1] != self.input_dim:
            raise ValueError("inputs must have shape [observations, input_dim]")
        if z_t.ndim == 2:
            z_t = z_t.unsqueeze(0)
        if z_t.ndim != 3 or z_t.shape[1:] != (x_t.shape[0], self.num_modules):
            raise ValueError(
                "latents must have shape [hypotheses, observations, modules]"
            )
        hotness = torch.count_nonzero(z_t, dim=-1)
        if bool((hotness == 0).any()):
            raise ValueError("task latents must activate at least one module")
        scaled_z = z_t / torch.sqrt(hotness.to(self.dtype)).unsqueeze(-1)
        module_preactivations = torch.einsum("jmd,nd->njm", self.module_weights, x_t)
        module_preactivations = module_preactivations + self.module_biases.unsqueeze(0)
        hidden = torch.relu(torch.einsum("hnm,njm->hnj", scaled_z, module_preactivations))
        return hidden * math.sqrt(self.scale / self.num_features)

    def kernel(
        self,
        x_a: np.ndarray | Tensor,
        z_a: np.ndarray | Tensor,
        x_b: np.ndarray | Tensor,
        z_b: np.ndarray | Tensor,
    ) -> Tensor:
        """Evaluate the Monte Carlo kernel for paired or broadcast hypotheses."""
        phi_a = self.features(x_a, z_a)
        phi_b = self.features(x_b, z_b)
        if phi_a.shape[0] == 1 and phi_b.shape[0] != 1:
            phi_a = phi_a.expand(phi_b.shape[0], -1)
        if phi_b.shape[0] == 1 and phi_a.shape[0] != 1:
            phi_b = phi_b.expand(phi_a.shape[0], -1)
        if phi_a.shape != phi_b.shape:
            raise ValueError("kernel inputs must have equal or broadcastable hypothesis counts")
        return torch.sum(phi_a * phi_b, dim=-1)

    def content_hash(self) -> str:
        """Hash the feature draw for cache provenance."""
        digest = hashlib.sha256()
        digest.update(self.module_weights.detach().cpu().numpy().tobytes())
        digest.update(self.module_biases.detach().cpu().numpy().tobytes())
        digest.update(repr((self.scale, self.seed)).encode())
        return digest.hexdigest()


def sample_feature_bank(
    *,
    input_dim: int,
    num_modules: int,
    scale: float,
    num_features: int,
    seed: int,
    device: str | torch.device = "cpu",
    dtype: str | torch.dtype = "float64",
    use_bias: bool = True,
) -> FeatureBank:
    """Draw a deterministic feature bank from the actual module priors."""
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    resolved_device = validate_observer_device(device)
    resolved_dtype = torch_dtype(dtype) if isinstance(dtype, str) else dtype
    weight_std = math.sqrt(scale / input_dim)
    weights = np.empty((num_features, num_modules, input_dim), dtype=np.float32)
    biases = np.empty((num_features, num_modules), dtype=np.float32)
    for feature_index in range(num_features):
        weight_rng = np.random.Generator(
            np.random.Philox(np.random.SeedSequence([seed, feature_index, 0]))
        )
        weights[feature_index] = sample_truncated_normal(
            weight_rng,
            (num_modules, input_dim),
            weight_std,
        )
        if use_bias:
            bias_rng = np.random.Generator(
                np.random.Philox(np.random.SeedSequence([seed, feature_index, 1]))
            )
            biases[feature_index] = bias_rng.uniform(0.0, 0.5, size=num_modules)
        else:
            biases[feature_index] = 0.0
    return FeatureBank(
        module_weights=torch.as_tensor(weights, dtype=resolved_dtype, device=resolved_device),
        module_biases=torch.as_tensor(biases, dtype=resolved_dtype, device=resolved_device),
        scale=scale,
        seed=seed,
    )
