"""Numerically stable online multi-output Gaussian-process inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GPPrediction:
    """A predictive distribution plus the triangular-solve work for updating."""

    mean: Tensor  # [hypotheses, outputs]
    variance: Tensor  # [hypotheses]
    triangular_solution: Tensor  # [hypotheses, observations]


def gaussian_log_predictive_density(target: Tensor, prediction: GPPrediction) -> Tensor:
    """Independent-output Gaussian log density for every hypothesis."""
    residual = target.unsqueeze(0) - prediction.mean
    output_dim = residual.shape[-1]
    return -0.5 * (
        output_dim * torch.log(2.0 * math.pi * prediction.variance)
        + torch.sum(residual.square(), dim=-1) / prediction.variance
    )


def batch_gp_factor(
    features: Tensor,
    targets: Tensor,
    jitter: float,
) -> tuple[Tensor, Tensor]:
    """Build batched Cholesky factors and whitened targets."""
    if features.ndim != 3:
        raise ValueError("features must have shape [hypotheses, observations, features]")
    hypotheses, observations, _ = features.shape
    if targets.ndim != 2 or targets.shape[0] != observations:
        raise ValueError("targets must have shape [observations, outputs]")
    if observations == 0:
        return (
            features.new_zeros((hypotheses, 0, 0)),
            features.new_zeros((hypotheses, 0, targets.shape[1])),
        )
    kernel = features @ features.transpose(-1, -2)
    identity = torch.eye(observations, dtype=features.dtype, device=features.device)
    cholesky = torch.linalg.cholesky(kernel + jitter * identity.unsqueeze(0))
    expanded_targets = targets.unsqueeze(0).expand(hypotheses, -1, -1)
    whitened_targets = torch.linalg.solve_triangular(
        cholesky,
        expanded_targets,
        upper=False,
    )
    return cholesky, whitened_targets


def batch_log_marginal_likelihood(features: Tensor, targets: Tensor, jitter: float) -> Tensor:
    """Exact GP log marginal likelihood for each feature-history hypothesis."""
    cholesky, whitened_targets = batch_gp_factor(features, targets, jitter)
    observations = features.shape[1]
    output_dim = targets.shape[1]
    if observations == 0:
        return features.new_zeros(features.shape[0])
    data_fit = torch.sum(whitened_targets.square(), dim=(-2, -1))
    log_determinant_term = output_dim * torch.sum(
        torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)),
        dim=-1,
    )
    normalizer = 0.5 * observations * output_dim * math.log(2.0 * math.pi)
    return -0.5 * data_fit - log_determinant_term - normalizer


class BatchedOnlineGP:
    """Independent-output GPs sharing a kernel, updated by Cholesky extension."""

    def __init__(
        self,
        *,
        num_hypotheses: int,
        num_features: int,
        output_dim: int,
        max_observations: int,
        jitter: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if num_hypotheses <= 0 or num_features <= 0 or output_dim <= 0:
            raise ValueError("GP dimensions must be positive")
        if max_observations <= 0:
            raise ValueError("max_observations must be positive")
        if jitter <= 0.0:
            raise ValueError("jitter must be positive")
        self.num_hypotheses = num_hypotheses
        self.num_features = num_features
        self.output_dim = output_dim
        self.max_observations = max_observations
        self.jitter = float(jitter)
        self.features = torch.zeros(
            (num_hypotheses, max_observations, num_features),
            dtype=dtype,
            device=device,
        )
        self.cholesky = torch.zeros(
            (num_hypotheses, max_observations, max_observations),
            dtype=dtype,
            device=device,
        )
        self.whitened_targets = torch.zeros(
            (num_hypotheses, max_observations, output_dim),
            dtype=dtype,
            device=device,
        )
        self.targets = torch.zeros((max_observations, output_dim), dtype=dtype, device=device)
        self.num_observations = 0

    @property
    def device(self) -> torch.device:
        return self.features.device

    @property
    def dtype(self) -> torch.dtype:
        return self.features.dtype

    def predict(self, features: Tensor) -> GPPrediction:
        """Predict before observing the corresponding output."""
        if features.shape != (self.num_hypotheses, self.num_features):
            raise ValueError(
                f"expected feature shape {(self.num_hypotheses, self.num_features)}, "
                f"got {tuple(features.shape)}"
            )
        kernel_diagonal = torch.sum(features.square(), dim=-1) + self.jitter
        n = self.num_observations
        if n == 0:
            return GPPrediction(
                mean=features.new_zeros((self.num_hypotheses, self.output_dim)),
                variance=kernel_diagonal,
                triangular_solution=features.new_zeros((self.num_hypotheses, 0)),
            )
        cross_kernel = torch.einsum("hnj,hj->hn", self.features[:, :n], features)
        triangular_solution = torch.linalg.solve_triangular(
            self.cholesky[:, :n, :n],
            cross_kernel.unsqueeze(-1),
            upper=False,
        ).squeeze(-1)
        mean = torch.einsum(
            "hn,hno->ho",
            triangular_solution,
            self.whitened_targets[:, :n],
        )
        variance = kernel_diagonal - torch.sum(triangular_solution.square(), dim=-1)
        return GPPrediction(
            mean=mean,
            variance=variance,
            triangular_solution=triangular_solution,
        )

    def append(self, features: Tensor, target: Tensor, prediction: GPPrediction) -> None:
        """Extend every hypothesis with a shared observed output."""
        n = self.num_observations
        if n >= self.max_observations:
            self._grow_capacity(max(1, 2 * self.max_observations))
        if target.shape != (self.output_dim,):
            raise ValueError(
                f"expected target shape {(self.output_dim,)}, got {tuple(target.shape)}"
            )
        if not bool(torch.isfinite(prediction.variance).all()):
            raise RuntimeError("non-finite GP predictive variance")
        tolerance = torch.finfo(self.dtype).eps * 100.0
        if bool((prediction.variance <= tolerance).any()):
            raise RuntimeError("non-positive GP predictive variance")
        diagonal = torch.sqrt(prediction.variance)
        self.features[:, n] = features
        if n:
            self.cholesky[:, n, :n] = prediction.triangular_solution
        self.cholesky[:, n, n] = diagonal
        self.whitened_targets[:, n] = (
            target.unsqueeze(0) - prediction.mean
        ) / diagonal.unsqueeze(-1)
        self.targets[n] = target
        self.num_observations += 1

    def _grow_capacity(self, capacity: int) -> None:
        """Grow storage geometrically without exposing future demonstration counts."""
        if capacity <= self.max_observations:
            return
        feature_storage = self.features.new_zeros(
            (self.num_hypotheses, capacity, self.num_features)
        )
        cholesky_storage = self.cholesky.new_zeros(
            (self.num_hypotheses, capacity, capacity)
        )
        whitened_storage = self.whitened_targets.new_zeros(
            (self.num_hypotheses, capacity, self.output_dim)
        )
        target_storage = self.targets.new_zeros((capacity, self.output_dim))
        old_capacity = self.max_observations
        feature_storage[:, :old_capacity] = self.features
        cholesky_storage[:, :old_capacity, :old_capacity] = self.cholesky
        whitened_storage[:, :old_capacity] = self.whitened_targets
        target_storage[:old_capacity] = self.targets
        self.features = feature_storage
        self.cholesky = cholesky_storage
        self.whitened_targets = whitened_storage
        self.targets = target_storage
        self.max_observations = capacity

    def predict_and_append(self, features: Tensor, target: Tensor) -> tuple[GPPrediction, Tensor]:
        """Return the causal prediction and its log density, then update state."""
        prediction = self.predict(features)
        log_density = gaussian_log_predictive_density(target, prediction)
        self.append(features, target, prediction)
        return prediction, log_density

    def rebuild(self, jitter: float | None = None) -> None:
        """Refactor the accumulated kernel, optionally at a larger jitter."""
        if jitter is not None:
            if jitter < self.jitter:
                raise ValueError("jitter escalation cannot decrease jitter")
            self.jitter = float(jitter)
        n = self.num_observations
        if n == 0:
            return
        cholesky, whitened = batch_gp_factor(
            self.features[:, :n],
            self.targets[:n],
            self.jitter,
        )
        self.cholesky[:, :n, :n] = cholesky
        self.whitened_targets[:, :n] = whitened

    def resample(self, indices: Tensor) -> None:
        """Select hypotheses with replacement while retaining GP histories."""
        if indices.shape != (self.num_hypotheses,):
            raise ValueError("resampling indices must match the hypothesis count")
        self.features = self.features.index_select(0, indices)
        self.cholesky = self.cholesky.index_select(0, indices)
        self.whitened_targets = self.whitened_targets.index_select(0, indices)

    def replace_hypothesis(self, index: int, features: Tensor) -> None:
        """Replace one hypothesis and rebuild its factors from observed targets."""
        n = self.num_observations
        if features.shape != (n, self.num_features):
            raise ValueError(f"expected replacement features shape {(n, self.num_features)}")
        cholesky, whitened = batch_gp_factor(
            features.unsqueeze(0),
            self.targets[:n],
            self.jitter,
        )
        self.features[index, :n] = features
        self.cholesky[index, :n, :n] = cholesky[0]
        self.whitened_targets[index, :n] = whitened[0]

    def log_marginal_likelihood(self) -> Tensor:
        """Return the accumulated data evidence for every hypothesis."""
        n = self.num_observations
        return batch_log_marginal_likelihood(
            self.features[:, :n],
            self.targets[:n],
            self.jitter,
        )
