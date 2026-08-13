"""Shared observer protocol and the exact current-task latent mixture."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from iccl.analysis.structured_observer.gp import (
    BatchedOnlineGP,
    GPPrediction,
    gaussian_log_predictive_density,
)
from iccl.analysis.structured_observer.kernel import FeatureBank
from iccl.analysis.structured_observer.schedule import ScheduleConfig, canonical_task_classes


@dataclass(frozen=True)
class ObserverPrediction:
    """A causal posterior-predictive summary before the current output."""

    mean: np.ndarray
    covariance_trace: float
    effective_sample_size: float
    max_weight: float
    unique_prefixes: int
    log_evidence: float
    relative_jitter: float


@dataclass(frozen=True)
class ObserverUpdate:
    """Posterior diagnostics after incorporating one revealed output."""

    effective_sample_size: float
    max_weight: float
    resampled: bool
    unique_prefixes: int
    log_evidence: float
    relative_jitter: float
    rejuvenation_acceptance: float
    completion_attempts: int


@dataclass(frozen=True)
class TaskEndDiagnostics:
    """Diagnostics for the resample-move operation at a completed boundary."""

    rejuvenation_acceptance: float
    completion_attempts: int
    resampling_events: int
    unique_prefixes: int


def normalize_log_weights(log_weights: Tensor) -> tuple[Tensor, Tensor]:
    """Normalize log weights and return their log normalizing constant."""
    log_normalizer = torch.logsumexp(log_weights, dim=0)
    return log_weights - log_normalizer, log_normalizer


def weight_diagnostics(log_weights: Tensor) -> tuple[float, float]:
    """Return effective sample size and largest normalized weight."""
    weights = torch.exp(log_weights)
    ess = 1.0 / float(torch.sum(weights.square()).item())
    return ess, float(weights.max().item())


def mixture_prediction(
    component_prediction: GPPrediction,
    log_weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Mean and covariance trace of a weighted independent-output GP mixture."""
    weights = torch.exp(log_weights)
    mean = torch.sum(weights.unsqueeze(-1) * component_prediction.mean, dim=0)
    between_component = torch.sum(
        (component_prediction.mean - mean.unsqueeze(0)).square(),
        dim=-1,
    )
    trace_by_component = (
        component_prediction.mean.shape[-1] * component_prediction.variance
        + between_component
    )
    covariance_trace = torch.sum(weights * trace_by_component)
    return mean, covariance_trace


def stable_gp_prediction(
    gp: BatchedOnlineGP,
    features: Tensor,
    *,
    maximum_jitter: float,
) -> GPPrediction:
    """Predict with deterministic decade-wise jitter escalation when needed."""
    while True:
        prediction = gp.predict(features)
        tolerance = torch.finfo(gp.dtype).eps * 100.0
        valid = bool(torch.isfinite(prediction.variance).all()) and bool(
            (prediction.variance > tolerance).all()
        )
        if valid:
            return prediction
        next_jitter = gp.jitter * 10.0
        if next_jitter > maximum_jitter * (1.0 + 1e-12):
            raise RuntimeError(
                f"GP jitter would exceed configured maximum {maximum_jitter:.8g}"
            )
        gp.rebuild(next_jitter)


class CurrentTaskObserver:
    """Exact mixture over exchangeable current-task coefficient classes."""

    def __init__(
        self,
        *,
        feature_bank: FeatureBank,
        schedule_config: ScheduleConfig,
        output_dim: int,
        relative_jitter: float,
        max_relative_jitter: float,
        initial_capacity: int = 8,
    ) -> None:
        if not 0.0 < relative_jitter <= max_relative_jitter:
            raise ValueError("relative jitter must be positive and at most its maximum")
        latents, log_prior = canonical_task_classes(schedule_config)
        self.feature_bank = feature_bank
        self.latents = torch.as_tensor(
            latents,
            dtype=feature_bank.dtype,
            device=feature_bank.device,
        )
        self.log_prior = torch.as_tensor(
            log_prior,
            dtype=feature_bank.dtype,
            device=feature_bank.device,
        )
        self.output_dim = output_dim
        self.relative_jitter_setting = relative_jitter
        self.max_relative_jitter_setting = max_relative_jitter
        self.initial_capacity = initial_capacity
        self.gp: BatchedOnlineGP | None = None
        self.log_weights = self.log_prior.clone()
        self.log_evidence = 0.0
        self.reference_diagonal = math.nan
        self.pending_features: Tensor | None = None
        self.pending_prediction: GPPrediction | None = None

    def start_task(self) -> None:
        """Discard every statistic learned from the preceding task."""
        if self.pending_features is not None:
            raise RuntimeError("cannot reset with a pending demonstration output")
        self.gp = None
        self.log_weights = self.log_prior.clone()
        self.log_evidence = 0.0
        self.reference_diagonal = math.nan

    def end_task(self) -> TaskEndDiagnostics:
        """Validate that the task ends after a complete demonstration."""
        if self.pending_features is not None:
            raise RuntimeError("task ended with a pending demonstration output")
        return TaskEndDiagnostics(
            rejuvenation_acceptance=math.nan,
            completion_attempts=0,
            resampling_events=0,
            unique_prefixes=len(self.latents),
        )

    def _initialize_gp(self, features: Tensor) -> BatchedOnlineGP:
        reference = float(torch.mean(torch.sum(features.square(), dim=-1)).item())
        if not math.isfinite(reference) or reference <= 0.0:
            raise RuntimeError("random-feature kernel has a non-positive reference diagonal")
        self.reference_diagonal = reference
        return BatchedOnlineGP(
            num_hypotheses=len(self.latents),
            num_features=self.feature_bank.num_features,
            output_dim=self.output_dim,
            max_observations=self.initial_capacity,
            jitter=self.relative_jitter_setting * reference,
            device=self.feature_bank.device,
            dtype=self.feature_bank.dtype,
        )

    def predict(self, x: np.ndarray) -> ObserverPrediction:
        """Form the posterior mean before the corresponding output is visible."""
        if self.pending_features is not None:
            raise RuntimeError("predict called twice without observing an output")
        features = self.feature_bank.features(x, self.latents)
        if self.gp is None:
            self.gp = self._initialize_gp(features)
        prediction = stable_gp_prediction(
            self.gp,
            features,
            maximum_jitter=self.max_relative_jitter_setting * self.reference_diagonal,
        )
        mean, covariance_trace = mixture_prediction(prediction, self.log_weights)
        ess, max_weight = weight_diagnostics(self.log_weights)
        self.pending_features = features
        self.pending_prediction = prediction
        return ObserverPrediction(
            mean=mean.detach().cpu().numpy(),
            covariance_trace=float(covariance_trace.item()),
            effective_sample_size=ess,
            max_weight=max_weight,
            unique_prefixes=len(self.latents),
            log_evidence=self.log_evidence,
            relative_jitter=self.gp.jitter / self.reference_diagonal,
        )

    def observe(self, y: np.ndarray) -> ObserverUpdate:
        """Update class probabilities using the revealed output's GP density."""
        if self.gp is None or self.pending_features is None or self.pending_prediction is None:
            raise RuntimeError("observe called without a pending prediction")
        target = torch.as_tensor(y, dtype=self.gp.dtype, device=self.gp.device)
        log_density = gaussian_log_predictive_density(target, self.pending_prediction)
        self.log_weights, log_increment = normalize_log_weights(
            self.log_weights + log_density
        )
        self.log_evidence += float(log_increment.item())
        self.gp.append(self.pending_features, target, self.pending_prediction)
        self.pending_features = None
        self.pending_prediction = None
        ess, max_weight = weight_diagnostics(self.log_weights)
        return ObserverUpdate(
            effective_sample_size=ess,
            max_weight=max_weight,
            resampled=False,
            unique_prefixes=len(self.latents),
            log_evidence=self.log_evidence,
            relative_jitter=self.gp.jitter / self.reference_diagonal,
            rejuvenation_acceptance=math.nan,
            completion_attempts=0,
        )
