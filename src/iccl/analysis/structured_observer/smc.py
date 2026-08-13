"""Full-history sequential Monte Carlo over valid latent task schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from iccl.analysis.structured_observer.gp import (
    BatchedOnlineGP,
    GPPrediction,
    batch_gp_factor,
    gaussian_log_predictive_density,
    log_marginal_likelihood_from_factor,
)
from iccl.analysis.structured_observer.kernel import FeatureBank
from iccl.analysis.structured_observer.observer import (
    ObserverPrediction,
    ObserverUpdate,
    TaskEndDiagnostics,
    mixture_prediction,
    normalize_log_weights,
    stable_gp_prediction,
    weight_diagnostics,
)
from iccl.analysis.structured_observer.schedule import (
    ScheduleConfig,
    canonicalize_schedule_prefix,
    prefix_key,
    sample_conditional_completion,
    sample_valid_schedule,
    systematic_resampling_indices,
)


@dataclass(frozen=True)
class SMCConfig:
    """Particle count, resampling threshold, and rejuvenation controls."""

    num_particles: int
    ess_fraction: float
    task_end_rejuvenation_sweeps: int
    max_completion_attempts: int
    initial_gp_capacity: int = 8


class FullHistoryObserver:
    """Particle approximation to the valid-schedule posterior using all tasks."""

    def __init__(
        self,
        *,
        feature_bank: FeatureBank,
        schedule_config: ScheduleConfig,
        output_dim: int,
        relative_jitter: float,
        max_relative_jitter: float,
        smc_config: SMCConfig,
        seed: int,
        initial_schedules: np.ndarray | None = None,
    ) -> None:
        if smc_config.num_particles <= 0:
            raise ValueError("num_particles must be positive")
        if not 0.0 < smc_config.ess_fraction <= 1.0:
            raise ValueError("ess_fraction must be in (0, 1]")
        if smc_config.task_end_rejuvenation_sweeps < 0:
            raise ValueError("task-end rejuvenation sweeps cannot be negative")
        if not 0.0 < relative_jitter <= max_relative_jitter:
            raise ValueError("relative jitter must be positive and at most its maximum")
        self.feature_bank = feature_bank
        self.schedule_config = schedule_config
        self.output_dim = output_dim
        self.relative_jitter_setting = relative_jitter
        self.max_relative_jitter_setting = max_relative_jitter
        self.smc_config = smc_config
        self.rng = np.random.Generator(np.random.Philox(seed))
        if initial_schedules is None:
            sampled = [
                sample_valid_schedule(
                    self.rng,
                    schedule_config,
                    max_attempts=smc_config.max_completion_attempts,
                )
                for _ in range(smc_config.num_particles)
            ]
            self.schedules = np.stack([item[0] for item in sampled])
            self.completion_attempts = sum(item[1] for item in sampled)
        else:
            expected_shape = (
                smc_config.num_particles,
                schedule_config.num_tasks,
                schedule_config.num_modules,
            )
            if initial_schedules.shape != expected_shape:
                raise ValueError(
                    f"expected initial_schedules shape {expected_shape}, "
                    f"got {initial_schedules.shape}"
                )
            self.schedules = initial_schedules.astype(np.float64, copy=True)
            self.completion_attempts = 0
        self.log_weights = torch.full(
            (smc_config.num_particles,),
            -math.log(smc_config.num_particles),
            dtype=feature_bank.dtype,
            device=feature_bank.device,
        )
        self.gp: BatchedOnlineGP | None = None
        self.reference_diagonal = math.nan
        self.log_evidence = 0.0
        self.task_index = -1
        self.input_history: list[np.ndarray] = []
        self.task_history: list[int] = []
        self.pending_features: Tensor | None = None
        self.pending_prediction: GPPrediction | None = None
        self.last_rejuvenation_acceptance = math.nan
        self.resampling_events = 0

    def start_task(self) -> None:
        """Advance the causal prefix and relabel particles from that prefix alone."""
        if self.pending_features is not None:
            raise RuntimeError("cannot start a task with a pending output")
        self.task_index += 1
        if self.task_index >= self.schedule_config.num_tasks:
            raise RuntimeError("observer received more tasks than its schedule prior")
        observed_tasks = self.task_index + 1
        self.schedules = np.stack(
            [
                canonicalize_schedule_prefix(schedule, observed_tasks)
                for schedule in self.schedules
            ]
        )

    def _current_latents(self) -> Tensor:
        return torch.as_tensor(
            self.schedules[:, self.task_index],
            dtype=self.feature_bank.dtype,
            device=self.feature_bank.device,
        )

    def _initialize_gp(self, features: Tensor) -> BatchedOnlineGP:
        reference = float(torch.mean(torch.sum(features.square(), dim=-1)).item())
        if not math.isfinite(reference) or reference <= 0.0:
            raise RuntimeError("random-feature kernel has a non-positive reference diagonal")
        self.reference_diagonal = reference
        return BatchedOnlineGP(
            num_hypotheses=self.smc_config.num_particles,
            num_features=self.feature_bank.num_features,
            output_dim=self.output_dim,
            max_observations=self.smc_config.initial_gp_capacity,
            jitter=self.relative_jitter_setting * reference,
            device=self.feature_bank.device,
            dtype=self.feature_bank.dtype,
        )

    def _unique_prefix_count(self) -> int:
        observed_tasks = self.task_index + 1
        return len(
            {prefix_key(schedule, observed_tasks) for schedule in self.schedules}
        )

    def predict(self, x: np.ndarray) -> ObserverPrediction:
        """Average particle GP predictions before the output is available."""
        if self.task_index < 0:
            raise RuntimeError("predict called before the first task boundary")
        if self.pending_features is not None:
            raise RuntimeError("predict called twice without observing an output")
        features = self.feature_bank.features(x, self._current_latents())
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
        self.input_history.append(np.asarray(x, dtype=np.float64).copy())
        self.task_history.append(self.task_index)
        return ObserverPrediction(
            mean=mean.detach().cpu().numpy(),
            covariance_trace=float(covariance_trace.item()),
            effective_sample_size=ess,
            max_weight=max_weight,
            unique_prefixes=self._unique_prefix_count(),
            log_evidence=self.log_evidence,
            relative_jitter=self.gp.jitter / self.reference_diagonal,
        )

    def _resample(self) -> None:
        if self.gp is None:
            raise RuntimeError("cannot resample an uninitialized GP")
        weights = torch.exp(self.log_weights).detach().cpu().numpy()
        ancestors = systematic_resampling_indices(weights, self.rng)
        self.schedules = self.schedules[ancestors].copy()
        indices = torch.as_tensor(ancestors, dtype=torch.long, device=self.feature_bank.device)
        self.gp.resample(indices)
        self.log_weights.fill_(-math.log(self.smc_config.num_particles))
        self.resampling_events += 1

    def _history_features(self, schedules: np.ndarray) -> Tensor:
        if not self.input_history:
            return self.feature_bank.module_weights.new_zeros(
                (len(schedules), 0, self.feature_bank.num_features)
            )
        inputs = np.stack(self.input_history)
        latents = np.stack(
            [schedules[:, index] for index in self.task_history],
            axis=1,
        )
        return self.feature_bank.features_for_history(inputs, latents)

    def _rejuvenate(self, sweeps: int) -> float:
        """Apply conditional-prior tail proposals with GP-evidence MH ratios."""
        if self.gp is None or sweeps == 0 or self.gp.num_observations == 0:
            return math.nan
        accepted = 0
        proposed = 0
        current_log_likelihoods = self.gp.log_marginal_likelihood().detach().cpu().numpy()
        fixed_prefix_length = self.task_index
        for _ in range(sweeps):
            candidates: list[np.ndarray] = []
            for particle in range(self.smc_config.num_particles):
                prefix = self.schedules[particle, :fixed_prefix_length]
                candidate, attempts = sample_conditional_completion(
                    self.rng,
                    prefix,
                    self.schedule_config,
                    max_attempts=self.smc_config.max_completion_attempts,
                )
                self.completion_attempts += attempts
                candidates.append(
                    canonicalize_schedule_prefix(candidate, self.task_index + 1)
                )
            candidate_schedules = np.stack(candidates)
            candidate_features = self._history_features(candidate_schedules)
            candidate_cholesky, candidate_whitened = batch_gp_factor(
                candidate_features,
                self.gp.targets[: self.gp.num_observations],
                self.gp.jitter,
            )
            candidate_log_likelihoods = (
                log_marginal_likelihood_from_factor(
                    candidate_cholesky,
                    candidate_whitened,
                )
                .detach()
                .cpu()
                .numpy()
            )
            log_acceptance = candidate_log_likelihoods - current_log_likelihoods
            accept = np.log(self.rng.random(self.smc_config.num_particles)) < np.minimum(
                0.0,
                log_acceptance,
            )
            accepted += int(accept.sum())
            proposed += self.smc_config.num_particles
            if bool(accept.any()):
                self.schedules[accept] = candidate_schedules[accept]
                mask = torch.as_tensor(
                    accept,
                    dtype=torch.bool,
                    device=self.feature_bank.device,
                )
                self.gp.replace_hypotheses(
                    mask,
                    candidate_features,
                    candidate_cholesky,
                    candidate_whitened,
                )
                current_log_likelihoods[accept] = candidate_log_likelihoods[accept]
        return accepted / proposed if proposed else math.nan

    def _refresh_future_tails(self) -> None:
        """Redraw unobserved suffixes from their conditional prior."""
        prefix_length = self.task_index + 1
        if prefix_length == self.schedule_config.num_tasks:
            return
        refreshed: list[np.ndarray] = []
        for schedule in self.schedules:
            candidate, attempts = sample_conditional_completion(
                self.rng,
                schedule[:prefix_length],
                self.schedule_config,
                max_attempts=self.smc_config.max_completion_attempts,
            )
            self.completion_attempts += attempts
            refreshed.append(canonicalize_schedule_prefix(candidate, prefix_length))
        self.schedules = np.stack(refreshed)

    def observe(self, y: np.ndarray) -> ObserverUpdate:
        """Reweight particles by the revealed output and resample when ESS is low."""
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

        ess, _ = weight_diagnostics(self.log_weights)
        resampled = ess < self.smc_config.ess_fraction * self.smc_config.num_particles
        acceptance = math.nan
        if resampled:
            self._resample()
            acceptance = self._rejuvenate(1)
        self.last_rejuvenation_acceptance = acceptance
        ess, max_weight = weight_diagnostics(self.log_weights)
        return ObserverUpdate(
            effective_sample_size=ess,
            max_weight=max_weight,
            resampled=resampled,
            unique_prefixes=self._unique_prefix_count(),
            log_evidence=self.log_evidence,
            relative_jitter=self.gp.jitter / self.reference_diagonal,
            rejuvenation_acceptance=acceptance,
            completion_attempts=self.completion_attempts,
        )

    def end_task(self) -> TaskEndDiagnostics:
        """Resample-move the observed tail, then refresh only unobserved tasks."""
        if self.pending_features is not None:
            raise RuntimeError("task ended with a pending demonstration output")
        if self.gp is None:
            raise RuntimeError("task ended without demonstrations")
        self._resample()
        self.last_rejuvenation_acceptance = self._rejuvenate(
            self.smc_config.task_end_rejuvenation_sweeps
        )
        self._refresh_future_tails()
        return TaskEndDiagnostics(
            rejuvenation_acceptance=self.last_rejuvenation_acceptance,
            completion_attempts=self.completion_attempts,
            resampling_events=self.resampling_events,
            unique_prefixes=self._unique_prefix_count(),
        )
