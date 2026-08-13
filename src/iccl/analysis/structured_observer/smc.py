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
    extend_gp_factor,
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
    ConditionedSchedulePrior,
    ScheduleConfig,
    canonicalize_schedule_prefix,
    conditioned_schedule_prior,
    prefix_key,
    systematic_resampling_indices,
)


@dataclass(frozen=True)
class SMCConfig:
    """Particle count, resampling threshold, and rejuvenation controls."""

    num_particles: int
    ess_fraction: float
    task_end_rejuvenation_sweeps: int
    initial_gp_capacity: int = 8
    proposal_chunk_size: int = 128


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
        if smc_config.proposal_chunk_size <= 0:
            raise ValueError("proposal_chunk_size must be positive")
        if not 0.0 < relative_jitter <= max_relative_jitter:
            raise ValueError("relative jitter must be positive and at most its maximum")
        self.feature_bank = feature_bank
        self.schedule_config = schedule_config
        self.output_dim = output_dim
        self.relative_jitter_setting = relative_jitter
        self.max_relative_jitter_setting = max_relative_jitter
        self.smc_config = smc_config
        self.rng = np.random.Generator(np.random.Philox(seed))
        self.conditioned_prior: ConditionedSchedulePrior = conditioned_schedule_prior(
            schedule_config
        )
        self._fixed_initial_schedules = initial_schedules is not None
        if initial_schedules is None:
            self.schedules = np.zeros(
                (
                    smc_config.num_particles,
                    schedule_config.num_tasks,
                    schedule_config.num_modules,
                ),
                dtype=np.float64,
            )
            self.completion_attempts = 0
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
        self.module_preactivation_history: list[Tensor] = []
        self.task_start_observation = 0
        self.pending_features: Tensor | None = None
        self.pending_prediction: GPPrediction | None = None
        self.last_rejuvenation_acceptance = math.nan
        self.last_rejuvenation_unique_proposals = 0
        self.resampling_events = 0

    def start_task(self) -> None:
        """Advance the causal prefix and relabel particles from that prefix alone."""
        if self.pending_features is not None:
            raise RuntimeError("cannot start a task with a pending output")
        self.task_index += 1
        if self.task_index >= self.schedule_config.num_tasks:
            raise RuntimeError("observer received more tasks than its schedule prior")
        if not self._fixed_initial_schedules:
            self.schedules[:, self.task_index] = self.conditioned_prior.sample_next_batch(
                self.rng,
                self.schedules[:, : self.task_index],
            )
        observed_tasks = self.task_index + 1
        self.schedules = np.stack(
            [
                canonicalize_schedule_prefix(schedule, observed_tasks)
                for schedule in self.schedules
            ]
        )
        self.task_start_observation = 0 if self.gp is None else self.gp.num_observations

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
        module_preactivations = self.feature_bank.module_preactivations(x)
        features = self.feature_bank.features_from_module_preactivations(
            module_preactivations,
            self._current_latents(),
        )
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
        self.module_preactivation_history.append(module_preactivations)
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

    def _conditional_task_log_likelihood(self) -> np.ndarray:
        """Return each particle's current-task evidence given its fixed past."""
        if self.gp is None:
            raise RuntimeError("cannot evaluate an uninitialized GP")
        n = self.gp.num_observations
        start = self.task_start_observation
        full = log_marginal_likelihood_from_factor(
            self.gp.cholesky[:, :n, :n],
            self.gp.whitened_targets[:, :n],
        )
        past = log_marginal_likelihood_from_factor(
            self.gp.cholesky[:, :start, :start],
            self.gp.whitened_targets[:, :start],
        )
        return (full - past).detach().cpu().numpy()

    def _sample_current_task_proposals(self) -> np.ndarray:
        """Draw current tasks from the exact prior conditional on each past prefix."""
        candidates = self.schedules.copy()
        candidates[:, self.task_index + 1 :] = 0.0
        candidates[:, self.task_index] = self.conditioned_prior.sample_next_batch(
            self.rng,
            self.schedules[:, : self.task_index],
        )
        return np.stack(
            [
                canonicalize_schedule_prefix(schedule, self.task_index + 1)
                for schedule in candidates
            ]
        )

    def _rejuvenate(self, sweeps: int) -> float:
        """Apply current-task prior proposals with conditional GP-evidence ratios."""
        if self.gp is None or sweeps == 0 or self.gp.num_observations == 0:
            return math.nan
        accepted = 0
        proposed = 0
        current_log_likelihoods = self._conditional_task_log_likelihood()
        n = self.gp.num_observations
        block_start = self.task_start_observation
        block_preactivations = torch.stack(
            self.module_preactivation_history[block_start:n]
        )
        block_targets = self.gp.targets[block_start:n]
        chunk_size = self.smc_config.proposal_chunk_size
        unique_proposals = 0
        for _ in range(sweeps):
            candidate_schedules = self._sample_current_task_proposals()
            proposal_groups: dict[bytes, list[int]] = {}
            for particle, schedule in enumerate(candidate_schedules):
                key = prefix_key(schedule, self.task_index + 1)
                proposal_groups.setdefault(key, []).append(particle)
            groups = list(proposal_groups.values())
            unique_proposals += len(groups)
            for chunk_start in range(0, len(groups), chunk_size):
                chunk_end = min(
                    chunk_start + chunk_size,
                    len(groups),
                )
                chunk_groups = groups[chunk_start:chunk_end]
                representatives = np.asarray(
                    [group[0] for group in chunk_groups],
                    dtype=np.int64,
                )
                representative_tensor = torch.as_tensor(
                    representatives,
                    dtype=torch.long,
                    device=self.feature_bank.device,
                )
                candidate_latents = candidate_schedules[
                    representatives,
                    self.task_index,
                ]
                candidate_features = (
                    self.feature_bank.features_from_module_preactivations(
                        block_preactivations,
                        candidate_latents,
                    )
                )
                candidate_cholesky, candidate_whitened, candidate_likelihood = (
                    extend_gp_factor(
                        self.gp.features[
                            :, :block_start
                        ].index_select(0, representative_tensor),
                        self.gp.cholesky[
                            :, :block_start, :block_start
                        ].index_select(0, representative_tensor),
                        self.gp.whitened_targets[
                            :, :block_start
                        ].index_select(0, representative_tensor),
                        candidate_features,
                        block_targets,
                        self.gp.jitter,
                    )
                )
                candidate_values = candidate_likelihood.detach().cpu().numpy()
                for local_index, particles in enumerate(chunk_groups):
                    particle_indices = np.asarray(particles, dtype=np.int64)
                    log_acceptance = (
                        candidate_values[local_index]
                        - current_log_likelihoods[particle_indices]
                    )
                    accept = np.log(self.rng.random(len(particles))) < np.minimum(
                        0.0,
                        log_acceptance,
                    )
                    accepted += int(accept.sum())
                    proposed += len(particles)
                    if not bool(accept.any()):
                        continue
                    global_indices = particle_indices[accept]
                    self.schedules[global_indices] = candidate_schedules[global_indices]
                    global_tensor = torch.as_tensor(
                        global_indices,
                        dtype=torch.long,
                        device=self.feature_bank.device,
                    )
                    repeats = len(global_indices)
                    self.gp.replace_extended_block(
                        global_tensor,
                        block_start,
                        candidate_features[local_index : local_index + 1].expand(
                            repeats, -1, -1
                        ),
                        candidate_cholesky[local_index : local_index + 1].expand(
                            repeats, -1, -1
                        ),
                        candidate_whitened[local_index : local_index + 1].expand(
                            repeats, -1, -1
                        ),
                    )
                    current_log_likelihoods[global_indices] = candidate_values[
                        local_index
                    ]
        self.last_rejuvenation_unique_proposals = unique_proposals
        return accepted / proposed if proposed else math.nan

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
        """Resample and rejuvenate the current task before the next boundary."""
        if self.pending_features is not None:
            raise RuntimeError("task ended with a pending demonstration output")
        if self.gp is None:
            raise RuntimeError("task ended without demonstrations")
        self._resample()
        self.last_rejuvenation_acceptance = self._rejuvenate(
            self.smc_config.task_end_rejuvenation_sweeps
        )
        return TaskEndDiagnostics(
            rejuvenation_acceptance=self.last_rejuvenation_acceptance,
            completion_attempts=self.completion_attempts,
            resampling_events=self.resampling_events,
            unique_prefixes=self._unique_prefix_count(),
        )
