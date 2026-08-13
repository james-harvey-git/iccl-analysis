from itertools import product

import numpy as np
import torch

from iccl.analysis.structured_observer.gp import GPPrediction, gaussian_log_predictive_density
from iccl.analysis.structured_observer.kernel import sample_feature_bank
from iccl.analysis.structured_observer.observer import (
    CurrentTaskObserver,
    normalize_log_weights,
)
from iccl.analysis.structured_observer.schedule import (
    ScheduleConfig,
    canonicalize_schedule_prefix,
    is_valid_schedule,
)
from iccl.analysis.structured_observer.smc import FullHistoryObserver, SMCConfig


def _schedule_config() -> ScheduleConfig:
    return ScheduleConfig(
        num_modules=3,
        num_tasks=3,
        hotness=2,
        weight_values=(1.0,),
    )


def _valid_schedules(cfg: ScheduleConfig) -> np.ndarray:
    tasks = []
    for inactive in range(cfg.num_modules):
        latent = np.ones(cfg.num_modules, dtype=np.float64)
        latent[inactive] = 0.0
        tasks.append(latent)
    schedules = [np.stack(choice) for choice in product(tasks, repeat=cfg.num_tasks)]
    return np.stack([schedule for schedule in schedules if is_valid_schedule(schedule, cfg)])


def test_current_task_observer_resets_all_posterior_state() -> None:
    cfg = ScheduleConfig(
        num_modules=3,
        num_tasks=2,
        hotness=2,
        weight_values=(0.5, 1.0),
    )
    bank = sample_feature_bank(
        input_dim=2,
        num_modules=3,
        scale=1.5,
        num_features=64,
        seed=2,
    )
    observer = CurrentTaskObserver(
        feature_bank=bank,
        schedule_config=cfg,
        output_dim=1,
        relative_jitter=1e-6,
        max_relative_jitter=1e-3,
    )
    observer.start_task()
    initial = observer.predict(np.array([0.2, -0.1]))
    observer.observe(np.array([0.8]))
    assert observer.log_evidence != 0.0
    observer.end_task()
    observer.start_task()
    reset = observer.predict(np.array([0.2, -0.1]))
    assert np.allclose(reset.mean, initial.mean)
    assert reset.effective_sample_size == initial.effective_sample_size
    assert reset.log_evidence == 0.0


def test_enumerated_full_history_observer_matches_brute_force_gp_mixture() -> None:
    cfg = _schedule_config()
    schedules = _valid_schedules(cfg)
    assert len(schedules) == 24
    bank = sample_feature_bank(
        input_dim=2,
        num_modules=3,
        scale=1.25,
        num_features=48,
        seed=5,
    )
    observer = FullHistoryObserver(
        feature_bank=bank,
        schedule_config=cfg,
        output_dim=1,
        relative_jitter=1e-5,
        max_relative_jitter=1e-3,
        smc_config=SMCConfig(
            num_particles=len(schedules),
            ess_fraction=1e-12,
            task_end_rejuvenation_sweeps=0,
            max_completion_attempts=1000,
        ),
        seed=3,
        initial_schedules=schedules,
    )

    log_weights = torch.full(
        (len(schedules),),
        -np.log(len(schedules)),
        dtype=torch.float64,
    )
    history_features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    inputs = [
        np.array([0.2, -0.4]),
        np.array([0.8, 0.1]),
        np.array([-0.3, 0.5]),
    ]
    outputs = [np.array([0.7]), np.array([-0.2]), np.array([0.4])]
    task_indices = [0, 0, 1]
    canonical_schedules = schedules.copy()
    reference_diagonal = None
    brute_force_log_evidence = 0.0

    observer.start_task()
    for index, (x, y, task_index) in enumerate(
        zip(inputs, outputs, task_indices, strict=True)
    ):
        if index == 2:
            observer.start_task()
            canonical_schedules = np.stack(
                [canonicalize_schedule_prefix(schedule, 2) for schedule in canonical_schedules]
            )
        elif index == 0:
            canonical_schedules = np.stack(
                [canonicalize_schedule_prefix(schedule, 1) for schedule in canonical_schedules]
            )

        component_features = bank.features(x, canonical_schedules[:, task_index])
        if reference_diagonal is None:
            reference_diagonal = float(
                torch.mean(torch.sum(component_features.square(), dim=-1)).item()
            )
        jitter = 1e-5 * reference_diagonal
        if history_features:
            feature_history = torch.stack(history_features, dim=1)
            kernel = feature_history @ feature_history.transpose(-1, -2)
            identity = torch.eye(len(history_features), dtype=torch.float64)
            target_history = torch.stack(targets)
            cross = torch.einsum("hnj,hj->hn", feature_history, component_features)
            alpha = torch.linalg.solve(
                kernel + jitter * identity.unsqueeze(0),
                target_history.unsqueeze(0).expand(len(schedules), -1, -1),
            )
            component_mean = torch.einsum("hn,hno->ho", cross, alpha)
            solved_cross = torch.linalg.solve(
                kernel + jitter * identity.unsqueeze(0),
                cross.unsqueeze(-1),
            ).squeeze(-1)
            component_variance = (
                torch.sum(component_features.square(), dim=-1)
                + jitter
                - torch.sum(cross * solved_cross, dim=-1)
            )
        else:
            component_mean = torch.zeros((len(schedules), 1), dtype=torch.float64)
            component_variance = torch.sum(component_features.square(), dim=-1) + jitter
        weights = torch.exp(log_weights)
        brute_force_mean = torch.sum(weights.unsqueeze(-1) * component_mean, dim=0)

        prediction = observer.predict(x)
        assert np.allclose(prediction.mean, brute_force_mean.numpy(), atol=1e-10)

        target = torch.as_tensor(y, dtype=torch.float64)
        component_prediction = GPPrediction(
            mean=component_mean,
            variance=component_variance,
            triangular_solution=torch.empty((len(schedules), 0), dtype=torch.float64),
        )
        log_density = gaussian_log_predictive_density(target, component_prediction)
        log_weights, log_increment = normalize_log_weights(log_weights + log_density)
        brute_force_log_evidence += float(log_increment.item())
        update = observer.observe(y)
        assert torch.allclose(observer.log_weights, log_weights, atol=1e-10)
        assert np.isclose(update.log_evidence, brute_force_log_evidence, atol=1e-10)
        history_features.append(component_features)
        targets.append(target)


def test_full_observer_is_causal_with_respect_to_future_outputs() -> None:
    cfg = _schedule_config()
    schedules = _valid_schedules(cfg)
    bank = sample_feature_bank(
        input_dim=1,
        num_modules=3,
        scale=1.0,
        num_features=24,
        seed=6,
    )

    def first_two_predictions(future_output: float) -> list[np.ndarray]:
        observer = FullHistoryObserver(
            feature_bank=bank,
            schedule_config=cfg,
            output_dim=1,
            relative_jitter=1e-5,
            max_relative_jitter=1e-3,
            smc_config=SMCConfig(
                num_particles=len(schedules),
                ess_fraction=1e-12,
                task_end_rejuvenation_sweeps=0,
                max_completion_attempts=1000,
            ),
            seed=1,
            initial_schedules=schedules,
        )
        observer.start_task()
        predictions = [observer.predict(np.array([0.1])).mean]
        observer.observe(np.array([0.4]))
        predictions.append(observer.predict(np.array([0.2])).mean)
        observer.observe(np.array([future_output]))
        return predictions

    first = first_two_predictions(0.7)
    second = first_two_predictions(-9.0)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
