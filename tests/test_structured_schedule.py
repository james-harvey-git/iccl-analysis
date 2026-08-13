import numpy as np

from iccl.analysis.structured_observer.schedule import (
    ConditionedSchedulePrior,
    ScheduleConfig,
    canonical_task_classes,
    canonicalize_schedule_prefix,
    enumerate_task_latents,
    is_valid_schedule,
    sample_conditional_completion,
    sample_valid_schedule,
    systematic_resampling_indices,
)


def _config() -> ScheduleConfig:
    return ScheduleConfig(
        num_modules=4,
        num_tasks=4,
        hotness=2,
        weight_values=(0.5, 1.0),
    )


def test_task_enumeration_and_canonical_class_masses() -> None:
    cfg = _config()
    assert enumerate_task_latents(cfg).shape == (24, 4)
    classes, log_prior = canonical_task_classes(cfg)
    assert classes.shape == (3, 4)
    assert np.allclose(np.exp(log_prior), [0.25, 0.5, 0.25])


def test_valid_schedule_sampling_and_conditional_completion() -> None:
    cfg = _config()
    rng = np.random.default_rng(8)
    schedule, attempts = sample_valid_schedule(rng, cfg, max_attempts=10_000)
    assert attempts >= 1
    assert is_valid_schedule(schedule, cfg)
    completed, completion_attempts = sample_conditional_completion(
        rng,
        schedule[:2],
        cfg,
        max_attempts=10_000,
    )
    assert completion_attempts >= 1
    assert np.array_equal(completed[:2], schedule[:2])
    assert is_valid_schedule(completed, cfg)


def test_conditioned_prefix_prior_counts_and_samples_exact_valid_support_prior() -> None:
    cfg = ScheduleConfig(
        num_modules=3,
        num_tasks=3,
        hotness=2,
        weight_values=(1.0,),
    )
    prior = ConditionedSchedulePrior(cfg)
    assert prior.completion_count(prior.initial_state, 3) == 24

    first = np.array([[1.0, 1.0, 0.0]])
    state = prior.state_from_prefix(first)
    assert np.array_equal(prior.support_counts(state, 1), [2.0, 3.0, 3.0])

    rng = np.random.default_rng(17)
    counts: dict[bytes, int] = {}
    for _ in range(12_000):
        schedule = prior.sample_schedule(rng)
        assert is_valid_schedule(schedule, cfg)
        key = (schedule != 0.0).tobytes()
        counts[key] = counts.get(key, 0) + 1
    assert len(counts) == 24
    frequencies = np.asarray(list(counts.values()), dtype=np.float64) / 12_000
    assert np.max(np.abs(frequencies - 1.0 / 24.0)) < 0.01


def test_prefix_canonicalization_never_uses_future_rows() -> None:
    schedule = np.array(
        [
            [0.0, 0.5, 0.5, 0.0],
            [1.0, 0.0, 0.0, 0.8],
            [0.0, 0.9, 0.6, 0.0],
        ]
    )
    altered_future = schedule.copy()
    altered_future[2] = [1.0, 0.0, 0.0, 0.5]
    canonical = canonicalize_schedule_prefix(schedule, observed_tasks=2)
    altered = canonicalize_schedule_prefix(altered_future, observed_tasks=2)
    assert np.array_equal(canonical[:2], altered[:2])
    assert np.array_equal(canonical[0], [0.0, 0.0, 0.5, 0.5])


def test_systematic_resampling_is_deterministic_for_seed() -> None:
    weights = np.array([0.05, 0.15, 0.3, 0.5])
    first = systematic_resampling_indices(weights, np.random.default_rng(12))
    second = systematic_resampling_indices(weights, np.random.default_rng(12))
    assert np.array_equal(first, second)
    assert first.shape == (4,)
