import copy

import numpy as np

from iccl.analysis.structured_observer.kernel import sample_feature_bank
from iccl.analysis.structured_observer.runner import compute_structured_observers
from iccl.analysis.structured_observer.schedule import ScheduleConfig
from iccl.analysis.structured_observer.smc import SMCConfig
from iccl.data.sequences import TOKEN_BOUNDARY, TOKEN_PAD, TOKEN_X, TOKEN_Y


def _toy_suite() -> dict[str, np.ndarray]:
    tokens = np.zeros((1, 10, 2), dtype=np.float32)
    token_types = np.full((1, 10), TOKEN_PAD, dtype=np.int64)
    targets = np.zeros((1, 10, 1), dtype=np.float32)
    task_spans = np.zeros((1, 3, 2), dtype=np.int64)
    outputs = [0.4, -0.2, 0.7]
    for task in range(3):
        boundary = 3 * task
        x_position = boundary + 1
        y_position = boundary + 2
        token_types[0, boundary:y_position + 1] = [TOKEN_BOUNDARY, TOKEN_X, TOKEN_Y]
        tokens[0, x_position, :2] = [0.1 * (task + 1), -0.2]
        tokens[0, y_position, 0] = outputs[task]
        targets[0, x_position, 0] = outputs[task]
        task_spans[0, task] = [x_position, y_position + 1]
    return {
        "tokens": tokens,
        "token_type": token_types,
        "targets": targets,
        "demo_counts": np.ones((1, 3), dtype=np.int64),
        "task_spans": task_spans,
        "base_mse": np.ones((1, 3, 1), dtype=np.float32),
        "latents": np.array(
            [[[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
        "world_readout": np.full((1, 4, 1), 999.0, dtype=np.float32),
    }


def _compute(suite: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    schedule = ScheduleConfig(
        num_modules=3,
        num_tasks=3,
        hotness=2,
        weight_values=(1.0,),
    )
    bank = sample_feature_bank(
        input_dim=2,
        num_modules=3,
        scale=1.0,
        num_features=16,
        seed=0,
    )
    return compute_structured_observers(
        suite,
        schedule_config=schedule,
        feature_bank=bank,
        modes=("full_history", "current_task"),
        smc_config=SMCConfig(
            num_particles=6,
            ess_fraction=1e-12,
            task_end_rejuvenation_sweeps=0,
            max_completion_attempts=1000,
        ),
        smc_seeds=(0,),
        relative_jitter=1e-5,
        max_relative_jitter=1e-3,
    )


def test_hidden_world_and_true_latents_cannot_change_predictions() -> None:
    suite = _toy_suite()
    altered = copy.deepcopy(suite)
    altered["world_readout"][:] = -12345.0
    altered["latents"] = altered["latents"][:, :, ::-1].copy()
    original_result = _compute(suite)
    altered_result = _compute(altered)
    assert np.array_equal(
        original_result["current_task_predictions"],
        altered_result["current_task_predictions"],
    )
    assert np.array_equal(
        original_result["full_predictions_by_seed"],
        altered_result["full_predictions_by_seed"],
    )


def test_base_mse_is_posthoc_only() -> None:
    suite = _toy_suite()
    altered = copy.deepcopy(suite)
    altered["base_mse"] *= 10.0
    original_result = _compute(suite)
    altered_result = _compute(altered)
    assert np.array_equal(
        original_result["full_predictions_by_seed"],
        altered_result["full_predictions_by_seed"],
    )
    assert not np.array_equal(original_result["full_nmse"], altered_result["full_nmse"])
