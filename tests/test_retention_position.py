from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from iccl.data.controls import sample_retention_control_latent
from iccl.data.curriculum import SequenceConfig, check_compositional, check_connected
from iccl.data.dataset import sequence_rng
from iccl.data.retention_position import build_paired_position_group, build_rehearsal_position_group
from iccl.data.sequences import SequenceSample
from iccl.data.teacher import HyperTeacher, TeacherConfig


def make_family(weighting: str = "discrete", modules: int = 8) -> HyperTeacher:
    return HyperTeacher(
        TeacherConfig(
            input_dim=4,
            output_dim=4,
            hidden_dims=(4,),
            use_bias=True,
            num_modules=modules,
            scale=3.0,
            weighting=weighting,
        ),
        max_hotness=2,
    )


def make_cfg() -> SequenceConfig:
    return SequenceConfig(
        phases=(),
        demos_per_task=3,
        signal_boundaries=True,
        require_identifiable=True,
        curriculum_sampler="constructive",
        hotness=2,
        surplus_tasks=1,
    )


def block(sample: SequenceSample, task: int) -> tuple[np.ndarray, np.ndarray]:
    start, end = sample.info["task_spans"][task]
    return sample.tokens[start:end], sample.targets[start:end]


def assert_conditions(group: dict[str, list[SequenceSample]]) -> None:
    reference = group["repeat"][0]
    for row, repeat in enumerate(group["repeat"]):
        p = repeat.info["original_task_position"]
        a = repeat.info["target_support"]
        target = repeat.info["latents"][-1]
        for condition, samples in group.items():
            sample = samples[row]
            np.testing.assert_array_equal(block(sample, -1), block(reference, -1))
            np.testing.assert_array_equal(
                sample.info["base_mse"][-1], reference.info["base_mse"][-1]
            )
            assert not sample.info["target_module_pre_exposures"].any()
            assert sample.info["prior_target_latent_count"] == int(condition == "repeat")
            assert sample.info["prior_target_support_count"] == int(condition != "unexposed")
            assert sample.info["world"] is reference.info["world"]
            for task in range(len(sample.info["latents"])):
                if task != p:
                    np.testing.assert_array_equal(block(sample, task), block(repeat, task))
            original = sample.info["latents"][p]
            if condition == "unexposed":
                assert not original[a].any()
                np.testing.assert_array_equal(np.sort(original[original != 0]), np.sort(target[a]))
            elif condition == "shared":
                np.testing.assert_array_equal(original != 0, target != 0)
                assert not np.array_equal(original, target)
            core = (
                sample.info["latents"][sample.info["background_task_indices"]][:, target == 0] != 0
            )
            assert check_compositional(core, len(target) - 2) and check_connected(core)


@pytest.mark.parametrize("modules,surplus", [(4, 0), (4, 3), (8, 0), (8, 1), (8, 4)])
@pytest.mark.parametrize("sampler", ["constructive", "rejection"])
@pytest.mark.parametrize("seed", [1, 7])
def test_connected_background_and_paired_frozen_blocks(
    modules: int, surplus: int, sampler: str, seed: int
) -> None:
    cfg = replace(make_cfg(), surplus_tasks=surplus, curriculum_sampler=sampler)
    group = build_paired_position_group(
        make_family(modules=modules), cfg, sequence_rng(seed, 2), group_id=7
    )
    tasks = modules - 1 + surplus
    assert all(len(samples) == tasks for samples in group.values())
    assert_conditions(group)
    for condition, samples in group.items():
        assert not any(s.info["target_module_post_exposures"].any() for s in samples)
        for logical_task in range(tasks):
            reference = None
            for p, sample in enumerate(samples):
                task = int(np.flatnonzero(sample.info["logical_task_id"] == logical_task)[0])
                current = block(sample, task)
                if reference is not None:
                    np.testing.assert_array_equal(current, reference)
                reference = current
                assert sample.info["pair_id"] == 7 * tasks + p
                assert sample.info["intervening_tasks"] == tasks - 1 - p
        if condition == "unexposed":
            assert all(
                s.info["history_connected"] and not s.info["history_covered"] for s in samples
            )
        else:
            assert all(
                not s.info["history_connected"] and s.info["history_covered"] for s in samples
            )


def test_rehearsal_has_common_later_exposure_and_no_pre_exposure() -> None:
    group = build_rehearsal_position_group(
        make_family(), make_cfg(), sequence_rng(8, 3), group_id=2
    )
    assert all(len(samples) == 6 for samples in group.values())
    assert_conditions(group)
    for condition, samples in group.items():
        for sample in samples:
            mode = sample.info["rehearsal_mode"]
            post = sample.info["target_module_post_exposures"]
            assert sorted(post) == {"none": [0, 0], "one": [0, 1], "both": [1, 1]}[mode]
            np.testing.assert_array_equal(
                sample.info["constituent_task_exposures"], post + int(condition != "unexposed")
            )
        for start in (0, 3):
            none, one, both = samples[start : start + 3]
            changed = np.any(none.info["latents"] != both.info["latents"], axis=1)
            np.testing.assert_array_equal(
                np.flatnonzero(changed), np.sort(none.info["rehearsal_positions"])
            )
            slots = none.info["rehearsal_positions"]
            np.testing.assert_array_equal(block(one, int(slots[0])), block(both, int(slots[0])))
            np.testing.assert_array_equal(block(one, int(slots[1])), block(none, int(slots[1])))
            for task in range(9):
                np.testing.assert_array_equal(block(none, task)[0][::2], block(both, task)[0][::2])


def test_rehearsal_requires_connected_core_and_two_later_slots() -> None:
    with pytest.raises(ValueError, match="T>=M"):
        build_rehearsal_position_group(
            make_family(), replace(make_cfg(), surplus_tasks=0), sequence_rng(0, 0), group_id=0
        )
    with pytest.raises(ValueError, match="post-middle"):
        build_rehearsal_position_group(
            make_family(modules=4),
            replace(make_cfg(), surplus_tasks=0),
            sequence_rng(0, 0),
            group_id=0,
        )


def test_rehearsed_constituent_balances_worlds() -> None:
    selected = []
    for world in range(4):
        group = build_rehearsal_position_group(
            make_family(), make_cfg(), sequence_rng(12, world), group_id=world
        )
        sample = group["repeat"][1]
        selected.append(
            sample.info["target_support"].tolist().index(sample.info["designated_constituent"])
        )
    assert selected == [0, 1, 0, 1]


@pytest.mark.parametrize("builder", [build_paired_position_group, build_rehearsal_position_group])
def test_generation_is_deterministic_and_binary_omits_shared(builder: Callable) -> None:
    first = builder(
        make_family("binary"),
        make_cfg(),
        sequence_rng(1, 3),
        group_id=3,
        control_modes=("unexposed",),
    )
    second = builder(
        make_family("binary"),
        make_cfg(),
        sequence_rng(1, 3),
        group_id=3,
        control_modes=("unexposed",),
    )
    assert set(first) == {"repeat", "unexposed"}
    for condition in first:
        for left, right in zip(first[condition], second[condition], strict=True):
            np.testing.assert_array_equal(left.tokens, right.tokens)


def test_shared_control_resamples_equal_serialized_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    family = make_family()
    target = np.array([0.25, 0.75, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    different = target.copy()
    different[:2] = [0.75, 0.25]
    candidates = iter([target.copy(), different])
    monkeypatch.setattr(family, "apply_weighting", lambda *_: next(candidates))
    np.testing.assert_array_equal(
        sample_retention_control_latent(family, target, sequence_rng(1, 0), "shared", 2), different
    )
    monkeypatch.setattr(family, "apply_weighting", lambda *_: target.copy())
    with pytest.raises(RuntimeError, match="distinct"):
        sample_retention_control_latent(family, target, sequence_rng(1, 0), "shared", 2)
    with pytest.raises(ValueError, match="binary"):
        sample_retention_control_latent(
            make_family("binary"), target, sequence_rng(1, 0), "shared", 2
        )
    with pytest.raises(ValueError, match="full_rank"):
        build_paired_position_group(
            family, replace(make_cfg(), require_full_rank=True), sequence_rng(1, 0), group_id=0
        )
