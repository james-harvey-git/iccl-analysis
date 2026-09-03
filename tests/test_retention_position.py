from dataclasses import replace

import numpy as np

from iccl.data.curriculum import SequenceConfig, check_connected
from iccl.data.dataset import sequence_rng
from iccl.data.retention_position import (
    REHEARSAL_MODES,
    build_paired_position_group,
    build_rehearsal_position_group,
)
from iccl.data.sequences import SequenceSample
from iccl.data.teacher import HyperTeacher, TeacherConfig


def make_family(weighting: str = "discrete") -> HyperTeacher:
    return HyperTeacher(
        TeacherConfig(
            input_dim=4,
            output_dim=4,
            hidden_dims=(4,),
            use_bias=True,
            num_modules=8,
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
        max_attempts=1000,
    )


def inputs(sample: SequenceSample, task: int) -> np.ndarray:
    info = sample.info
    start = int(info["task_spans"][task, 0])
    count = int(info["demo_counts"][task])
    return sample.tokens[start + 2 * np.arange(count), :4]


def block(sample: SequenceSample, task: int) -> tuple[np.ndarray, np.ndarray]:
    start, end = sample.info["task_spans"][task]
    return sample.tokens[start:end], sample.targets[start:end]


def test_paired_position_group_moves_identical_complete_task_blocks() -> None:
    group = build_paired_position_group(make_family(), make_cfg(), sequence_rng(4, 2), group_id=7)
    assert set(group) == {"repeat", "novel", "shared"}
    assert all(len(samples) == 8 for samples in group.values())
    repeats = group["repeat"]
    reference = repeats[0]
    reference_world = reference.info["world"]
    final_inputs = inputs(reference, -1)

    novel_latent = group["novel"][0].info["latents"][-1]
    shared_latent = group["shared"][0].info["latents"][-1]
    for position, repeat in enumerate(repeats):
        assert repeat.info["world"] is reference_world
        assert repeat.info["position_group_id"] == 7
        assert repeat.info["pair_id"] == 56 + position
        assert repeat.info["original_task_position"] == position
        assert repeat.info["intervening_tasks"] == 7 - position
        assert repeat.info["prior_target_latent_count"] == 1
        assert repeat.info["prior_target_support_count"] == 1
        assert repeat.info["rehearsal_mode"] == "natural"
        np.testing.assert_array_equal(inputs(repeat, -1), final_inputs)
        np.testing.assert_array_equal(group["novel"][position].info["latents"][-1], novel_latent)
        np.testing.assert_array_equal(group["shared"][position].info["latents"][-1], shared_latent)

    for logical_task in range(8):
        blocks = []
        for repeat in repeats:
            position = int(np.flatnonzero(repeat.info["logical_task_id"][:-1] == logical_task)[0])
            blocks.append(block(repeat, position))
        for task_tokens, task_targets in blocks[1:]:
            np.testing.assert_array_equal(blocks[0][0], task_tokens)
            np.testing.assert_array_equal(blocks[0][1], task_targets)

    reference_order = [value for value in repeats[0].info["logical_task_id"] if value >= 0]
    target_id = int(repeats[0].info["logical_task_id"][0])
    expected_others = [value for value in reference_order if value != target_id]
    for repeat in repeats:
        order = [value for value in repeat.info["logical_task_id"] if value >= 0]
        assert [value for value in order if value != target_id] == expected_others


def test_paired_position_controls_are_aligned_and_support_valid() -> None:
    group = build_paired_position_group(make_family(), make_cfg(), sequence_rng(1, 9), group_id=0)
    for position in range(8):
        repeat, novel, shared = (group[name][position] for name in ("repeat", "novel", "shared"))
        history = repeat.info["latents"][:8]
        supports = {tuple(np.flatnonzero(latent)) for latent in history}
        target = repeat.info["latents"][-1]
        assert tuple(np.flatnonzero(novel.info["latents"][-1])) not in supports
        np.testing.assert_array_equal(
            np.flatnonzero(shared.info["latents"][-1]), np.flatnonzero(target)
        )
        for control in (novel, shared):
            assert control.info["pair_id"] == repeat.info["pair_id"]
            np.testing.assert_array_equal(inputs(control, -1), inputs(repeat, -1))
            np.testing.assert_array_equal(control.tokens[: -2 * 3], repeat.tokens[: -2 * 3])


def test_controlled_rehearsal_grid_has_exact_exposures_and_connectivity_labels() -> None:
    group = build_rehearsal_position_group(
        make_family(), make_cfg(), sequence_rng(8, 3), group_id=2
    )
    assert all(len(samples) == 6 for samples in group.values())
    seen_cells = set()
    for repeat in group["repeat"]:
        position = int(repeat.info["original_task_position"])
        mode = str(repeat.info["rehearsal_mode"])
        seen_cells.add((position, mode))
        exposures = sorted(repeat.info["target_module_post_exposures"].tolist())
        assert exposures == {"none": [0, 0], "one": [0, 1], "both": [1, 1]}[mode]
        assert repeat.info["prior_target_latent_count"] == 1
        assert repeat.info["prior_target_support_count"] == 1
        status = repeat.info["support_status"]
        assert status == ("disconnected_ood" if (position, mode) == (0, "none") else "connected_id")
        assert check_connected(repeat.info["latents"][:8] != 0) == (status == "connected_id")
    assert seen_cells == {(position, mode) for position in (0, 3) for mode in REHEARSAL_MODES}


def test_controlled_rehearsal_reuses_world_inputs_and_control_latents() -> None:
    group = build_rehearsal_position_group(
        make_family(), make_cfg(), sequence_rng(2, 4), group_id=5
    )
    repeats = group["repeat"]
    world = repeats[0].info["world"]
    final_inputs = inputs(repeats[0], -1)
    for offset, repeat in enumerate(repeats):
        assert repeat.info["world"] is world
        np.testing.assert_array_equal(inputs(repeat, -1), final_inputs)
        np.testing.assert_array_equal(
            group["novel"][offset].info["latents"][-1],
            group["novel"][0].info["latents"][-1],
        )
        np.testing.assert_array_equal(
            group["shared"][offset].info["latents"][-1],
            group["shared"][0].info["latents"][-1],
        )

    for position_offset in (0, 3):
        position_rows = [
            row for row in repeats if row.info["original_task_position"] == position_offset
        ]
        for task in range(8):
            blocks = [inputs(row, task) for row in position_rows]
            for block in blocks[1:]:
                np.testing.assert_array_equal(blocks[0], block)

        histories = {row.info["rehearsal_mode"]: row.info["latents"][:8] for row in position_rows}
        none, one, both = (histories[mode] for mode in REHEARSAL_MODES)
        assert np.any(none != one, axis=1).sum() == 1
        assert np.any(none != both, axis=1).sum() == 2

    novel = group["novel"][0].info["latents"][-1]
    novel_support = tuple(np.flatnonzero(novel))
    for repeat in repeats:
        supports = {tuple(np.flatnonzero(latent)) for latent in repeat.info["latents"][:8]}
        assert novel_support not in supports


def test_rehearsed_constituent_is_balanced_across_worlds() -> None:
    family, cfg = make_family(), make_cfg()
    selected = []
    for group_id in range(8):
        group = build_rehearsal_position_group(
            family, cfg, sequence_rng(12, group_id), group_id=group_id
        )
        one = next(
            sample
            for sample in group["repeat"]
            if sample.info["original_task_position"] == 0 and sample.info["rehearsal_mode"] == "one"
        )
        support = one.info["target_support"].tolist()
        selected.append(support.index(one.info["designated_constituent"]))
    assert selected.count(0) == selected.count(1) == 4


def test_controlled_rehearsal_supports_the_connectivity_floor() -> None:
    cfg = replace(make_cfg(), surplus_tasks=0, demos_per_task=2)
    group = build_rehearsal_position_group(make_family(), cfg, sequence_rng(5, 1), group_id=0)
    assert all(len(samples) == 6 for samples in group.values())
    assert {sample.info["original_task_position"] for sample in group["repeat"]} == {0, 3}


def test_binary_groups_omit_shared_control() -> None:
    modes = ("novel",)
    paired = build_paired_position_group(
        make_family("binary"),
        make_cfg(),
        sequence_rng(3, 1),
        group_id=0,
        control_modes=modes,
    )
    rehearsal = build_rehearsal_position_group(
        make_family("binary"),
        make_cfg(),
        sequence_rng(3, 2),
        group_id=0,
        control_modes=modes,
    )
    assert set(paired) == set(rehearsal) == {"repeat", "novel"}
