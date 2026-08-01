from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from iccl.data.dataset import sequence_rng
from iccl.data.sequences import (
    TOKEN_BOUNDARY,
    TOKEN_X,
    TOKEN_Y,
    FinalTaskConfig,
    PhaseConfig,
    SequenceConfig,
    assert_feasible,
    build_paired_control,
    build_paired_retention_control,
    build_sequence,
    check_compositional,
    check_connected,
)
from iccl.data.teacher import HyperTeacher, TeacherConfig

DEFAULT_TEACHER = TeacherConfig(
    input_dim=4,
    output_dim=4,
    hidden_dims=(4,),
    use_bias=True,
    num_modules=8,
    scale=3.0,
    weighting="discrete",
)

DEFAULT_SEQUENCE = SequenceConfig(
    phases=(PhaseConfig(num_tasks=8, hotness=(2, 2)),),
    demos_per_task=4,
    signal_boundaries=True,
    require_identifiable=True,
)


def make_family(**overrides: Any) -> HyperTeacher:
    return HyperTeacher(replace(DEFAULT_TEACHER, **overrides), max_hotness=3)


def make_seq_cfg(**overrides: Any) -> SequenceConfig:
    return replace(DEFAULT_SEQUENCE, **overrides)


def test_check_connected_on_hand_built_patterns() -> None:
    chain = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.int8) > 0
    assert check_connected(chain)
    split = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.int8) > 0
    assert not check_connected(split)
    assert check_compositional(chain, 4)
    assert not check_compositional(chain[:2], 4)


def test_feasibility_bound() -> None:
    # 3 two-hot tasks connectedly cover at most 4 of 8 modules.
    with pytest.raises(ValueError, match="cannot connectedly cover"):
        assert_feasible(make_seq_cfg(phases=(PhaseConfig(3, (2, 2)),)), num_modules=8)
    assert_feasible(make_seq_cfg(), num_modules=8)


def test_built_sequences_are_identifiable() -> None:
    family = make_family()
    cfg = make_seq_cfg()
    for i in range(10):
        seq = build_sequence(family, cfg, sequence_rng(0, i))
        supports = seq.info["latents"] > 0
        assert check_compositional(supports, 8)
        assert check_connected(supports)


def test_token_layout_invariants() -> None:
    family = make_family()
    cfg = make_seq_cfg()
    seq = build_sequence(family, cfg, sequence_rng(0, 0))

    num_tasks, demos = 8, 4
    assert seq.tokens.shape == (num_tasks * (2 * demos + 1), 4)
    np.testing.assert_array_equal(seq.info["boundaries"], np.arange(num_tasks) * (2 * demos + 1))

    for start in seq.info["boundaries"]:
        assert seq.token_type[start] == TOKEN_BOUNDARY
        assert (seq.tokens[start] == 0).all()
    x_positions = seq.token_type == TOKEN_X
    y_positions = seq.token_type == TOKEN_Y
    assert x_positions.sum() == y_positions.sum() == num_tasks * demos
    np.testing.assert_array_equal(seq.loss_mask, x_positions.astype(np.float32))
    # Each x-position's target is revealed as the next token's content.
    np.testing.assert_array_equal(
        seq.targets[x_positions], seq.tokens[np.flatnonzero(x_positions) + 1]
    )
    assert (seq.targets[~x_positions] == 0).all()


def test_unsignalled_sequences_have_no_boundary_tokens() -> None:
    family = make_family()
    cfg = make_seq_cfg(signal_boundaries=False, demos_per_task=(3, 6))
    seq = build_sequence(family, cfg, sequence_rng(0, 0))
    assert (seq.token_type != TOKEN_BOUNDARY).all()
    assert len(seq.info["boundaries"]) == 0
    assert seq.tokens.shape[0] == 2 * seq.info["demo_counts"].sum()
    assert (seq.info["demo_counts"] >= 3).all() and (seq.info["demo_counts"] <= 6).all()


def test_composite_final_task_and_paired_control() -> None:
    family = make_family()
    cfg = make_seq_cfg()
    final = FinalTaskConfig(mode="composite", hotness=2, num_demos=2)
    rng = sequence_rng(0, 0)
    seq = build_sequence(family, cfg, rng, final_task=final, include_world=True)

    latents = seq.info["latents"]
    assert latents.shape[0] == 9
    final_support = tuple(np.flatnonzero(latents[-1]))
    demonstrated = {tuple(np.flatnonzero(lat)) for lat in latents[:-1]}
    assert final_support not in demonstrated
    assert seq.info["demo_counts"][-1] == 2

    control = build_paired_control(family, cfg, seq, final, rng)
    np.testing.assert_array_equal(control.info["latents"][0], latents[-1])
    assert control.info["num_curriculum_tasks"] == 0
    assert control.tokens.shape[0] == 2 * 2 + 1
    # Same world: the same x must map to the same y under both sequences' teachers.
    x_query = np.flatnonzero(control.token_type == TOKEN_X)[0]
    from iccl.data.teacher import teacher_forward

    y_a = teacher_forward(seq.info["world"], latents[-1], control.tokens[x_query : x_query + 1])
    np.testing.assert_allclose(control.targets[x_query], y_a[0], rtol=1e-5)


def test_revisit_appends_first_task() -> None:
    family = make_family()
    cfg = make_seq_cfg()
    seq = build_sequence(family, cfg, sequence_rng(0, 0), revisit_demos=3)
    latents = seq.info["latents"]
    np.testing.assert_array_equal(latents[-1], latents[0])
    assert seq.info["demo_counts"][-1] == 3


def test_paired_retention_control_shares_everything_but_the_final_task() -> None:
    family = make_family()
    cfg = make_seq_cfg()
    rng = sequence_rng(0, 0)
    seq = build_sequence(family, cfg, rng, revisit_demos=3, include_world=True)
    control = build_paired_retention_control(family, seq, rng, mode="novel")

    start, end = seq.info["task_spans"][-1]
    np.testing.assert_array_equal(control.tokens[:start], seq.tokens[:start])
    np.testing.assert_array_equal(control.targets[:start], seq.targets[:start])
    np.testing.assert_array_equal(control.token_type, seq.token_type)
    np.testing.assert_array_equal(control.loss_mask, seq.loss_mask)
    for key in ("demo_counts", "boundaries", "task_spans"):
        np.testing.assert_array_equal(control.info[key], seq.info[key])
    assert control.info["num_curriculum_tasks"] == seq.info["num_curriculum_tasks"]
    assert control.info["world"] is seq.info["world"]

    # The final block keeps its inputs and swaps its targets, so the model meets
    # the block in the same state and answers its first demonstration alike.
    x_positions = np.arange(start, end, 2)
    np.testing.assert_array_equal(control.tokens[x_positions], seq.tokens[x_positions])
    assert not np.array_equal(control.targets[x_positions], seq.targets[x_positions])
    np.testing.assert_array_equal(
        control.tokens[x_positions + 1, : family.cfg.output_dim],
        control.targets[x_positions],
    )
    expected = control.targets[x_positions]
    np.testing.assert_allclose(
        control.info["base_mse"][-1],
        ((expected - expected.mean(axis=0)) ** 2).mean(axis=0),
        rtol=1e-5,
    )


def test_retention_control_modes_pick_their_supports() -> None:
    family = make_family()
    cfg = make_seq_cfg()
    rng = sequence_rng(0, 0)
    seq = build_sequence(family, cfg, rng, revisit_demos=3, include_world=True)
    revisited = seq.info["latents"][-1]
    demonstrated = {tuple(np.flatnonzero(lat)) for lat in seq.info["latents"]}

    novel = build_paired_retention_control(family, seq, rng, mode="novel").info["latents"][-1]
    assert tuple(np.flatnonzero(novel)) not in demonstrated
    assert np.count_nonzero(novel) == np.count_nonzero(revisited)

    shared = build_paired_retention_control(family, seq, rng, mode="shared").info["latents"][-1]
    np.testing.assert_array_equal(np.flatnonzero(shared), np.flatnonzero(revisited))
    assert not np.array_equal(shared, revisited)


def test_shared_retention_control_is_degenerate_under_binary_weighting() -> None:
    family = make_family(weighting="binary")
    cfg = make_seq_cfg()
    rng = sequence_rng(0, 0)
    seq = build_sequence(family, cfg, rng, revisit_demos=3, include_world=True)
    with pytest.raises(ValueError, match="weighting=binary"):
        build_paired_retention_control(family, seq, rng, mode="shared")


def test_base_mse_matches_targets() -> None:
    family = make_family()
    cfg = make_seq_cfg()
    seq = build_sequence(family, cfg, sequence_rng(0, 0))
    start, end = seq.info["task_spans"][0]
    task_targets = seq.targets[start:end][seq.loss_mask[start:end] > 0]
    expected = ((task_targets - task_targets.mean(axis=0)) ** 2).mean(axis=0)
    np.testing.assert_allclose(seq.info["base_mse"][0], expected, rtol=1e-5)


def _distinct_edge_degrees(latents: np.ndarray) -> list[int]:
    edges = {tuple(np.flatnonzero(lat)) for lat in latents}
    degrees = np.zeros(latents.shape[1], dtype=int)
    for a, b in edges:
        degrees[[a, b]] += 1
    return sorted(degrees.tolist())


def test_structured_graph_families_have_expected_shape() -> None:
    family = make_family()
    cases = {
        "chain": (7, [1, 1, 2, 2, 2, 2, 2, 2]),
        "ring": (8, [2] * 8),
        "star": (7, [1] * 7 + [7]),
    }
    for graph, (num_tasks, expected_degrees) in cases.items():
        cfg = make_seq_cfg(
            phases=(PhaseConfig(num_tasks=num_tasks, hotness=(2, 2)),), task_graph=graph
        )
        for i in range(5):
            seq = build_sequence(family, cfg, sequence_rng(0, i))
            latents = seq.info["latents"]
            supports = latents > 0
            assert (supports.sum(axis=1) == 2).all()
            assert check_compositional(supports, 8)
            assert check_connected(supports)
            assert _distinct_edge_degrees(latents) == expected_degrees, graph


def test_structured_extra_tasks_duplicate_skeleton_edges() -> None:
    family = make_family()
    cfg = make_seq_cfg(phases=(PhaseConfig(num_tasks=11, hotness=(2, 2)),), task_graph="chain")
    seq = build_sequence(family, cfg, sequence_rng(0, 0))
    edges = [tuple(np.flatnonzero(lat)) for lat in seq.info["latents"]]
    assert len(edges) == 11
    assert len(set(edges)) == 7  # skeleton only; extras are duplicates


def test_ordered_chain_presents_path_order() -> None:
    family = make_family()
    cfg = make_seq_cfg(
        phases=(PhaseConfig(num_tasks=7, hotness=(2, 2)),),
        task_graph="chain",
        graph_ordered=True,
    )
    seq = build_sequence(family, cfg, sequence_rng(0, 0))
    supports = [set(np.flatnonzero(lat)) for lat in seq.info["latents"]]
    for prev, curr in zip(supports[:-1], supports[1:], strict=True):
        assert len(prev & curr) == 1


def test_structured_graph_validation() -> None:
    family = make_family()
    bad_hotness = make_seq_cfg(
        phases=(PhaseConfig(num_tasks=8, hotness=(1, 2)),), task_graph="chain"
    )
    with pytest.raises(ValueError, match="2-hot"):
        build_sequence(family, bad_hotness, sequence_rng(0, 0))
    too_few = make_seq_cfg(phases=(PhaseConfig(num_tasks=5, hotness=(2, 2)),), task_graph="star")
    with pytest.raises(ValueError, match="needs >= 7 tasks"):
        build_sequence(family, too_few, sequence_rng(0, 0))
