"""Fixed-world identities, support holdouts and causal exposure annotations."""

import numpy as np

from iccl.analysis.probe_worlds import make_trial, make_world
from iccl.data.teacher import TeacherConfig, teacher_forward


def world(phase="development"):
    return make_world(
        TeacherConfig(4, 3, (5,), True, 8, 1.0, "discrete"),
        seed=3,
        phase=phase,
        world_id=0,
        pair_counts={"train": 16, "validation": 6, "test_unseen": 6},
    )


def test_world_reuse_disjoint_supports_and_prefix_exposures():
    fixed = world()
    again = world()
    np.testing.assert_array_equal(fixed.pool.modules[0], again.pool.modules[0])
    assert not np.array_equal(fixed.pool.modules[0], world("confirmation").pool.modules[0])
    sets = {name: set(map(tuple, pairs)) for name, pairs in fixed.pairs.items()}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test_unseen"]
    assert not sets["validation"] & sets["test_unseen"]
    for position in (1, 5, 9):
        trial = make_trial(fixed, split="test_unseen", position=position, index=7, tasks=9, demos=4)
        latents = trial.sample.info["latents"]
        backgrounds = np.delete(latents, position - 1, axis=0)
        assert (backgrounds != 0).any(axis=0).all()
        assert tuple(trial.support) in sets["test_unseen"]
        assert all(tuple(np.flatnonzero(z)) in sets["train"] for z in backgrounds)
        expected = (latents[: position - 1, trial.support] != 0).sum(axis=0)
        np.testing.assert_array_equal(expected, trial.prior_exposures)
        assert trial.prior_exposures.sum() <= position - 1
        if position == 1:
            assert not trial.prior_exposures.any()
        if position == 9:
            assert (trial.prior_exposures > 0).all()
        np.testing.assert_array_equal(
            trial.branch_answer, teacher_forward(fixed.pool, trial.target, trial.branch_input)
        )
        assert len(trial.observations("final_input", 4, 3)) == 3 * 7 + 4
        assert len(trial.observations("final_answer", 4, 3)) == 4 * 7
        assert len(trial.observations("branch_input", 4, 3)) == 4 * 7 + 4
        repeat = make_trial(
            again, split="test_unseen", position=position, index=7, tasks=9, demos=4
        )
        np.testing.assert_array_equal(trial.sample.tokens, repeat.sample.tokens)


def test_independent_splits_and_balanced_support_cycles():
    fixed = world()

    def trial(split, index):
        return make_trial(fixed, split=split, position=5, index=index, tasks=9, demos=4)

    observed = {tuple(trial("train", index).support) for index in range(16)}
    assert observed == set(map(tuple, fixed.pairs["train"]))
    assert not np.array_equal(
        trial("train", 0).sample.tokens, trial("test_familiar", 0).sample.tokens
    )
