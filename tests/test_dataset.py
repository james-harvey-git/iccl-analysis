import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from iccl.data.curriculum import PhaseConfig, SequenceConfig
from iccl.data.dataset import (
    SequenceDataset,
    collate_sequences,
    module_count_config_from,
    sequence_dataset_from_config,
    sequence_rng,
)
from iccl.data.sequences import TOKEN_PAD
from iccl.data.teacher import HyperTeacher, TeacherConfig


def make_dataset(
    base_seed: int = 0,
    demos_per_task: int | tuple[int, int] = 4,
    num_sequences: int | None = None,
) -> SequenceDataset:
    family = HyperTeacher(
        TeacherConfig(
            input_dim=4,
            output_dim=4,
            hidden_dims=(4,),
            use_bias=True,
            num_modules=8,
            scale=3.0,
            weighting="discrete",
        ),
        max_hotness=2,
    )
    cfg = SequenceConfig(
        phases=(PhaseConfig(num_tasks=8, hotness=(2, 2)),),
        demos_per_task=demos_per_task,
        signal_boundaries=True,
        require_identifiable=True,
    )
    module_counts = module_count_config_from(OmegaConf.create({"num_modules": 8}))
    return SequenceDataset(
        {8: family},
        module_counts,
        cfg,
        base_seed=base_seed,
        num_sequences=num_sequences,
    )


def test_same_seed_and_index_reproduce_exactly() -> None:
    dataset = make_dataset()
    a, b = dataset.build(7), dataset.build(7)
    np.testing.assert_array_equal(a.tokens, b.tokens)
    np.testing.assert_array_equal(a.info["latents"], b.info["latents"])


def test_different_indices_and_seeds_differ() -> None:
    dataset = make_dataset()
    other_seed = make_dataset(base_seed=1)
    assert not np.array_equal(dataset.build(0).tokens, dataset.build(1).tokens)
    assert not np.array_equal(dataset.build(0).tokens, other_seed.build(0).tokens)


def test_content_is_independent_of_access_order() -> None:
    dataset = make_dataset()
    forward = [dataset.build(i).tokens for i in range(4)]
    backward = [dataset.build(i).tokens for i in reversed(range(4))]
    for i in range(4):
        np.testing.assert_array_equal(forward[i], backward[3 - i])


def test_philox_key_is_order_sensitive() -> None:
    # (seed, index) and (index, seed) must produce different streams.
    a = sequence_rng(0, 1).standard_normal(4)
    b = sequence_rng(1, 0).standard_normal(4)
    assert not np.array_equal(a, b)


def test_dataloader_yields_finite_batches() -> None:
    dataset = make_dataset(num_sequences=6)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_sequences)
    batches = list(loader)
    assert [b["tokens"].shape[0] for b in batches] == [4, 2]
    assert batches[0]["tokens"].dtype == torch.float32
    assert batches[0]["token_type"].dtype == torch.int64


def test_collate_pads_variable_lengths() -> None:
    dataset = make_dataset(num_sequences=8, demos_per_task=(3, 6))
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_sequences)
    batch = next(iter(loader))
    lengths = (batch["token_type"] != TOKEN_PAD).sum(dim=1)
    assert batch["tokens"].shape[1] == int(lengths.max())
    padded = batch["token_type"] == TOKEN_PAD
    assert (batch["tokens"][padded] == 0).all()
    assert (batch["loss_mask"][padded] == 0).all()


def variable_cfg() -> DictConfig:
    return OmegaConf.create(
        {
            "input_dim": 4,
            "output_dim": 4,
            "hidden_dims": [4],
            "use_bias": True,
            "num_modules": {"min": 4, "max": 8, "held_out": [6]},
            "scale": 3.0,
            "weighting": "discrete",
            "sequence": {
                "curriculum_sampler": "constructive",
                "hotness": 2,
                "surplus_tasks": [0, 2],
                "demos_per_task": {"min": 2, "max": 4, "scope": "per_task"},
                "signal_boundaries": True,
                "require_identifiable": True,
                "require_full_rank": False,
            },
        }
    )


def test_module_count_support_excludes_holdouts_and_fixed_value_uses_no_rng() -> None:
    cfg = variable_cfg()
    spec = module_count_config_from(cfg)
    assert spec.allowed == (4, 5, 7, 8)
    observed = {spec.sample(sequence_rng(0, i)) for i in range(100)}
    assert observed == set(spec.allowed)

    fixed = module_count_config_from(OmegaConf.create({"num_modules": 8}))
    left, right = sequence_rng(4, 2), sequence_rng(4, 2)
    assert fixed.sample(left) == 8
    np.testing.assert_array_equal(left.standard_normal(8), right.standard_normal(8))


def test_module_count_validation_rejects_bad_holdouts() -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        module_count_config_from(
            OmegaConf.create({"num_modules": {"min": 4, "max": 8, "held_out": [4]}})
        )
    with pytest.raises(ValueError, match="duplicates"):
        module_count_config_from(
            OmegaConf.create({"num_modules": {"min": 4, "max": 8, "held_out": [6, 6]}})
        )


def test_variable_dataset_is_deterministic_and_never_emits_heldout_modules() -> None:
    cfg = variable_cfg()
    dataset = sequence_dataset_from_config(cfg, base_seed=9)
    modules = []
    for i in range(30):
        first, second = dataset.build(i), dataset.build(i)
        np.testing.assert_array_equal(first.tokens, second.tokens)
        modules.append(first.info["num_modules"])
        assert first.info["latents"].shape[1] == first.info["num_modules"]
    assert 6 not in modules
    assert set(modules) == {4, 5, 7, 8}
