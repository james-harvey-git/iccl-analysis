import numpy as np
import torch
from torch.utils.data import DataLoader

from iccl.data.dataset import SequenceDataset, collate_sequences, sequence_rng
from iccl.data.sequences import TOKEN_PAD, PhaseConfig, SequenceConfig
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
    return SequenceDataset(family, cfg, base_seed=base_seed, num_sequences=num_sequences)


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
