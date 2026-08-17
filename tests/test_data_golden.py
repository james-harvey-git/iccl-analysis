"""Golden fingerprints for fixed-world and variable-world sequence streams."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

from iccl.data.curriculum import PhaseConfig, SequenceConfig
from iccl.data.dataset import sequence_dataset_from_config, sequence_rng
from iccl.data.sequences import build_sequence
from iccl.data.teacher import HyperTeacher, TeacherConfig

PILOT_FAMILY = TeacherConfig(
    input_dim=16,
    output_dim=16,
    hidden_dims=(16,),
    use_bias=True,
    num_modules=8,
    scale=1.7320508075688772,
    weighting="discrete",
)

PILOT_SEQUENCE = SequenceConfig(
    phases=(PhaseConfig(num_tasks=8, hotness=(2, 2)),),
    demos_per_task=32,
    signal_boundaries=True,
    require_identifiable=True,
)

# Fingerprints of sequences 0-2 under base seed 0.
GOLDEN = [
    {
        "tokens_mean": -0.04677049,
        "tokens_std": 0.64118946,
        "targets_abs_sum": 2316.69043,
        "latents_sum": 11.69999981,
        "supports": [(2, 4), (2, 7), (2, 6), (1, 2), (3, 6), (3, 5), (1, 5), (0, 3)],
    },
    {
        "tokens_mean": 0.01641509,
        "tokens_std": 0.60684633,
        "targets_abs_sum": 2035.65625,
        "latents_sum": 11.0,
        "supports": [(1, 3), (5, 6), (1, 7), (2, 3), (0, 3), (5, 7), (1, 6), (1, 4)],
    },
    {
        "tokens_mean": -0.11984272,
        "tokens_std": 0.71225899,
        "targets_abs_sum": 2685.57471,
        "latents_sum": 12.89999962,
        "supports": [(0, 2), (2, 3), (0, 1), (1, 4), (4, 5), (0, 6), (3, 5), (2, 7)],
    },
]


@pytest.mark.parametrize("index", range(len(GOLDEN)))
def test_pilot_stream_is_unchanged(index: int) -> None:
    family = HyperTeacher(PILOT_FAMILY, max_hotness=2)
    seq = build_sequence(family, PILOT_SEQUENCE, sequence_rng(0, index))
    golden = GOLDEN[index]

    assert seq.tokens.shape == (520, 16)
    supports = [tuple(int(m) for m in np.flatnonzero(lat)) for lat in seq.info["latents"]]
    assert supports == golden["supports"]
    np.testing.assert_allclose(seq.tokens.mean(), golden["tokens_mean"], atol=1e-6)
    np.testing.assert_allclose(seq.tokens.std(), golden["tokens_std"], atol=1e-6)
    np.testing.assert_allclose(np.abs(seq.targets).sum(), golden["targets_abs_sum"], rtol=1e-5)
    np.testing.assert_allclose(seq.info["latents"].sum(), golden["latents_sum"], atol=1e-5)


def test_fixed_config_selects_the_pinned_training_distribution() -> None:
    config_path = Path(__file__).parents[1] / "configs/data/hyperteacher_fixed.yaml"
    config = OmegaConf.load(config_path)
    assert isinstance(config, DictConfig)
    dataset = sequence_dataset_from_config(config, base_seed=0)
    configured = dataset.build(0)
    direct = build_sequence(
        HyperTeacher(PILOT_FAMILY, max_hotness=2), PILOT_SEQUENCE, sequence_rng(0, 0)
    )
    np.testing.assert_array_equal(configured.tokens, direct.tokens)
    np.testing.assert_array_equal(configured.info["latents"], direct.info["latents"])


@pytest.mark.parametrize(
    "index,modules,surplus,demos,tokens_mean,targets_sum,latents_sum",
    [
        (0, 7, 0, [2, 3, 2, 4, 3, 4], 0.096305415, 63.689713, 8.6000004),
        (1, 5, 2, [2, 2, 3, 2, 4, 2], -0.080361843, 39.106003, 9.3000002),
        (2, 4, 2, [4, 4, 4, 2, 2], 0.19349524, 63.953129, 7.5999999),
    ],
)
def test_variable_world_stream_is_pinned(
    index: int,
    modules: int,
    surplus: int,
    demos: list[int],
    tokens_mean: float,
    targets_sum: float,
    latents_sum: float,
) -> None:
    cfg = OmegaConf.create(
        {
            "input_dim": 4,
            "output_dim": 4,
            "hidden_dims": [4],
            "use_bias": True,
            "num_modules": {"min": 4, "max": 7, "held_out": [6]},
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
    dataset = sequence_dataset_from_config(cfg, base_seed=19)
    sample = dataset.build(index)
    assert sample.info["num_modules"] == modules
    assert sample.info["num_surplus_tasks"] == surplus
    assert sample.info["demo_counts"].tolist() == demos
    np.testing.assert_allclose(sample.tokens.mean(), tokens_mean, atol=1e-7)
    np.testing.assert_allclose(np.abs(sample.targets).sum(), targets_sum, rtol=1e-6)
    np.testing.assert_allclose(sample.info["latents"].sum(), latents_sum, atol=1e-6)
