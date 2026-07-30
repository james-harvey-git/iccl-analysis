"""Golden-stream regression tests for the on-the-fly data pipeline.

Pins the exact random stream produced by the pilot configuration under a fixed
seed: any code change that alters sampling order, distributions, or tokenization
fails here loudly instead of silently changing the training distribution. If a
change is *intended* to alter the stream, regenerate the fingerprints and treat
the update as a breaking change to dataset reproducibility (previously frozen
eval sets no longer correspond to the new stream).
"""

import numpy as np
import pytest

from iccl.data.dataset import sequence_rng
from iccl.data.sequences import PhaseConfig, SequenceConfig, build_sequence
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
