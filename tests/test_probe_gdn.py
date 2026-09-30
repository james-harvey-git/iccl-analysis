"""GDN baseline shares the worlds, queries and variance definition with the decoder."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from iccl.analysis.capture import CaptureEpisodes, capture_dataset, collate_capture
from iccl.analysis.probe_config import stream_seed
from iccl.analysis.probe_dataset import read_manifest
from iccl.analysis.probe_functional import functional_errors, functional_inputs, functional_targets
from iccl.analysis.probe_gdn import GDNFunctionalEvaluator
from iccl.data.sequences import TOKEN_X


@torch.inference_mode()
def test_gdn_queries_match_independent_full_prefixes(probe_cfg: DictConfig, tmp_path: Path) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    manifest = read_manifest(probe_cfg.probe.dataset.path)
    gdn = GDNFunctionalEvaluator(
        manifest, "test", probe_cfg.probe.evaluation.gdn, torch.device("cpu")
    )
    episodes = CaptureEpisodes(probe_cfg.data, stream_seed(0, "episodes/test"), 0, 2)
    batch = collate_capture([episodes[1], episodes[0]])
    inputs = np.stack([functional_inputs(int(i), 7, 2, 34) for i in batch["episode_index"]])
    actual = gdn.predict(batch, inputs)
    expected = np.empty_like(actual)
    for episode in range(2):
        for task in range(7):
            stop = (task + 1) * 65
            prefix = torch.from_numpy(batch["tokens"][episode, :stop]).repeat(2, 1, 1)
            types = torch.from_numpy(batch["token_type"][episode, :stop]).repeat(2, 1)
            expected[episode, task] = (
                gdn.model(
                    torch.cat((prefix, torch.from_numpy(inputs[episode, task, :, None])), dim=1),
                    torch.cat((types, torch.full((2, 1), TOKEN_X, dtype=types.dtype)), dim=1),
                )
                .preds[:, -1]
                .numpy()
            )
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)
    errors = gdn.score(batch, inputs_per_task=2, seed=34)
    reference = functional_errors(expected, functional_targets(batch, inputs))
    np.testing.assert_allclose(
        errors["functional_nmse_by_task"], reference["functional_nmse_by_task"], rtol=3e-5
    )
    gdn.batch_size, gdn.query_batch_size = 1, 1
    smaller = gdn.score(batch, inputs_per_task=2, seed=34)
    np.testing.assert_allclose(
        smaller["functional_nmse_by_task"], errors["functional_nmse_by_task"], rtol=3e-5
    )
    batch["world_modules"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="regenerated"):
        gdn.score(batch, inputs_per_task=2, seed=34)
    manifest["identity"]["source_model_digest"] = "different"
    with pytest.raises(ValueError, match="differs"):
        GDNFunctionalEvaluator(
            manifest, "test", probe_cfg.probe.evaluation.gdn, torch.device("cpu")
        )
