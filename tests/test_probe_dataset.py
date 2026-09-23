import copy
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from iccl.analysis.capture import CaptureEpisodes, capture_dataset, collate_capture
from iccl.analysis.probe_config import stream_seed
from iccl.analysis.probe_dataset import (
    CapturedDataset,
    DatasetWriter,
    EpisodeBatchSampler,
    read_manifest,
    shuffled_pairing,
    write_json,
)


def test_offline_artifact_capture_keeps_provenance_without_online_lineage(
    probe_cfg: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = Path(probe_cfg.probe.capture.checkpoint)
    reference = "wandb://entity/project/source:v0"
    probe_cfg.probe.capture.checkpoint = reference
    probe_cfg.wandb.mode = "offline"
    lineage: list[str] = []
    monkeypatch.setattr("iccl.analysis.capture.resolve_checkpoint_path", lambda _: (source, True))
    monkeypatch.setattr("iccl.analysis.capture.RunLogger.start", lambda self: None)
    monkeypatch.setattr(
        "iccl.analysis.capture.RunLogger.use_artifact", lambda self, ref: lineage.append(ref)
    )
    report = capture_dataset(probe_cfg, tmp_path / "capture")
    assert report["new_episodes_per_second"] > 0
    assert not lineage
    manifest = read_manifest(probe_cfg.probe.dataset.path)
    assert manifest["source_provenance"]["checkpoint_reference"] == reference


def test_capture_storage_resume_and_training_extension(
    probe_cfg: DictConfig, tmp_path: Path
) -> None:
    result = capture_dataset(probe_cfg, tmp_path / "capture")
    assert result["new_episodes"] == 8
    root = Path(probe_cfg.probe.dataset.path)
    manifest = read_manifest(root)
    assert manifest["complete"]
    assert manifest["identity"]["token_count"] == 521
    assert manifest["identity"]["input_features"] == 64
    train = CapturedDataset(root, "train")
    assert len(train) == 4
    first = train[0]
    assert first["states"].shape == (64,) and first["target"].shape == (4608,)
    assert first["states"].dtype == np.float32
    for index in (0, 3, 2, 1):
        assert train[index]["episode_index"] == index
    heldout = copy.deepcopy(manifest["shards"]["test"])
    repeat = capture_dataset(probe_cfg, tmp_path / "repeat")
    assert repeat["new_episodes"] == 0
    probe_cfg.probe.dataset.counts.train = 6
    extension = capture_dataset(probe_cfg, tmp_path / "extension")
    assert extension["new_episodes"] == 2
    extended = CapturedDataset(root, "train")
    assert len(extended) == 6
    np.testing.assert_array_equal(extended[0]["states"], first["states"])
    assert read_manifest(root)["shards"]["test"] == heldout
    probe_cfg.probe.capture.precision = "bf16"
    with pytest.raises(ValueError, match="identity/config"):
        capture_dataset(probe_cfg, tmp_path / "mismatch")


def test_episode_generation_invariant_to_workers_and_order(probe_cfg: DictConfig) -> None:
    seed = stream_seed(17, "episodes/train")
    dataset = CaptureEpisodes(probe_cfg.data, seed, 10, 13)
    rows = [dataset[index] for index in (2, 0, 1)]
    for workers in (0, 2):
        loader = DataLoader(
            dataset,
            batch_size=2,
            num_workers=workers,
            collate_fn=collate_capture,
            multiprocessing_context="spawn" if workers else None,
        )
        batches = list(loader)
        for i in range(3):
            for name in rows[0]:
                combined = np.concatenate([batch[name] for batch in batches])
                np.testing.assert_array_equal(combined[i], dataset[i][name])
        assert batches[0]["tokens"].shape == (2, 521, 16)
        assert (batches[0]["token_type"][:, -1] == 2).all()
    same_world = CaptureEpisodes(probe_cfg.data, seed, 12, 13)[0]
    np.testing.assert_array_equal(same_world["world_modules"], rows[0]["world_modules"])


def test_uncommitted_shard_recovery_and_corruption(probe_cfg: DictConfig, tmp_path: Path) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    root = Path(probe_cfg.probe.dataset.path)
    manifest = read_manifest(root)
    shard = manifest["shards"]["train"].pop()
    manifest["complete"] = False
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="incomplete"):
        CapturedDataset(root, "train")
    with DatasetWriter(
        root, manifest["identity"], manifest["requested_counts"], manifest["shard_size"]
    ) as writer:
        assert writer.completed("train") == 4
    assert read_manifest(root)["shards"]["train"][-1] == shard
    path = root / shard["name"] / "states.npy"
    with path.open("r+b") as stream:
        stream.seek(-1, 2)
        byte = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="checksum"):
        CapturedDataset(root, "train")


def test_shuffled_episode_targets_are_intact(probe_cfg: DictConfig, tmp_path: Path) -> None:
    capture_dataset(probe_cfg, tmp_path / "capture")
    plain = CapturedDataset(probe_cfg.probe.dataset.path, "train")
    pairing = shuffled_pairing(len(plain), 0)
    assert not (pairing == np.arange(len(plain))).any()
    shuffled = CapturedDataset(probe_cfg.probe.dataset.path, "train", pairing)
    for index, target_index in enumerate(pairing):
        actual, source, target = shuffled[index], plain[index], plain[int(target_index)]
        np.testing.assert_array_equal(actual["states"], source["states"])
        for field in ("target", "latents", "module_ids", "world_readout"):
            np.testing.assert_array_equal(actual[field], target[field])
    with pytest.raises(ValueError, match="training split"):
        CapturedDataset(probe_cfg.probe.dataset.path, "test", np.array([1, 0]))


def test_sampler_resume_ignores_prefetched_batches() -> None:
    first = iter(EpisodeBatchSampler(7, 3, seed=4))
    consumed = [next(first), next(first)]
    expected = [next(first) for _ in range(5)]
    resumed = iter(EpisodeBatchSampler(7, 3, seed=4, consumed=sum(map(len, consumed))))
    assert [next(resumed) for _ in range(5)] == expected
    assert len(expected[0]) == 1
