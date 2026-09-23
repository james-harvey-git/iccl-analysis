"""Immutable mmap shards, checked manifests and resumable sample order for probes."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset, Sampler

from iccl.analysis.probe_config import SPLITS, stream_seed
from iccl.analysis.probe_targets import PROTOCOL, episode_targets, flat_targets
from iccl.data.dataset import sequence_rng
from iccl.data.teacher import ModulePool


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Publish one complete JSON document without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def array_schema(features: int) -> dict[str, dict[str, Any]]:
    shapes = {
        "states": (features,),
        "modules": (8, 2, 17, 16),
        "readout": (16, 16),
        "world_modules": (8, 16, 16),
        "world_biases": (8, 16),
        "world_readout": (16, 16),
        "latents": (8, 8),
        "module_ids": (8, 2),
        "first_appearance": (8,),
        "occurrence_count": (8,),
        "latent_rank": (),
        "readout_norms": (16,),
        "episode_index": (),
    }
    integers = {
        "module_ids",
        "first_appearance",
        "occurrence_count",
        "latent_rank",
        "episode_index",
    }
    return {
        key: {
            "shape": list(shape),
            "dtype": "int64"
            if key in integers
            else "float64"
            if key == "readout_norms"
            else "float32",
        }
        for key, shape in shapes.items()
    }


def _validate_arrays(
    arrays: dict[str, np.ndarray], schema: dict[str, Any], start: int, stop: int
) -> None:
    if set(arrays) != set(schema) or stop <= start:
        raise ValueError("invalid probe shard fields or range")
    for name, spec in schema.items():
        value = arrays[name]
        if value.shape != (stop - start, *spec["shape"]) or value.dtype != np.dtype(spec["dtype"]):
            raise ValueError(f"invalid shape/dtype for probe array {name}")
        for offset in range(0, len(value), 64):
            if not np.isfinite(value[offset : offset + 64]).all():
                raise ValueError(f"nonfinite probe array {name}")
    if not np.array_equal(arrays["episode_index"], np.arange(start, stop)):
        raise ValueError("probe episode indices do not match the committed shard range")
    for row in range(stop - start):
        pool = ModulePool(
            [arrays["world_modules"][row]],
            [arrays["world_biases"][row]],
            arrays["world_readout"][row],
        )
        expected = episode_targets(pool, arrays["latents"][row])
        for name, value in expected.items():
            actual = arrays[name][row]
            equal = (
                np.array_equal(actual, value)
                if value.dtype.kind in "iu"
                else np.allclose(actual, value, rtol=2e-6, atol=1e-8)
            )
            if not equal:
                raise ValueError(f"inconsistent target/world metadata: {name}")


def _close_arrays(arrays: dict[str, np.ndarray]) -> None:
    for value in arrays.values():
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()


def read_manifest(root: Path | str) -> dict[str, Any]:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"no captured probe manifest at {path}")
    value = json.loads(path.read_text())
    if value.get("protocol") != PROTOCOL or value.get("dataset_id") != digest(
        value.get("identity")
    ):
        raise ValueError("incompatible or malformed probe dataset identity")
    features = value["identity"]["input_features"]
    if type(features) is not int or features < 1 or value.get("schema") != array_schema(features):
        raise ValueError("invalid probe array schema")
    if set(value["shards"]) != set(SPLITS) or set(value["requested_counts"]) != set(SPLITS):
        raise ValueError("invalid probe dataset splits")
    for split in SPLITS:
        count = value["requested_counts"][split]
        if type(count) is not int or count < 1:
            raise ValueError("invalid requested episode count")
        end = 0
        for shard in value["shards"][split]:
            if shard["start"] != end or not end < shard["stop"] <= count:
                raise ValueError("probe shard ranges have gaps, overlaps or invalid bounds")
            end = shard["stop"]
    return value


def completed_count(manifest: dict[str, Any], split: str) -> int:
    entries = manifest["shards"][split]
    return entries[-1]["stop"] if entries else 0


def validate_shard(root: Path, manifest: dict[str, Any], split: str, entry: dict[str, Any]) -> None:
    start, stop = entry["start"], entry["stop"]
    name = f"{split}/shard_{start:010d}_{stop:010d}"
    if entry["name"] != name or entry["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("invalid probe shard identity")
    path = root / name
    if json.loads((path / "shard.json").read_text()) != entry:
        raise ValueError(f"inconsistent shard metadata: {name}")
    if set(entry["files"]) != set(manifest["schema"]):
        raise ValueError("invalid shard file list")
    arrays = {}
    try:
        for key in manifest["schema"]:
            file = path / f"{key}.npy"
            if file_digest(file) != entry["files"][key]:
                raise ValueError(f"checksum mismatch for {file}")
            arrays[key] = np.load(file, mmap_mode="r", allow_pickle=False)
        _validate_arrays(arrays, manifest["schema"], start, stop)
    finally:
        _close_arrays(arrays)


class DatasetWriter:
    """Exclusive writer; published shards are immutable, including during train extension."""

    def __init__(
        self,
        root: Path | str,
        identity: dict[str, Any],
        counts: dict[str, int],
        shard_size: int,
        *,
        resume: bool = True,
    ) -> None:
        self.root, self.identity = Path(root), identity
        self.counts, self.shard_size, self.resume = counts, shard_size, resume
        self.manifest: dict[str, Any] = {}
        self._lock: Any = None

    def record_provenance(self, provenance: dict[str, Any]) -> None:
        self.manifest.setdefault("source_provenance", provenance)
        self._save()

    def __enter__(self) -> DatasetWriter:
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = (self.root / ".capture.lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._initialize()
        except BaseException:
            self._lock.close()
            raise
        return self

    def __exit__(self, *_: object) -> None:
        self._lock.close()

    def _initialize(self) -> None:
        if (self.root / "manifest.json").exists():
            if not self.resume:
                raise FileExistsError("dataset exists; enable capture.resume or choose a new path")
            self.manifest = read_manifest(self.root)
            if (
                self.manifest["identity"] != self.identity
                or self.manifest["shard_size"] != self.shard_size
            ):
                raise ValueError("capture identity/config differs from the existing dataset")
            previous = self.manifest["requested_counts"]
            if (
                any(self.counts[s] != previous[s] for s in ("validation", "test"))
                or self.counts["train"] < previous["train"]
            ):
                raise ValueError(
                    "resume may extend training count only; held-out populations are fixed"
                )
            for split in SPLITS:
                for entry in self.manifest["shards"][split]:
                    validate_shard(self.root, self.manifest, split, entry)
            self.manifest["requested_counts"] = dict(self.counts)
        else:
            if any(p.name != ".capture.lock" for p in self.root.iterdir()):
                raise ValueError(
                    "refusing to initialize a nonempty dataset directory without a manifest"
                )
            self.manifest = {
                "protocol": PROTOCOL,
                "identity": self.identity,
                "dataset_id": digest(self.identity),
                "schema": array_schema(self.identity["input_features"]),
                "requested_counts": dict(self.counts),
                "shard_size": self.shard_size,
                "shards": {split: [] for split in SPLITS},
            }
        self._save()
        # A crash after directory publication but before manifest publication leaves a
        # complete, self-describing shard that can be verified and adopted exactly once.
        for split in SPLITS:
            recorded = {entry["name"] for entry in self.manifest["shards"][split]}
            for directory in sorted((self.root / split).glob("shard_*")):
                name = f"{split}/{directory.name}"
                if name in recorded:
                    continue
                entry = json.loads((directory / "shard.json").read_text())
                if (
                    entry["name"] != name
                    or entry["start"] != self.completed(split)
                    or entry["stop"] > self.counts[split]
                ):
                    raise ValueError(
                        "uncommitted probe shard does not continue the requested range"
                    )
                validate_shard(self.root, self.manifest, split, entry)
                self.manifest["shards"][split].append(entry)
                self._save()

    def completed(self, split: str) -> int:
        return completed_count(self.manifest, split)

    def _save(self) -> None:
        self.manifest["complete"] = all(self.completed(s) == self.counts[s] for s in SPLITS)
        write_json(self.root / "manifest.json", self.manifest)

    def append(self, split: str, arrays: dict[str, np.ndarray]) -> None:
        start = self.completed(split)
        stop = start + len(arrays["states"])
        if stop > self.counts[split] or stop - start > self.shard_size:
            raise ValueError("shard exceeds the configured range or size")
        _validate_arrays(arrays, self.manifest["schema"], start, stop)
        parent = self.root / split
        parent.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".partial-", dir=parent))
        name = f"{split}/shard_{start:010d}_{stop:010d}"
        destination = self.root / name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite committed shard {name}")
        files = {}
        for key, value in arrays.items():
            path = temporary / f"{key}.npy"
            with path.open("wb") as stream:
                np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            files[key] = file_digest(path)
        entry = {
            "name": name,
            "dataset_id": self.manifest["dataset_id"],
            "start": start,
            "stop": stop,
            "files": files,
        }
        write_json(temporary / "shard.json", entry)
        temporary.rename(destination)
        self.manifest["shards"][split].append(entry)
        self._save()


class CapturedDataset(Dataset):
    """A checked split with bounded per-process mmap handles and copied sample buffers."""

    def __init__(
        self, root: Path | str, split: str, target_permutation: np.ndarray | None = None
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.manifest = read_manifest(self.root)
        if split not in SPLITS:
            raise ValueError(f"invalid probe split: {split}")
        self.split = split
        self.count = self.manifest["requested_counts"][split]
        if completed_count(self.manifest, split) != self.count:
            raise ValueError(f"captured split {split} is incomplete")
        self.entries = self.manifest["shards"][split]
        for entry in self.entries:
            validate_shard(self.root, self.manifest, split, entry)
        self.ends = [entry["stop"] for entry in self.entries]
        self.signature = digest(
            {"dataset": self.manifest["dataset_id"], "split": split, "shards": self.entries}
        )
        self.target_permutation = target_permutation
        if target_permutation is not None:
            if (
                split != "train"
                or target_permutation.dtype.kind not in "iu"
                or not np.array_equal(np.sort(target_permutation), np.arange(self.count))
            ):
                raise ValueError("target permutation must permute the complete training split")
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._pid = os.getpid()

    def __len__(self) -> int:
        return self.count

    def _row(self, index: int, *, include_states: bool = True) -> dict[str, np.ndarray]:
        if not 0 <= index < self.count:
            raise IndexError(index)
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        number = bisect_right(self.ends, index)
        if number not in self._cache:
            path = self.root / self.entries[number]["name"]
            self._cache[number] = {
                key: np.load(path / f"{key}.npy", mmap_mode="r", allow_pickle=False)
                for key in self.manifest["schema"]
            }
            if len(self._cache) > 2:
                _, old = self._cache.popitem(last=False)
                _close_arrays(old)
        self._cache.move_to_end(number)
        offset = index - self.entries[number]["start"]
        return {
            key: value[offset].copy()
            for key, value in self._cache[number].items()
            if include_states or key != "states"
        }

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        item = self._row(index)
        target_index = (
            index if self.target_permutation is None else int(self.target_permutation[index])
        )
        if target_index != index:
            target = self._row(target_index, include_states=False)
            for key in item.keys() - {"states", "episode_index"}:
                item[key] = target[key]
        item["target"] = flat_targets(item["modules"], item["readout"])
        item["target_episode_index"] = np.asarray(target_index, np.int64)
        return item

    def close(self) -> None:
        for arrays in self._cache.values():
            _close_arrays(arrays)
        self._cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        return state

    def __del__(self) -> None:
        if hasattr(self, "_cache"):
            self.close()


def shuffled_pairing(count: int, seed: int) -> np.ndarray:
    """One reproducible random cycle, ensuring no training episode retains its own target."""
    if count < 2:
        raise ValueError("shuffled targets require at least two training episodes")
    order = sequence_rng(stream_seed(seed, "target-shuffle"), 0).permutation(count)
    mapping = np.empty(count, dtype=np.int64)
    mapping[order] = np.roll(order, 1)
    return mapping


class EpisodeBatchSampler(Sampler[list[int]]):
    """Infinite shuffled epochs; the checkpoint cursor counts consumed, not prefetched samples."""

    def __init__(self, count: int, batch_size: int, seed: int, consumed: int = 0) -> None:
        if count < 1 or batch_size < 1 or consumed < 0:
            raise ValueError("invalid probe sampler dimensions/cursor")
        self.count, self.batch_size, self.seed, self.consumed = count, batch_size, seed, consumed

    def __iter__(self) -> Iterator[list[int]]:
        epoch, offset = divmod(self.consumed, self.count)
        while True:
            order = sequence_rng(stream_seed(self.seed, "training-order"), epoch).permutation(
                self.count
            )
            while offset < self.count:
                stop = min(offset + self.batch_size, self.count)
                yield order[offset:stop].tolist()
                offset = stop
            epoch += 1
            offset = 0
