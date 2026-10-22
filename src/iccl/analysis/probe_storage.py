"""Versioned, atomic task-vector artifacts and bounded reads of immutable feature shards."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from iccl.analysis.probe_features import FeatureSelection
from iccl.analysis.probe_worlds import PROTOCOL


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def lock(path: Path, *, blocking: bool = True) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
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


def save_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, name)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, allow_pickle=False, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def read_manifest(root: Path | str, *, complete: bool = True) -> dict[str, Any]:
    manifest = read_json(Path(root) / "manifest.json")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(
            "incompatible module-decoder/unknown dataset; capture a task-vector dataset"
        )
    if manifest.get("dataset_id") != digest(manifest["identity"]):
        raise ValueError("dataset identity checksum mismatch")
    if complete and not manifest.get("complete"):
        raise ValueError(
            "dataset creation is incomplete; finish capture before training/evaluation"
        )
    return manifest


def read_artifact(root: Path, kind: str, *, complete: bool = True) -> dict[str, Any]:
    value = read_json(root / "manifest.json")
    if value.get("protocol") != PROTOCOL or value.get("kind") != kind:
        raise ValueError(f"expected a task-vector {kind} artifact")
    if value.get("artifact_id") != digest(value["identity"]):
        raise ValueError("artifact identity checksum mismatch")
    if complete and not value.get("complete"):
        raise ValueError(f"{kind} stage is incomplete")
    return value


def initialize_artifact(root: Path, kind: str, identity: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".manifest.lock"):
        if (root / "manifest.json").exists():
            value = read_artifact(root, kind, complete=False)
            if value["artifact_id"] != digest(identity):
                raise ValueError(f"{kind} configuration differs; choose a fresh output directory")
            return value
        if any(path.name != ".manifest.lock" for path in root.iterdir()):
            raise ValueError(f"unrecognized nonempty {kind} output directory")
        value = {
            "protocol": PROTOCOL,
            "kind": kind,
            "identity": identity,
            "artifact_id": digest(identity),
            "complete": False,
            "worlds": {},
        }
        write_json(root / "manifest.json", value)
        return value


def complete_artifact(root: Path, kind: str, worlds: int) -> dict[str, Any]:
    with lock(root / ".manifest.lock"):
        value = read_artifact(root, kind, complete=False)
        value["worlds"] = {
            str(w): file_digest(root / f"world-{w:04d}" / "complete.json") for w in range(worlds)
        }
        value["complete"] = True
        write_json(root / "manifest.json", value)
    return value


def read_world_artifact(root: Path, manifest: dict[str, Any], world: int) -> dict[str, Any]:
    path = root / f"world-{world:04d}" / "complete.json"
    if file_digest(path) != manifest["worlds"][str(world)]:
        raise ValueError("world artifact checksum mismatch")
    return read_json(path)


def source_hashes(*names: str) -> dict[str, str]:
    package = Path(__file__).resolve().parent
    return {name: file_digest(package / name) for name in names}


def initialize_dataset(
    root: Path, identity: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / ".manifest.lock"):
        path = root / "manifest.json"
        if path.exists():
            manifest = read_manifest(root, complete=False)
            if manifest["dataset_id"] != digest(identity):
                raise ValueError(
                    "dataset configuration/checkpoint differs; use a different dataset path"
                )
            return manifest
        if any(p.name != ".manifest.lock" for p in root.iterdir()):
            raise ValueError("refusing to initialize a nonempty unrecognized dataset directory")
        manifest = {
            "protocol": PROTOCOL,
            "dataset_id": digest(identity),
            "identity": identity,
            "provenance": provenance,
            "complete": False,
            "worlds": {},
        }
        write_json(path, manifest)
        return manifest


def shard_path(root: Path, world: int, position: int, split: str, start: int) -> Path:
    return root / f"world-{world:04d}" / f"position-{position:03d}" / split / f"shard-{start:08d}"


def validate_shard(path: Path, dataset_id: str, *, checksum: bool = True) -> dict[str, Any]:
    entry = read_json(path / "shard.json")
    if entry["dataset_id"] != dataset_id or entry["stop"] <= entry["start"]:
        raise ValueError(f"incompatible shard: {path}")
    for name, spec in entry["arrays"].items():
        file = path / f"{name}.npy"
        if checksum and file_digest(file) != spec["sha256"]:
            raise ValueError(f"checksum mismatch: {file}")
        value = np.load(file, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]:
            raise ValueError(f"invalid shard shape/dtype: {file}")
        mapping = getattr(value, "_mmap", None)
        if mapping is not None:
            mapping.close()
    return entry


def write_shard(
    path: Path, arrays: dict[str, np.ndarray], context: dict[str, Any]
) -> dict[str, Any]:
    """Publish one immutable folder only after every array and checksum is complete."""
    if path.exists():
        return validate_shard(path, context["dataset_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent))
    try:
        schema = {}
        for name, array in arrays.items():
            if (
                "/" in name
                or array.dtype.hasobject
                or len(array) != context["stop"] - context["start"]
            ):
                raise ValueError(f"invalid shard array: {name}")
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite captured array: {name}")
            file = pending / f"{name}.npy"
            np.save(file, array, allow_pickle=False)
            schema[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": file_digest(file),
            }
        entry = context | {"arrays": schema}
        write_json(pending / "shard.json", entry)
        os.replace(pending, path)
        return entry
    finally:
        if pending.exists():
            shutil.rmtree(pending)


class CapturedSplit:
    """One world's position/split; feature tensors are opened lazily with a bounded mmap cache."""

    def __init__(self, root: Path | str, world: int, position: int, split: str) -> None:
        self.root = Path(root)
        self.manifest = read_manifest(self.root)
        self.world, self.position, self.split = world, position, split
        world_dir = self.root / f"world-{world:04d}"
        world_manifest = read_json(world_dir / "complete.json")
        expected_world_digest = self.manifest["worlds"][str(world)]
        if file_digest(world_dir / "complete.json") != expected_world_digest:
            raise ValueError("world manifest checksum mismatch")
        self.entries = world_manifest["splits"][f"{position}/{split}"]
        expected = self.manifest["identity"]["dataset"]["trajectories"][split]
        self.ends = np.asarray([entry["stop"] for entry in self.entries], dtype=np.int64)
        offset = 0
        for entry in self.entries:
            path = shard_path(self.root, world, position, split, entry["start"])
            if (
                entry["start"] != offset
                or validate_shard(path, self.manifest["dataset_id"]) != entry
            ):
                raise ValueError("noncontiguous or corrupt dataset shards")
            offset = entry["stop"]
        if offset != expected:
            raise ValueError("split does not contain its declared trial count")
        self.count = offset
        self._cache: OrderedDict[tuple[int, str], np.ndarray] = OrderedDict()
        self.targets = self.metadata("target")

    def close(self) -> None:
        while self._cache:
            _, value = self._cache.popitem(last=False)
            mapping = getattr(value, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def array(self, shard: int, name: str) -> np.ndarray:
        key = (shard, name)
        if key not in self._cache:
            entry = self.entries[shard]
            path = shard_path(self.root, self.world, self.position, self.split, entry["start"])
            self._cache[key] = np.load(path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            while len(self._cache) > 24:
                _, old = self._cache.popitem(last=False)
                mapping = getattr(old, "_mmap", None)
                if mapping is not None:
                    mapping.close()
        self._cache.move_to_end(key)
        return self._cache[key]

    def metadata(self, name: str) -> np.ndarray:
        return np.concatenate([np.array(self.array(i, name)) for i in range(len(self.entries))])

    def matrix(
        self, key: str, selection: FeatureSelection | None = None, rows: np.ndarray | None = None
    ) -> FeatureMatrix:
        if key not in self.entries[0]["arrays"]:
            raise ValueError(f"representation not captured: {key}")
        width = int(np.prod(self.entries[0]["arrays"][key]["shape"][1:]))
        return FeatureMatrix(
            self,
            key,
            selection,
            np.arange(self.count) if rows is None else rows,
            width if selection is None else selection.dimension,
        )


class FeatureMatrix:
    """Stream rows from a feature view, with optional deterministic nested training subsets."""

    def __init__(
        self,
        dataset: CapturedSplit,
        key: str,
        selection: FeatureSelection | None,
        indices: np.ndarray,
        width: int,
    ) -> None:
        self.dataset, self.key, self.selection = dataset, key, selection
        self.indices = np.asarray(indices, dtype=np.int64)
        self.shape = (len(self.indices), width)

    def rows(self, indices: np.ndarray) -> np.ndarray:
        absolute = self.indices[indices]
        result = np.empty((len(indices), self.shape[1]), dtype=np.float32)
        shards = np.searchsorted(self.dataset.ends, absolute, side="right")
        for shard in np.unique(shards):
            mask = np.flatnonzero(shards == shard)
            start = self.dataset.entries[int(shard)]["start"]
            values = self.dataset.array(int(shard), self.key)[absolute[mask] - start]
            result[mask] = (
                values.reshape(len(values), -1)
                if self.selection is None
                else self.selection.select(values)
            )
        return result

    def batches(self, size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, self.shape[0], size):
            ids = np.arange(start, min(start + size, self.shape[0]))
            yield ids, self.rows(ids)
