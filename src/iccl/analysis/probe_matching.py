"""Lazy, checked C ABI for exact CPU assignment; importing does not build native code."""

from __future__ import annotations

import ctypes as ct
import fcntl
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import threading
import uuid
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from iccl.analysis.probe_targets import OUTPUT_FEATURES


@dataclass(frozen=True)
class Assignment:
    """p[i] is the target hidden coordinate assigned to prediction coordinate i."""

    scores: np.ndarray
    masks: np.ndarray
    permutations: np.ndarray
    counters: np.ndarray
    timings: np.ndarray


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_library(cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Serialize cache initialization across processes and atomically publish a binary."""
    if sys.platform not in ("darwin", "linux"):
        raise RuntimeError("native probe matching supports Linux and macOS")
    source = Path(__file__).with_name("native") / "assignment.cpp"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        raise ValueError("CXX must name a C++17 compiler")
    try:
        version = subprocess.run(
            [*compiler, "--version"], capture_output=True, text=True, check=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("a C++17 compiler is required; install one or set CXX") from error
    flags = ["-O3", "-std=c++17", "-pthread", "-fPIC"]
    flags += ["-dynamiclib" if sys.platform == "darwin" else "-shared"]
    identity = {
        "source_sha256": _sha(source),
        "compiler": compiler,
        "compiler_version": version,
        "flags": flags,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "method": "branch_warm",
        "abi": 1,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = cache_dir.expanduser().resolve() / key
    directory.mkdir(parents=True, exist_ok=True)
    library = directory / ("assignment.dylib" if sys.platform == "darwin" else "assignment.so")
    metadata = directory / "build.json"
    with (directory / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if library.is_file() and metadata.is_file():
            recorded = json.loads(metadata.read_text())
            if recorded["identity"] == identity and recorded["binary_sha256"] == _sha(library):
                return library, recorded
        temporary = directory / f"build-{os.getpid()}-{uuid.uuid4().hex}{library.suffix}"
        command = [*compiler, *flags, str(source), "-o", str(temporary)]
        try:
            subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
            record = {"identity": identity, "command": command, "binary_sha256": _sha(temporary)}
            temporary.replace(library)
            meta_tmp = metadata.with_suffix(".tmp")
            meta_tmp.write_text(json.dumps(record, indent=2) + "\n")
            meta_tmp.replace(metadata)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"native assignment build failed:\n{error.stderr}") from error
        finally:
            temporary.unlink(missing_ok=True)
    return library, record


def _release(library: Any, handle: int, owner_pid: int) -> None:
    # Worker threads do not survive fork; destroying an inherited pool would wait forever.
    if os.getpid() == owner_pid:
        library.probe_destroy_pool(handle)


class AssignmentSolver:
    """One persistent native worker pool, owned and serialized by its creating process."""

    def __init__(
        self, num_threads: int = 8, cache_dir: Path | str = "outputs/.cache/probe-assignment"
    ):
        if type(num_threads) is not int or not 1 <= num_threads <= 256:
            raise ValueError("solver num_threads must be an integer in [1,256]")
        path, self.build = build_library(Path(cache_dir))
        self.library_path = path
        self.num_threads = int(num_threads)
        self._lib = ct.CDLL(str(path))
        self._lib.probe_last_error.argtypes = []
        self._lib.probe_last_error.restype = ct.c_char_p
        self._lib.probe_create_pool.argtypes = [ct.c_int]
        self._lib.probe_create_pool.restype = ct.c_void_p
        self._lib.probe_destroy_pool.argtypes = [ct.c_void_p]
        self._lib.probe_destroy_pool.restype = None
        self._lib.probe_match.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_double,
            ct.c_int,
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_void_p,
        ]
        self._lib.probe_match.restype = ct.c_int
        self._lib.probe_construct.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_double,
            ct.c_void_p,
        ]
        self._lib.probe_construct.restype = ct.c_int
        self._lib.probe_single.argtypes = [ct.c_void_p, ct.c_int, ct.c_void_p, ct.c_void_p]
        self._lib.probe_single.restype = ct.c_int
        self._owner = os.getpid()
        self._lock = threading.RLock()
        handle = self._lib.probe_create_pool(self.num_threads)
        if not handle:
            self._check(-1)
        self._handle = handle
        self._finalizer = weakref.finalize(self, _release, self._lib, handle, self._owner)

    @property
    def identity(self) -> dict[str, Any]:
        """Build identity is independent of the temporary output path used during compilation."""
        return self.build["identity"]

    def _check(self, code: int) -> None:
        if code:
            raise RuntimeError(self._lib.probe_last_error().decode("utf-8", errors="replace"))

    def _ensure_open(self) -> None:
        if os.getpid() != self._owner:
            raise RuntimeError(
                "assignment pools cannot be used after fork; create one in this process"
            )
        if not self._handle:
            raise RuntimeError("assignment pool is closed")

    def close(self) -> None:
        """Wait for any active call before releasing native workers; repeated closes are safe."""
        with self._lock:
            self._finalizer()
            self._handle = None

    def __enter__(self) -> AssignmentSolver:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _array(value: np.ndarray, dtype: Any, tail: tuple[int, ...]) -> np.ndarray:
        if value.dtype != dtype or value.ndim != len(tail) + 1 or value.shape[1:] != tail:
            raise ValueError(f"expected {dtype} [batch,{','.join(map(str, tail))}]")
        if not np.isfinite(value).all():
            raise ValueError("assignment inputs must be finite")
        return np.ascontiguousarray(value)

    @staticmethod
    def _weight(value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("readout_weight must be positive and finite")

    def _run(
        self,
        batch: int,
        prediction: np.ndarray | None,
        target: np.ndarray | None,
        costs: np.ndarray | None,
        weight: float,
        exhaustive: bool,
        profile: bool,
    ) -> Assignment:
        result = Assignment(
            np.empty(batch, np.float64),
            np.empty(batch, np.int32),
            np.empty((batch, 16), np.int32),
            np.empty((batch, 2), np.int64),
            np.empty((batch, 2), np.float64),
        )
        with self._lock:
            self._ensure_open()
            self._check(
                self._lib.probe_match(
                    self._handle,
                    None if prediction is None else prediction.ctypes.data,
                    None if target is None else target.ctypes.data,
                    None if costs is None else costs.ctypes.data,
                    batch,
                    weight,
                    exhaustive,
                    profile,
                    result.scores.ctypes.data,
                    result.masks.ctypes.data,
                    result.permutations.ctypes.data,
                    result.counters.ctypes.data,
                    result.timings.ctypes.data,
                )
            )
        return result

    def match(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        readout_weight: float = 1.0,
        *,
        profile: bool = False,
    ) -> Assignment:
        """Match detached FP32 [batch,4608] predictions to FP32 target episodes."""
        self._weight(readout_weight)
        predictions = self._array(predictions, np.float32, (OUTPUT_FEATURES,))
        targets = self._array(targets, np.float32, (OUTPUT_FEATURES,))
        if predictions.shape != targets.shape:
            raise ValueError("prediction and target batch sizes differ")
        return self._run(
            len(predictions), predictions, targets, None, readout_weight, False, profile
        )

    def match_costs(self, costs: np.ndarray, *, exhaustive: bool = False) -> Assignment:
        """Verification interface; exhaustive search is not a production config option."""
        costs = self._array(costs, np.float64, (17, 16, 16))
        return self._run(len(costs), None, None, costs, 1.0, exhaustive, False)

    def construct_costs(
        self, predictions: np.ndarray, targets: np.ndarray, readout_weight: float = 1.0
    ) -> np.ndarray:
        """Expose native cost construction for independent numerical checks."""
        self._weight(readout_weight)
        predictions = self._array(predictions, np.float32, (OUTPUT_FEATURES,))
        targets = self._array(targets, np.float32, (OUTPUT_FEATURES,))
        if predictions.shape != targets.shape:
            raise ValueError("prediction and target batch sizes differ")
        costs = np.empty((len(predictions), 17, 16, 16), dtype=np.float64)
        with self._lock:
            self._ensure_open()
            self._check(
                self._lib.probe_construct(
                    predictions.ctypes.data,
                    targets.ctypes.data,
                    len(predictions),
                    readout_weight,
                    costs.ctypes.data,
                )
            )
        return costs

    def single(self, costs: np.ndarray) -> tuple[float, np.ndarray]:
        """Verification interface for the Hungarian solver, for order 1 through 16."""
        costs = np.ascontiguousarray(costs, dtype=np.float64)
        if costs.ndim != 2 or costs.shape[0] != costs.shape[1] or not 1 <= len(costs) <= 16:
            raise ValueError("costs must be square with order 1 through 16")
        if not np.isfinite(costs).all():
            raise ValueError("costs must be finite")
        permutation = np.empty(len(costs), dtype=np.int32)
        value = np.empty(1, dtype=np.float64)
        with self._lock:
            self._ensure_open()
            self._check(
                self._lib.probe_single(
                    costs.ctypes.data, len(costs), value.ctypes.data, permutation.ctypes.data
                )
            )
        return float(value[0]), permutation
