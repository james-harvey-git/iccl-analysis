import builtins
import sys
from pathlib import Path
from typing import Any

import pytest

from iccl.checkpoints import resolve_checkpoint_path


class FakeArtifact:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def download(self) -> str:
        return str(self.directory)


class FakeApi:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.calls: list[tuple[str, str]] = []

    def artifact(self, reference: str, *, type: str) -> FakeArtifact:
        self.calls.append((reference, type))
        return FakeArtifact(self.directory)


class FakeWandb:
    def __init__(self, api: FakeApi) -> None:
        self.api = api

    def Api(self) -> FakeApi:
        return self.api


def test_local_checkpoint_resolution_does_not_import_wandb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "last.pt"
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb":
            raise AssertionError("local checkpoint resolution imported wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert resolve_checkpoint_path(str(checkpoint)) == (checkpoint, False)


def test_wandb_reference_downloads_one_model_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "step_0100000.pt"
    checkpoint.write_bytes(b"weights")
    api = FakeApi(tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb(api))

    resolved = resolve_checkpoint_path("wandb://entity/project/weights:v3")

    assert resolved == (checkpoint, True)
    assert api.calls == [("entity/project/weights:v3", "model")]


@pytest.mark.parametrize("names", [[], ["first.pt", "second.pt"]])
def test_wandb_reference_requires_exactly_one_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    names: list[str],
) -> None:
    for name in names:
        (tmp_path / name).write_bytes(b"weights")
    monkeypatch.setitem(sys.modules, "wandb", FakeWandb(FakeApi(tmp_path)))

    with pytest.raises(ValueError, match=rf"holds {len(names)} checkpoints"):
        resolve_checkpoint_path("wandb://entity/project/weights:v3")
