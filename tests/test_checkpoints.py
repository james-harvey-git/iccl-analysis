import builtins
import sys
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from iccl.checkpoints import (
    evaluation_checkpoint_references,
    resolve_checkpoint_path,
    validate_evaluation_config,
)


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


def test_evaluation_requires_its_own_checkpoint_list() -> None:
    cfg = OmegaConf.create(
        {"evaluation": {"checkpoints": []}, "training": {"resume": "training-state.pt"}}
    )

    with pytest.raises(ValueError, match="evaluation.checkpoints"):
        evaluation_checkpoint_references(cfg)


def test_evaluation_preserves_the_configured_trajectory_order() -> None:
    cfg = OmegaConf.create({"evaluation": {"checkpoints": ["step_2.pt", "step_10.pt"]}})

    assert evaluation_checkpoint_references(cfg) == ["step_2.pt", "step_10.pt"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("model.n_heads", 4),
        ("model.norm_eps", 0.01),
        ("data.input_dim", 8),
        ("data.hidden_dims", [32]),
        ("data.use_bias", False),
        ("data.scale", 2.0),
        ("data.sequence.signal_boundaries", False),
    ],
)
def test_evaluation_rejects_incompatible_checkpoint_config(key: str, value: Any) -> None:
    cfg = OmegaConf.create(
        {
            "model": {"n_heads": 2, "norm_eps": 1e-5, "backend": "auto"},
            "data": {
                "input_dim": 4,
                "output_dim": 4,
                "hidden_dims": [4],
                "use_bias": True,
                "scale": 1.0,
                "num_modules": 8,
                "sequence": {"signal_boundaries": True},
                "eval_sets": {"retention": {"controls": ["novel", "shared"]}},
            },
        }
    )
    checkpoint = {"config": OmegaConf.to_container(cfg, resolve=True)}
    cfg.data.eval_sets.retention.controls = ["unexposed", "shared"]
    cfg.data.num_modules = 10
    cfg.model.backend = "reference"
    validate_evaluation_config(checkpoint, cfg)
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError, match="differs from the checkpoint"):
        validate_evaluation_config(checkpoint, cfg)
