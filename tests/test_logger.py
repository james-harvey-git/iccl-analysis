import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from iccl.training.logger import RunLogger, SourceRun, source_from_checkpoint

REFERENCE = {
    "entity": "an-entity",
    "project": "a-project",
    "run_id": "t03malzi",
    "name": "different-sound-6",
}


def make_logger(out_dir: Path, source: SourceRun | None = None) -> RunLogger:
    cfg = OmegaConf.create({"wandb": {"mode": "disabled", "project": "test", "entity": None}})
    return RunLogger(cfg, out_dir, job_type="eval", source=source)


def test_log_eval_writes_curves_and_namespaces_scalars(tmp_path: Path) -> None:
    logger = make_logger(tmp_path)
    logger.start()
    path = logger.log_eval(
        {"retention/savings_mean": 0.33}, {"retention/savings_curve": np.array([0.1, 0.2])}, 40
    )
    assert path == tmp_path / "eval" / "step_0000040.npz"
    with np.load(path) as data:
        # Slashes are not valid in npz member names, so curve keys are dotted.
        np.testing.assert_allclose(data["retention.savings_curve"], [0.1, 0.2])
    assert logger.run_reference() is None


def test_source_round_trips_through_a_checkpoint() -> None:
    source = source_from_checkpoint({"step": 100000, "wandb_run": REFERENCE})
    assert source == SourceRun(step=100000, **REFERENCE)
    assert source is not None
    assert source.label == "different-sound-6"
    assert source.url == "https://wandb.ai/an-entity/a-project/runs/t03malzi"


def test_an_unsynced_offline_source_still_resolves() -> None:
    """A run trained offline — the cluster case — has no name until it syncs,
    but its id is preserved, so the link and label are still recoverable."""
    source = source_from_checkpoint({"step": 40, "wandb_run": {**REFERENCE, "name": None}})
    assert source is not None
    assert source.label == "t03malzi"
    assert source.url == "https://wandb.ai/an-entity/a-project/runs/t03malzi"


def test_checkpoints_without_a_run_reference_stand_alone() -> None:
    """Checkpoints trained with W&B disabled, and those written before the
    reference was recorded, carry no source — evaluation must still run."""
    assert source_from_checkpoint({"step": 4, "wandb_run": None}) is None
    assert source_from_checkpoint({"step": 100000}) is None


def test_a_source_run_becomes_a_link_a_tag_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class FakeWandb:
        def init(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb())
    cfg = OmegaConf.create({"wandb": {"mode": "online", "project": "p", "entity": "e"}})
    RunLogger(cfg, tmp_path, job_type="eval", source=SourceRun(step=100000, **REFERENCE)).start()

    assert captured["job_type"] == "eval"
    assert captured["tags"] == ["eval", "source:different-sound-6"]
    # Notes render as markdown in the run overview, so the source run is a
    # clickable link there; the config copy is what makes evals filterable.
    assert captured["notes"] == (
        "Checkpoint from [different-sound-6]"
        "(https://wandb.ai/an-entity/a-project/runs/t03malzi) at step 100000."
    )
    assert captured["config"]["source_run"]["run_id"] == "t03malzi"
    assert captured["config"]["source_run"]["url"].endswith("/runs/t03malzi")


def test_a_source_run_does_not_disturb_a_disabled_logger(tmp_path: Path) -> None:
    logger = make_logger(tmp_path, source=SourceRun(step=100000, **REFERENCE))
    logger.start()
    logger.log_eval({"in_dist/nmse_mean": 0.5}, {}, 100000)
    logger.finish()
    assert logger.run is None
