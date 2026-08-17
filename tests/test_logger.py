import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from omegaconf import OmegaConf

from iccl.training.logger import (
    RunLogger,
    SourceRun,
    repo_relative,
    source_from_checkpoint,
    weights_artifact_name,
)

REFERENCE = {
    "entity": "an-entity",
    "project": "a-project",
    "run_id": "t03malzi",
    "name": "different-sound-6",
}


class FakeArtifact:
    def __init__(self, name: str, type: str) -> None:
        self.name = name
        self.type = type
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class FakeTable:
    def __init__(self, columns: list[str], data: list[list[Any]]) -> None:
        self.columns = columns
        self.data = data


class FakeRun:
    def __init__(self, name: str | None = "different-sound-6") -> None:
        self.name = name
        self.id = "t03malzi"
        self.entity, self.project = "an-entity", "a-project"
        self.logged: list[tuple[FakeArtifact, list[str]]] = []
        self.used: list[str] = []
        self.records: list[tuple[dict[str, Any], int]] = []

    def log_artifact(self, artifact: FakeArtifact, aliases: list[str]) -> None:
        self.logged.append((artifact, aliases))

    def use_artifact(self, reference: str) -> None:
        self.used.append(reference)

    def log(self, payload: dict[str, Any], step: int) -> None:
        self.records.append((payload, step))


class FakeWandb:
    """Stands in for the ``wandb`` module: records what ``init`` was given and
    hands back a run whose artifact calls are inspectable."""

    Artifact = FakeArtifact
    Table = FakeTable

    @staticmethod
    def Plotly(figure: Any) -> Any:
        return figure

    def __init__(self, run: FakeRun | None = None) -> None:
        self.captured: dict[str, Any] = {}
        self.run = run

    def init(self, **kwargs: Any) -> FakeRun | None:
        self.captured.update(kwargs)
        return self.run


def make_logger(out_dir: Path, source: SourceRun | None = None) -> RunLogger:
    cfg = OmegaConf.create({"wandb": {"mode": "disabled", "project": "test", "entity": None}})
    return RunLogger(cfg, out_dir, job_type="eval", source=source)


def start_with_fake_wandb(
    monkeypatch: pytest.MonkeyPatch,
    out_dir: Path,
    run: FakeRun | None = None,
    **wandb_overrides: Any,
) -> tuple[RunLogger, FakeWandb]:
    fake = FakeWandb(run)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    wandb_cfg = {"mode": "online", "project": "p", "entity": "e"} | wandb_overrides
    logger = RunLogger(OmegaConf.create({"wandb": wandb_cfg}), out_dir, job_type="train")
    logger.start()
    return logger, fake


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
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    cfg = OmegaConf.create({"wandb": {"mode": "online", "project": "p", "entity": "e"}})
    RunLogger(cfg, tmp_path, job_type="eval", source=SourceRun(step=100000, **REFERENCE)).start()
    captured = fake.captured

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


def test_repo_relative_paths_strip_the_repo_root(tmp_path: Path) -> None:
    import iccl

    root = Path(iccl.__file__).resolve().parents[2]
    run_dir = root / "outputs" / "2026-08-01" / "18-31-02"
    assert repo_relative(run_dir) == str(Path("outputs") / "2026-08-01" / "18-31-02")
    # A run dir outside the repo has no relative form; report it in full.
    assert repo_relative(tmp_path / "run") == str((tmp_path / "run").resolve())


def test_the_run_records_where_it_writes_its_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overview page is otherwise silent about the run dir, leaving no way
    to get from a W&B run to the checkpoints it produced."""
    _, fake = start_with_fake_wandb(monkeypatch, tmp_path / "run")
    paths = fake.captured["config"]["paths"]
    assert paths["run_dir"].endswith("run")
    assert paths["checkpoints"].endswith(str(Path("run") / "checkpoints"))
    assert paths["snapshots"].endswith(str(Path("run") / "snapshots"))


def test_a_run_can_be_named_at_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, named = start_with_fake_wandb(monkeypatch, tmp_path, name="pilot-logsnapshots")
    assert named.captured["name"] == "pilot-logsnapshots"
    # Unset, W&B generates one; None is its documented value for that.
    _, unnamed = start_with_fake_wandb(monkeypatch, tmp_path)
    assert unnamed.captured["name"] is None


def test_final_weights_upload_as_a_versioned_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "step_0100000.pt"
    snapshot.write_bytes(b"weights")
    run = FakeRun()
    logger, _ = start_with_fake_wandb(monkeypatch, tmp_path, run=run, upload_weights=True)
    logger.upload_weights(snapshot)

    artifact, aliases = run.logged[0]
    assert artifact.name == "weights-different-sound-6"
    assert (artifact.type, aliases) == ("model", ["final"])
    assert artifact.files == [str(snapshot)]


def test_uploading_is_opt_in_and_needs_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "step_0100000.pt"
    snapshot.write_bytes(b"weights")

    run = FakeRun()
    logger, _ = start_with_fake_wandb(monkeypatch, tmp_path, run=run, upload_weights=False)
    logger.upload_weights(snapshot)
    assert run.logged == []

    # W&B disabled: no run to attach an artifact to, and the flag is moot.
    disabled = make_logger(tmp_path)
    disabled.start()
    disabled.upload_weights(snapshot)


def test_artifact_names_survive_unnamed_and_awkwardly_named_runs() -> None:
    # An offline run is unnamed until it syncs; its id still identifies it.
    assert weights_artifact_name("t03malzi") == "weights-t03malzi"
    # `wandb.name` is free text from a hydra override, but artifact names are not.
    assert weights_artifact_name("pilot run #2") == "weights-pilot-run--2"


def test_capability_rows_write_tables_and_named_panels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = FakeRun()
    logger, _ = start_with_fake_wandb(monkeypatch, tmp_path, run=run)
    summary = [
        {
            "capability": "icl",
            "condition": "ordinary",
            "slice": "fixed_surplus",
            "suite": "icl-cell",
            "status": "seen",
            "sampler": "constructive",
            "weighting": "discrete",
            "M": 8,
            "T": 8,
            "S": 1,
            "B_history": 256,
            "L_history": 520,
            "D_target": 32,
            "D_min": 32.0,
            "D_max": 32.0,
            "D_mean": 32.0,
            "D_cv": 0.0,
            "constituent_exposure_min": None,
            "constituent_exposure_max": None,
            "intervening_tasks": None,
            "prediction_token_delay": None,
            "serialized_token_delay": None,
            "metric": "nmse_aulc",
            "value": 0.4,
            "ci_low": 0.3,
            "ci_high": 0.5,
            "n_sequences": 16,
        }
    ]
    curve_rows = [
        {
            "capability": "icl",
            "condition": "ordinary",
            "slice": "fixed_surplus",
            "status": "seen",
            "M": 8,
            "T": 8,
            "S": 1,
            "B_history": 256,
            "L_history": 520,
            "curve_type": "learning_curve",
            "x_name": "demo_index",
            "x_value": 0,
            "mse": 0.8,
            "nmse": 0.4,
            "ci_low": 0.3,
            "ci_high": 0.5,
            "n_sequences": 16,
        }
    ]
    logger.log_eval({}, {}, 20, summary_rows=summary, curve_rows=curve_rows)

    assert (tmp_path / "eval" / "step_0000020_summary.json").exists()
    payload, step = run.records[-1]
    assert step == 20
    assert "eval-tables/capability_summary" in payload
    assert "eval-tables/capability_curves" in payload
    assert "capability/icl/aulc_vs_M" in payload
    assert "capability/icl/learning_curve" in payload
