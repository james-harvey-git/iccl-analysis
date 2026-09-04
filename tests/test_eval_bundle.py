import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from iccl.data.eval_bundle import (
    prepare_eval_bundle,
    select_evaluation_suite,
    validate_eval_bundle,
)
from iccl.evaluation.metrics import _evaluate, demo_mse, demo_nmse, load_eval_suites
from iccl.reporting.figures import evaluation_figures
from iccl.training.trainer import is_monitor_suite


@pytest.fixture(scope="module")
def frozen_bundle(tmp_path_factory: pytest.TempPathFactory) -> DictConfig:
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.eval_sets.out_dir={tmp_path_factory.mktemp('bundle') / 'active'}",
                "data.input_dim=4",
                "data.output_dim=4",
                "data.hidden_dims=[4]",
                "data.eval_sets.module_counts.min=8",
                "data.eval_sets.module_counts.max=8",
                "data.eval_sets.num_sequences=8",
                "data.eval_sets.demos_per_task=2",
                "data.eval_sets.task_variation.surplus_tasks.max=1",
                "data.eval_sets.retention.position_diagnostic.num_worlds=2",
            ],
        )
    prepare_eval_bundle(cfg)
    return cfg


@pytest.fixture
def bundle(
    frozen_bundle: DictConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(frozen_bundle, resolve=True))
    assert isinstance(cfg, DictConfig)
    cfg.data.eval_sets.out_dir = str(tmp_path / "active")
    shutil.copytree(frozen_bundle.data.eval_sets.out_dir, cfg.data.eval_sets.out_dir)
    monkeypatch.chdir(tmp_path)
    return cfg


def test_one_bundle_serves_every_consumer(bundle: DictConfig) -> None:
    manifest = validate_eval_bundle(bundle)
    root = Path(bundle.data.eval_sets.out_dir)
    assert manifest["num_suites"] == 21  # validation + two cells x seven + six diagnostic
    assert not any(path.is_dir() for path in root.iterdir())
    for selection, expected in [("all", 21), ("capabilities", 14), ("retention_position", 6)]:
        suites = load_eval_suites(
            root, select=lambda meta, selection=selection: select_evaluation_suite(meta, selection)
        )
        assert len(suites) == expected
    assert len(load_eval_suites(root, select=is_monitor_suite)) == 7
    with pytest.raises(ValueError, match="evaluation.suites"):
        select_evaluation_suite({}, "typo")


def test_matching_bundle_is_reused_across_training_and_reporting_changes(
    bundle: DictConfig,
) -> None:
    root = Path(bundle.data.eval_sets.out_dir)
    times = {path.name: path.stat().st_mtime_ns for path in root.iterdir()}
    bundle.training.num_steps = 400000
    bundle.training.lr = 0.001
    bundle.wandb.name = "long-run"
    bundle.data.eval_sets.bootstrap_replicates = 200
    bundle.data.distribution_diagnostic.num_sequences = 20
    assert prepare_eval_bundle(bundle) == root
    assert times == {path.name: path.stat().st_mtime_ns for path in root.iterdir()}
    assert not Path("outputs").exists()


def test_capabilities_and_diagnostics_share_one_report(bundle: DictConfig) -> None:
    suites = load_eval_suites(
        Path(bundle.data.eval_sets.out_dir), select=lambda meta: meta["capability"] != "validation"
    )
    errors, normalized = {}, {}
    for name, suite in suites.items():
        predictions = np.zeros_like(suite["targets"])
        errors[name] = demo_mse(predictions, suite)
        normalized[name] = demo_nmse(predictions, suite)
    report = _evaluate(suites, errors, normalized, {}, bootstrap_seed=0, bootstrap_replicates=0)
    figures = evaluation_figures(report.summary_rows, report.curve_rows)
    assert {
        "evaluation/icl_within_task",
        "evaluation/retention_controls",
        "evaluation/retention_position",
        "evaluation/retention_rehearsal",
    } <= figures.keys()


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", 12),
        ("data.sequence.surplus_tasks", [0, 4]),
        ("data.eval_sets.retention.position_diagnostic.num_worlds", 4),
    ],
)
def test_different_generation_config_is_rejected(
    bundle: DictConfig, field: str, value: object
) -> None:
    OmegaConf.update(bundle, field, value)
    with pytest.raises(ValueError, match="does not match.*data/seed"):
        validate_eval_bundle(bundle)


@pytest.mark.parametrize("damage", ["missing", "corrupt", "extra", "version"])
def test_bundle_integrity_is_checked(bundle: DictConfig, damage: str) -> None:
    root = Path(bundle.data.eval_sets.out_dir)
    if damage == "missing":
        next(root.glob("*.npz")).unlink()
    elif damage == "corrupt":
        next(root.glob("*.npz")).write_bytes(b"corrupt")
    elif damage == "extra":
        (root / "obsolete-version").mkdir()
    else:
        path = root / "manifest.json"
        data = json.loads(path.read_text())
        data["bundle_version"] = -1
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="make_eval_sets"):
        validate_eval_bundle(bundle)


def test_successful_replacement_preserves_the_entire_old_directory(bundle: DictConfig) -> None:
    root = Path(bundle.data.eval_sets.out_dir)
    (root / "old-version").mkdir()
    (root / "old-version" / "notes.txt").write_text("preserve me")
    prepare_eval_bundle(bundle)
    validate_eval_bundle(bundle)
    assert not (root / "old-version").exists()
    backups = list(Path("outputs/eval-set-backups").iterdir())
    assert len(backups) == 1
    assert (backups[0] / "old-version" / "notes.txt").read_text() == "preserve me"
    assert (backups[0] / "validation.npz").exists()


def test_failed_generation_leaves_the_active_bundle_untouched(
    bundle: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(bundle.data.eval_sets.out_dir)
    old_manifest = (root / "manifest.json").read_bytes()
    bundle.seed += 1

    def fail(*args: object, **kwargs: object) -> int:
        raise RuntimeError("generation failed")

    monkeypatch.setattr("iccl.data.eval_bundle.export_retention_position_sets", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        prepare_eval_bundle(bundle)
    assert (root / "manifest.json").read_bytes() == old_manifest
    assert sorted(path.name for path in root.parent.iterdir()) == ["active"]


def test_build_lock_prevents_simultaneous_replacement(bundle: DictConfig) -> None:
    root = Path(bundle.data.eval_sets.out_dir)
    root.with_name(f".{root.name}.lock").mkdir()
    with pytest.raises(RuntimeError, match="locked"):
        prepare_eval_bundle(bundle)


def test_failed_publication_restores_the_previous_bundle(
    bundle: DictConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(bundle.data.eval_sets.out_dir)
    original_manifest = (root / "manifest.json").read_bytes()
    bundle.seed += 1

    def fail_rename(self: Path, target: Path) -> Path:
        raise OSError("publication failed")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError, match="publication failed"):
        prepare_eval_bundle(bundle)
    assert (root / "manifest.json").read_bytes() == original_manifest
    bundle.seed -= 1
    validate_eval_bundle(bundle)


def test_project_root_cannot_be_replaced(bundle: DictConfig) -> None:
    bundle.data.eval_sets.out_dir = str(Path.cwd())
    with pytest.raises(ValueError, match="dedicated"):
        prepare_eval_bundle(bundle)
