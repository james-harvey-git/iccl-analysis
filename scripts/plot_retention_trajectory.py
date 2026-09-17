"""Matched retention-position trajectories from local training snapshots.

Only the exact-repeat and novel-pair conditions of the paired-permutation
diagnostic are scored. The existing evaluator defines total savings and performs
the whole-world bootstrap. Plots show total savings, novel-task error and
repeat-task error against the number of intervening tasks, without smoothing.
The two error plots use the same y-axis limits and all plots share a colour scale.

Run from the repository root, using the same code and frozen evaluation bundle
as the report. No W&B connection or checkpoint upload is needed.

    uv run python scripts/plot_retention_trajectory.py --mode discover
    uv run python scripts/plot_retention_trajectory.py
    uv run python scripts/plot_retention_trajectory.py --mode plot --cmap magma_r

Discovery runs on CPU. Run evaluation in a cluster GPU allocation. Completed
checkpoints are cached, so rerunning the same command resumes after interruption.
Plot mode needs only the saved results and can run without a GPU or snapshots.
The defaults select 100k through 2100k at 100k intervals from run 0bdfkn9e.
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import iccl.evaluation.metrics as evaluation_metrics
from iccl.checkpoints import source_from_checkpoint
from iccl.data.eval_bundle import validate_eval_bundle
from iccl.evaluation.metrics import METRIC_VERSION, _aggregate, evaluate_suites, load_eval_suites
from iccl.evaluation.results import read_rows, write_evaluation_results
from iccl.evaluation.retention_position import _matrix
from iccl.models.model import model_from_config
from iccl.training.trainer import resolve_autocast_dtype
from iccl.utils import resolve_device, seed_everything


@dataclass(frozen=True)
class Snapshot:
    """A verified local checkpoint and its model-content identity."""

    step: int
    path: Path
    model_sha256: str


def checkpoint_model_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Architecture and token dimensions, independent of execution backend."""
    cfg = checkpoint["config"]
    model = dict(cfg["model"])
    model.pop("backend", None)
    return {
        "model": model,
        "data": {key: cfg["data"][key] for key in ("input_dim", "output_dim")},
    }


def checkpoint_model_digest(checkpoint: dict[str, Any]) -> str:
    """Hash architecture and tensors rather than serialization or optimizer state."""
    digest = hashlib.sha256(
        json.dumps(checkpoint_model_config(checkpoint), sort_keys=True).encode()
    )
    for name, tensor in sorted(checkpoint["model"].items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def discover_snapshot_series(
    roots: list[Path],
    steps: list[int],
    source_run: str,
    *,
    explicit_paths: list[Path] | None = None,
) -> list[Snapshot]:
    """Select exact steps from one entity/project/run across Hydra stage directories.

    Explicit paths pin particular steps when a run has conflicting snapshots.
    Otherwise byte-identical model states are deduplicated, and missing steps or
    conflicting weights fail before evaluation. Only requested snapshot filenames
    are loaded during recursive discovery. Metadata, not filenames, is authoritative.
    """
    if len(source_run.split("/")) != 3 or any(not part for part in source_run.split("/")):
        raise ValueError("--source-run must be entity/project/run_id")
    if not steps or steps != sorted(set(steps)) or steps[0] <= 0:
        raise ValueError("steps must be unique positive integers in increasing order")
    wanted = set(steps)
    pins: dict[int, Snapshot] = {}
    candidates: dict[int, list[Snapshot]] = {}

    def inspect(path: Path, *, explicit: bool, named_step: int | None = None) -> Snapshot | None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source = source_from_checkpoint(checkpoint)
        identity = None if source is None else f"{source.entity}/{source.project}/{source.run_id}"
        if identity != source_run:
            if explicit:
                raise ValueError(f"{path} belongs to {identity}, expected {source_run}")
            return None
        step = int(checkpoint["step"])
        if named_step is not None and step != named_step:
            raise ValueError(f"{path} says step {named_step} but contains step {step}")
        if step not in wanted:
            raise ValueError(f"{path} contains unrequested step {step}")
        return Snapshot(step, path.resolve(), checkpoint_model_digest(checkpoint))

    for path in explicit_paths or []:
        snapshot = inspect(path.expanduser(), explicit=True)
        assert snapshot is not None
        if snapshot.step in pins:
            raise ValueError(f"multiple explicit paths for step {snapshot.step}")
        pins[snapshot.step] = snapshot
    visited: set[Path] = set()
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"snapshot search root does not exist: {root}")
        for path in sorted(root.rglob("step_*.pt")):
            suffix = path.stem.removeprefix("step_")
            if not suffix.isdigit() or int(suffix) not in wanted or int(suffix) in pins:
                continue
            resolved = path.resolve()
            if resolved in visited:
                continue
            visited.add(resolved)
            snapshot = inspect(path, explicit=False, named_step=int(suffix))
            if snapshot is not None:
                candidates.setdefault(snapshot.step, []).append(snapshot)
    selected: list[Snapshot] = []
    missing = [step for step in steps if step not in candidates and step not in pins]
    if missing:
        raise FileNotFoundError(f"Missing snapshots for {source_run} at steps {missing}")
    for step in steps:
        if step in pins:
            selected.append(pins[step])
            continue
        matches = candidates[step]
        if len({item.model_sha256 for item in matches}) != 1:
            paths = ", ".join(str(item.path) for item in matches)
            raise ValueError(
                f"Conflicting weights at step {step}: {paths}. Pin with --checkpoint-paths."
            )
        selected.append(matches[0])
    return selected


def requested_steps(cfg: argparse.Namespace) -> list[int]:
    """An inclusive exact interval, with no implicit interpolation or nearest steps."""
    start, stop, every = (int(getattr(cfg, key)) for key in ("start", "stop", "every"))
    if start <= 0 or stop < start or every <= 0 or (stop - start) % every:
        raise ValueError("steps require 0 < start <= stop, every > 0, and stop on the interval")
    return list(range(start, stop + 1, every))


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _implementation_digest() -> str:
    root = Path(evaluation_metrics.__file__).resolve().parents[1]
    files = sorted(
        path for folder in ("models", "evaluation") for path in (root / folder).glob("*.py")
    )
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def position_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract one complete paired-permutation total-savings curve."""
    selected = sorted(
        (
            row
            for row in rows
            if row["capability"] == "retention_position"
            and row["diagnostic_family"] == "paired_permutation"
            and row["curve_type"] == "retention_position"
            and row["retention_component"] == "total"
        ),
        key=lambda row: row["x_value"],
    )
    if not selected:
        raise ValueError("No paired-permutation total-savings position curve found")
    cells = {(row["M"], row["T"], row["D"], row["n_sequences"]) for row in selected}
    if len(cells) != 1 or [r["x_value"] for r in selected] != list(range(selected[0]["T"])):
        raise ValueError("Expected exactly one complete curve over all history-task positions")
    if not all(np.isfinite(row[key]) for row in selected for key in ("nmse", "ci_low", "ci_high")):
        raise ValueError("Position curve contains non-finite values")
    return selected


def cached_curve(
    path: Path, expected: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Reuse completed results only when their identity and file checksums agree."""
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    identity = manifest.get("trajectory_identity")
    if identity is None or (expected is not None and identity != expected):
        raise ValueError(f"Incompatible cached evaluation in {path}. Use a different --out-dir")
    checksums = manifest.get("trajectory_files", {})
    if set(checksums) != {"curves.csv", "summary.csv", "raw_errors.npz", "scalars.json"}:
        raise ValueError(f"Incomplete cached evaluation in {path}")
    for name, checksum in checksums.items():
        file = path / name
        if not file.is_file() or _digest(file) != checksum:
            raise ValueError(f"Cached result checksum mismatch: {file}")
    rows = position_rows(read_rows(path / "curves.csv"))
    if {row["step"] for row in rows} != {identity["step"]}:
        raise ValueError(f"Cached curve step disagrees with its manifest: {path}")
    return identity, rows


def condition_curves(
    path: Path, identity: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Recover final-task errors, averaging demos within each paired world.

    Columns follow ascending original task position. Both conditions use the same
    whole-world bootstrap draws, preserving their pairing across all positions.
    Only cached arrays and metadata are needed, without snapshots or frozen tokens.
    """
    manifest = json.loads((path / "manifest.json").read_text())
    names = {meta["logical_name"]: name for name, meta in manifest["raw_arrays"].items()}
    prefix = "retention_position/paired_permutation"
    curves = {}
    with np.load(path / "raw_errors.npz", allow_pickle=False) as raw:
        groups = raw[names[f"{prefix}/position_group_id"]]
        positions = raw[names[f"{prefix}/original_task_position"]]
        for condition in ("novel", "repeat"):
            suites = [
                name
                for name, meta in manifest["suites"].items()
                if meta.get("capability") == "retention_position"
                and meta.get("diagnostic_family") == "paired_permutation"
                and meta.get("condition") == condition
            ]
            if len(suites) != 1:
                raise ValueError(f"Expected one paired-permutation {condition} suite in {path}")
            errors = raw[names[f"{suites[0]}/nmse"]]
            expected_shape = (len(groups), rows[0]["T"] + 1, rows[0]["D"])
            if errors.shape != expected_shape or not np.isfinite(errors[:, -1]).all():
                raise ValueError(f"Invalid cached {condition} errors in {path}")
            matrix, coordinates = _matrix(errors[:, -1].mean(axis=1), groups, positions)
            if matrix.shape[0] != rows[0]["n_sequences"] or not np.array_equal(
                coordinates, [row["x_value"] for row in rows]
            ):
                raise ValueError(f"Cached {condition} worlds/positions disagree with {path}")
            curves[condition] = _aggregate(
                matrix,
                seed=int(identity["bootstrap_seed"]),
                replicates=int(identity["bootstrap_replicates"]),
            )
    if not np.allclose(
        curves["novel"][0] - curves["repeat"][0],
        [row["nmse"] for row in rows],
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError(f"Novel minus repeat errors disagree with saved total savings in {path}")
    return curves


def plot_retention_trajectory(
    root: Path, steps: list[int], source_run: str, *, cmap: str = "viridis_r", show_ci: bool = False
) -> list[Path]:
    """Draw bright-to-dark curves with training step mapped to a continuous colourbar."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    if not steps:
        raise ValueError("At least one checkpoint is required for plotting")
    series = []
    common = None
    delays = None
    for step in steps:
        step_dir = root / "evaluation-results" / f"step_{step:07d}"
        cached = cached_curve(step_dir)
        if cached is None:
            raise FileNotFoundError(f"No completed evaluation for step {step}")
        identity, rows = cached
        if identity["step"] != step or identity["source_run"] != source_run:
            raise ValueError(f"Cached provenance disagrees with requested run/step {step}")
        comparison = {k: v for k, v in identity.items() if k not in {"step", "model_sha256"}}
        if common is not None and comparison != common:
            raise ValueError(
                "Cannot combine curves from different suites, models or evaluation settings"
            )
        common = comparison
        coordinates = np.array([row["intervening_tasks"] for row in rows])
        if not np.array_equal(coordinates, rows[0]["T"] - 1 - np.arange(len(rows))):
            raise ValueError(
                f"Intervening-task counts disagree with original positions in {step_dir}"
            )
        if delays is not None and not np.array_equal(coordinates, delays):
            raise ValueError("Cannot combine curves with different intervening-task counts")
        delays = coordinates
        series.append(
            {
                "total": tuple(
                    np.array([row[key] for row in rows]) for key in ("nmse", "ci_low", "ci_high")
                ),
                **condition_curves(step_dir, identity, rows),
            }
        )

    assert delays is not None
    order = np.argsort(delays)
    x = delays[order]
    error_max = max(
        float(curves[condition][2 if show_ci else 0].max())
        for curves in series
        for condition in ("novel", "repeat")
    )
    plots = (
        ("total", "retention-position-trajectory", "Total savings (nMSE)"),
        ("novel", "retention-position-novel-trajectory", r"$E^{\mathrm{novel}}$ (nMSE)"),
        ("repeat", "retention-position-repeat-trajectory", r"$E^{\mathrm{repeat}}$ (nMSE)"),
    )
    paths = []

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        norm = Normalize(
            min(steps) / 1000, max(steps) / 1000 if len(steps) > 1 else steps[0] / 1000 + 1
        )
        colours = plt.get_cmap(cmap)
        for condition, stem, ylabel in plots:
            fig, ax = plt.subplots(figsize=(6.3, 3.7), layout="constrained")
            for step, curves in zip(steps, series, strict=True):
                mean, low, high = curves[condition]
                colour = colours(norm(step / 1000))
                ax.plot(x, mean[order], color=colour, lw=1.5, marker="o", markersize=2.3)
                if show_ci:
                    ax.fill_between(
                        x, low[order], high[order], color=colour, alpha=0.07, linewidth=0
                    )
            ax.axhline(0, color="0.55", lw=0.6, ls="--", zorder=0)
            ax.set(xlabel="Number of intervening tasks", ylabel=ylabel, xticks=x)
            if condition != "total":
                ax.set_ylim(0, max(error_max * 1.05, 1e-6))
            ax.grid(axis="y", color="0.91", linewidth=0.5)
            ax.set_axisbelow(True)
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colours), ax=ax, pad=0.035)
            bar.set_label("Training steps (thousands)")
            ticks = sorted(
                {steps[0] / 1000, *[s / 1000 for s in steps if s % 500000 == 0], steps[-1] / 1000}
            )
            bar.set_ticks(ticks)
            for suffix in ("pdf", "png"):
                path = root / f"{stem}.{suffix}"
                fig.savefig(path, dpi=250, bbox_inches="tight")
                paths.append(path)
            plt.close(fig)
    return paths


def _frozen_position_suites(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify and reuse the bundle under its own recorded generation settings."""
    manifest = json.loads((root / "manifest.json").read_text())
    cfg = OmegaConf.create(manifest["generation_config"])
    assert isinstance(cfg, DictConfig)
    cfg.data.eval_sets.out_dir = str(root)
    bundle = validate_eval_bundle(cfg)
    suites = load_eval_suites(
        root,
        select=lambda meta: (
            meta.get("capability") == "retention_position"
            and meta.get("diagnostic_family") == "paired_permutation"
            and meta.get("condition") in {"repeat", "novel"}
        ),
    )
    if len(suites) != 2:
        raise ValueError("Expected one paired-permutation repeat/novel suite pair")
    return bundle, suites


def evaluate_retention_trajectory(
    cfg: argparse.Namespace, snapshots: list[Snapshot], root: Path
) -> None:
    """Resume verified per-step evaluations without any W&B initialization."""
    bundle, suites = _frozen_position_suites(Path(cfg.eval_dir).expanduser())
    device = resolve_device(str(cfg.device))
    dtype = resolve_autocast_dtype(str(cfg.precision), device)
    first = torch.load(snapshots[0].path, map_location="cpu", weights_only=False)
    architecture = checkpoint_model_config(first)
    model_cfg = OmegaConf.create(architecture)
    model_cfg.model.backend = str(cfg.backend)
    for suite in suites.values():
        if (
            suite["tokens"].shape[-1] != max(architecture["data"].values())
            or suite["targets"].shape[-1] != architecture["data"]["output_dim"]
        ):
            raise ValueError("Checkpoint token dimensions do not match the frozen evaluation data")
    del first
    seed_everything(0)
    model = None
    common = {
        "schema": 1,
        "source_run": str(cfg.source_run),
        "metric_version": METRIC_VERSION,
        "implementation_sha256": _implementation_digest(),
        "architecture": architecture,
        "suite_files": {
            f"{name}{ext}": bundle["files"][f"{name}{ext}"]
            for name in suites
            for ext in (".npz", ".meta.json")
        },
        "bootstrap_seed": int(cfg.bootstrap_seed),
        "bootstrap_replicates": int(cfg.bootstrap_replicates),
        "batch_size": int(cfg.batch_size),
        "backend": str(cfg.backend),
        "device_type": device.type,
        "precision": str(dtype),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
    }
    results = root / "evaluation-results"
    results.mkdir(parents=True, exist_ok=True)
    for index, snapshot in enumerate(snapshots, start=1):
        identity = common | {"step": snapshot.step, "model_sha256": snapshot.model_sha256}
        step_dir = results / f"step_{snapshot.step:07d}"
        if cached_curve(step_dir, identity) is not None:
            print(f"[{index}/{len(snapshots)}] Reusing step {snapshot.step}", flush=True)
            continue
        if step_dir.exists():
            raise ValueError(f"Incomplete results in {step_dir}. Use a different --out-dir")
        checkpoint = torch.load(snapshot.path, map_location="cpu", weights_only=False)
        if checkpoint_model_config(checkpoint) != architecture:
            raise ValueError(f"Architecture changes at {snapshot.path}")
        if (
            int(checkpoint["step"]) != snapshot.step
            or checkpoint_model_digest(checkpoint) != snapshot.model_sha256
        ):
            raise ValueError(f"Checkpoint changed after discovery: {snapshot.path}")
        if model is None:
            model = model_from_config(model_cfg).to(device).eval()
        model.load_state_dict(checkpoint["model"])
        source = source_from_checkpoint(checkpoint)
        print(f"[{index}/{len(snapshots)}] Evaluating step {snapshot.step} on {device}", flush=True)
        report = evaluate_suites(
            model,
            suites,
            device,
            batch_size=int(cfg.batch_size),
            autocast_dtype=dtype,
            bootstrap_seed=int(cfg.bootstrap_seed),
            bootstrap_replicates=int(cfg.bootstrap_replicates),
        )
        position_rows(report.curve_rows)
        with TemporaryDirectory(prefix=f".step_{snapshot.step:07d}-", dir=results) as temporary:
            staged = write_evaluation_results(
                report,
                Path(temporary),
                snapshot.step,
                {
                    "checkpoint_reference": str(snapshot.path),
                    "source_run": None if source is None else asdict(source),
                    "trajectory_identity": identity,
                    "eval_bundle": bundle,
                    "suites": {name: suite["__meta__"] for name, suite in suites.items()},
                },
            )
            manifest_path = staged / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["trajectory_files"] = {
                name: _digest(staged / name)
                for name in ("curves.csv", "summary.csv", "raw_errors.npz", "scalars.json")
            }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
            staged.rename(step_dir)
        del checkpoint, report


def run_retention_trajectory(cfg: argparse.Namespace) -> None:
    """Discover, resume scoring and plot without modifying repository configuration."""
    if cfg.mode not in {"discover", "evaluate", "plot"}:
        raise ValueError("mode must be discover, evaluate or plot")
    if not cfg.source_run:
        raise ValueError("Set --source-run entity/project/run_id")
    steps = requested_steps(cfg)
    root = Path(cfg.out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if cfg.mode != "plot":
        snapshots = discover_snapshot_series(
            [Path(path) for path in cfg.roots],
            steps,
            str(cfg.source_run),
            explicit_paths=[Path(path) for path in cfg.checkpoint_paths],
        )
        for snapshot in snapshots:
            print(f"{snapshot.step:>9}  {snapshot.path}", flush=True)
        (root / "checkpoints.json").write_text(
            json.dumps(
                {
                    "source_run": str(cfg.source_run),
                    "checkpoints": [asdict(snapshot) for snapshot in snapshots],
                },
                indent=2,
                default=str,
            )
        )
        if cfg.mode == "discover":
            print(
                f"Verified all {len(snapshots)} checkpoints. Inventory: {root / 'checkpoints.json'}"
            )
            return
        evaluate_retention_trajectory(cfg, snapshots, root)
    paths = plot_retention_trajectory(
        root, steps, str(cfg.source_run), cmap=str(cfg.cmap), show_ci=bool(cfg.show_ci)
    )
    print("Saved " + ", ".join(str(path) for path in paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("discover", "evaluate", "plot"), default="evaluate")
    parser.add_argument(
        "--source-run", default="jublett-university-of-oxford/iccl-analysis/0bdfkn9e"
    )
    parser.add_argument(
        "--roots", nargs="+", default=["outputs"], help="Recursively search these directories"
    )
    parser.add_argument(
        "--checkpoint-paths", nargs="*", default=[], help="Pin exact files for ambiguous steps"
    )
    parser.add_argument("--start", type=int, default=100000)
    parser.add_argument("--stop", type=int, default=2100000)
    parser.add_argument("--every", type=int, default=100000)
    parser.add_argument(
        "--eval-dir", default="data/eval_sets", help="Existing authoritative frozen bundle"
    )
    parser.add_argument("--out-dir", default="outputs/retention-trajectory/results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=("auto", "fla", "reference"), default="auto")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--cmap", default="viridis_r", help="Bright early and dark late curves")
    parser.add_argument(
        "--show-ci", action="store_true", help="Draw saved 95%% bootstrap intervals"
    )
    cfg = parser.parse_args()
    if cfg.batch_size <= 0 or cfg.bootstrap_replicates < 2:
        parser.error("batch size must be positive and bootstrap replicates must be at least two")
    run_retention_trajectory(cfg)


if __name__ == "__main__":
    main()
