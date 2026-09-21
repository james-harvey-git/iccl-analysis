"""Matched retention-position trajectories from local training snapshots.

Only the exact-repeat and unexposed-pair conditions of the fully paired canonical retention
diagnostic are scored. The existing evaluator defines total savings and performs
the whole-world bootstrap. Plots show total savings, unexposed-task error and
repeat-task error against the number of intervening tasks, without smoothing.
The two error plots use the same y-axis limits and all plots share a colour scale.

Run from the repository root, using the same code and frozen evaluation bundle
as the report. No W&B connection or checkpoint upload is needed.

    uv run python scripts/plotting/plot_retention_trajectory.py --mode discover
    uv run python scripts/plotting/plot_retention_trajectory.py
    uv run python scripts/plotting/plot_retention_trajectory.py --mode plot --cmap magma_r
    uv run python scripts/plotting/plot_retention_trajectory.py --mode plot --paper

Discovery runs on CPU. Run evaluation in a cluster GPU allocation. Completed
checkpoints are cached, so rerunning the same command resumes after interruption.
Plot mode needs only the saved results and can run without a GPU or snapshots.
The defaults select 100k through 2100k at 100k intervals from run 0bdfkn9e.
The paper view selects 100k, 200k, 500k, 1.4M and 2.1M to show successive
changes in curve shape. It writes separate panels, a shared legend and a LaTeX
subfigure snippet under paper/, preserving the full trajectory figures.
Use --plot-steps to choose another set of displayed checkpoints.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from iccl.checkpoints import (
    checkpoint_model_config,
    checkpoint_model_digest,
    source_from_checkpoint,
)
from iccl.data.eval_bundle import validate_eval_bundle
from iccl.evaluation.metrics import evaluate_suites, load_eval_suites
from iccl.evaluation.results import (
    evaluation_identity,
    read_rows,
    validate_cached_results,
    write_evaluation_results,
)
from iccl.models.model import model_from_config
from iccl.training.trainer import resolve_autocast_dtype
from iccl.utils import resolve_device, seed_everything

PAPER_STEPS = [100000, 200000, 500000, 1400000, 2100000]


@dataclass(frozen=True)
class Snapshot:
    """A verified local checkpoint and its model-content identity."""

    step: int
    path: Path
    model_sha256: str


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


def delay_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract one complete fully paired canonical retention total-savings curve."""
    selected = sorted(
        (
            row
            for row in rows
            if row["capability"] == "retention"
            and "canonical" in str(row["family_memberships"]).split("|")
            and row.get("sample_scope") == "full"
            and row["curve_type"] == "retention_delay"
            and row["retention_component"] == "total"
        ),
        key=lambda row: row["x_value"],
    )
    if not selected:
        raise ValueError("No fully paired canonical retention total-savings position curve found")
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
    manifest = validate_cached_results(path, expected)
    if manifest is None:
        return None
    identity = manifest["evaluation_identity"]
    rows = delay_rows(read_rows(path / "curves.csv"))
    if {row["step"] for row in rows} != {identity["step"]}:
        raise ValueError(f"Cached curve step disagrees with its manifest: {path}")
    names = {
        name
        for name, meta in manifest["suites"].items()
        if meta.get("capability") == "retention"
        and "canonical" in meta.get("family_memberships", ())
        and meta.get("condition") in {"repeat", "unexposed"}
    }
    identity = dict(
        identity,
        suite_files={
            key: value
            for key, value in identity["suite_files"].items()
            if any(key == name + ext for name in names for ext in (".npz", ".meta.json"))
        },
        selections={name: identity.get("selections", {}).get(name) for name in names},
    )
    return identity, rows


def condition_curves(
    path: Path, rows: list[dict[str, Any]]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read authoritative error-delay estimates, including saved paired-world CIs."""
    curves = {}
    saved = read_rows(path / "curves.csv")
    for condition in ("unexposed", "repeat"):
        selected = sorted(
            (
                row
                for row in saved
                if row["capability"] == "retention"
                and row["cell_id"] == rows[0]["cell_id"]
                and row.get("sample_scope") == "full"
                and row["curve_type"] == "retention_error_delay"
                and row["condition"] == condition
            ),
            key=lambda row: row["x_value"],
        )
        if [row["x_value"] for row in selected] != [row["x_value"] for row in rows] or any(
            row["n_sequences"] != rows[0]["n_sequences"] for row in selected
        ):
            raise ValueError(f"Incomplete cached {condition} error-delay curve in {path}")
        curves[condition] = tuple(
            np.array([row[key] for row in selected]) for key in ("nmse", "ci_low", "ci_high")
        )
    if not np.allclose(
        curves["unexposed"][0] - curves["repeat"][0],
        [row["nmse"] for row in rows],
        rtol=1e-7,
        atol=1e-10,
    ):
        raise ValueError(
            f"Unexposed minus repeat errors disagree with saved total savings in {path}"
        )
    return curves


def plot_retention_trajectory(
    root: Path,
    steps: list[int],
    source_run: str,
    *,
    cmap: str = "viridis_r",
    show_ci: bool = False,
    paper: bool = False,
) -> list[Path]:
    """Draw trajectories or aligned paper panels from the same cached estimates."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    if not steps or steps != sorted(set(steps)) or steps[0] <= 0:
        raise ValueError("Plot steps must be unique positive integers in increasing order")
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
        if not np.array_equal(coordinates, np.arange(len(rows))):
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
                **condition_curves(step_dir, rows),
            }
        )

    assert delays is not None
    order = np.argsort(delays)
    x = delays[order]
    error_max = max(
        float(curves[condition][2 if show_ci else 0].max())
        for curves in series
        for condition in ("unexposed", "repeat")
    )
    error_min = min(
        float(curves[condition][1 if show_ci else 0].min())
        for curves in series
        for condition in ("unexposed", "repeat")
    )
    plots = (
        ("total", "retention-position-trajectory", "Total savings (nMSE)"),
        (
            "unexposed",
            "retention-position-unexposed-trajectory",
            r"$E^{\mathrm{unexposed}}$ (nMSE)",
        ),
        ("repeat", "retention-position-repeat-trajectory", r"$E^{\mathrm{repeat}}$ (nMSE)"),
    )
    paths = []
    destination = root / "paper" if paper else root
    destination.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 8.5 if paper else 10,
            "mathtext.fontset": "dejavuserif" if paper else "dejavusans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.65 if paper else 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        norm = Normalize(
            min(steps) / 1000, max(steps) / 1000 if len(steps) > 1 else steps[0] / 1000 + 1
        )
        colours = plt.get_cmap(cmap)
        colour_values = (
            np.linspace(0.10, 0.95, len(steps)) if paper else norm(np.array(steps) / 1000)
        )
        step_colours = [colours(value) for value in colour_values]
        for condition, stem, ylabel in plots:
            if paper:
                fig, ax = plt.subplots(figsize=(2.2, 2.05))
                fig.subplots_adjust(left=0.255, right=0.975, bottom=0.23, top=0.965)
                ax.tick_params(length=3, width=0.65, labelsize=8)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            else:
                fig, ax = plt.subplots(figsize=(6.3, 3.7), layout="constrained")
            for colour, curves in zip(step_colours, series, strict=True):
                mean, low, high = curves[condition]
                ax.plot(
                    x,
                    mean[order],
                    color=colour,
                    lw=1.2 if paper else 1.5,
                    marker="o",
                    markersize=2.3,
                )
                if show_ci:
                    ax.fill_between(
                        x, low[order], high[order], color=colour, alpha=0.07, linewidth=0
                    )
            if condition == "total" or not paper:
                ax.axhline(0, color="0.55", lw=0.6, ls="--", zorder=0)
            ax.set(
                xlabel="Intervening tasks" if paper else "Number of intervening tasks",
                ylabel=ylabel,
                xticks=x,
            )
            if condition != "total":
                padding = max(0.06 * (error_max - error_min), 1e-6)
                if paper:
                    ax.set_ylim(max(0, error_min - padding), error_max + padding)
                else:
                    ax.set_ylim(0, max(error_max * 1.05, 1e-6))
            ax.grid(axis="y", color="0.91", linewidth=0.5)
            ax.set_axisbelow(True)
            if not paper:
                bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colours), ax=ax, pad=0.035)
                bar.set_label("Training steps (thousands)")
                ticks = sorted(
                    {
                        steps[0] / 1000,
                        *[s / 1000 for s in steps if s % 500000 == 0],
                        steps[-1] / 1000,
                    }
                )
                bar.set_ticks(ticks)
            else:
                stem = f"retention-{'savings' if condition == 'total' else condition}"
            for suffix in ("pdf", "png"):
                path = destination / f"{stem}.{suffix}"
                fig.savefig(path, dpi=250, bbox_inches=None if paper else "tight")
                paths.append(path)
            plt.close(fig)
        if paper:
            fig = plt.figure(figsize=(6.875, 0.45))
            handles = [
                Line2D([], [], color=colour, lw=1.2, marker="o", markersize=2.3)
                for colour in step_colours
            ]
            labels = [f"{step / 1e6:g}M" if step >= 1e6 else f"{step / 1e3:g}k" for step in steps]
            fig.legend(
                handles,
                labels,
                loc="center",
                ncol=len(steps),
                frameon=False,
                title="Meta-training steps",
                title_fontsize=8.5,
                handlelength=1.8,
            )
            for suffix in ("pdf", "png"):
                path = destination / f"retention-legend.{suffix}"
                fig.savefig(path, dpi=250)
                paths.append(path)
            plt.close(fig)
            paths.append(write_paper_latex(destination, labels, show_ci=show_ci))
    return paths


def write_paper_latex(root: Path, labels: list[str], *, show_ci: bool) -> Path:
    """Assemble separately referenceable panels with labels owned by LaTeX."""
    lines = [
        r"% Preamble: \usepackage{graphicx,subcaption} and \usepackage{cleveref}.",
        "% Place the four panel/legend PDFs alongside the main .tex file, or adjust paths.",
        r"\begin{figure}[t]",
        r"    \centering",
    ]
    for index, (name, caption) in enumerate(
        (
            ("unexposed", "Unexposed-task error."),
            ("repeat", "Repeated-task error."),
            ("savings", "Total savings."),
        )
    ):
        lines += [
            r"    \begin{subfigure}[t]{0.32\textwidth}",
            r"        \centering",
            rf"        \includegraphics[width=\linewidth]{{retention-{name}.pdf}}",
            rf"        \caption{{{caption}}}",
            rf"        \label{{fig:retention-training-{name}}}",
            r"    \end{subfigure}" + (r"\hfill" if index < 2 else ""),
        ]
    lines += [
        r"    \par\smallskip",
        r"    \includegraphics[width=\textwidth]{retention-legend.pdf}",
        r"    \caption{Retention across meta-training. Curves show checkpoints at "
        + ", ".join(labels)
        + " steps, selected to illustrate changes in curve shape.",
        "        All checkpoints use the same frozen evaluation episodes.",
        "        Errors are averaged over demonstrations in the final task",
        "        and across paired worlds.",
        r"        Total savings are $E^{\mathrm{unexposed}}-E^{\mathrm{repeat}}$.",
        "        Zero intervening tasks denotes an immediate repetition in the repeat condition.",
    ]
    if show_ci:
        lines.append(
            r"        Shading shows 95\% bootstrap confidence intervals over paired worlds."
        )
    lines += [
        r"    }",
        r"    \label{fig:retention-training}",
        r"\end{figure}",
        "% Examples: " + r"\Cref{fig:retention-training-repeat} and \Cref{fig:retention-training}.",
    ]
    path = root / "retention-figure.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


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
            meta.get("capability") == "retention"
            and "canonical" in meta.get("family_memberships", ())
            and meta.get("condition") in {"repeat", "unexposed"}
        ),
    )
    if len(suites) != 2:
        raise ValueError(
            "Expected one fully paired canonical retention repeat/unexposed suite pair"
        )
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
    first_identity = {key: value for key, value in first.items() if key != "optimizer"}
    del first
    seed_everything(0)
    model = None
    common = evaluation_identity(
        first_identity,
        suites,
        bundle,
        batch_size=int(cfg.batch_size),
        bootstrap_seed=int(cfg.bootstrap_seed),
        bootstrap_replicates=int(cfg.bootstrap_replicates),
        backend=str(cfg.backend),
        device=device,
        dtype=dtype,
    )
    del first_identity
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
        delay_rows(report.curve_rows)
        with TemporaryDirectory(prefix=f".step_{snapshot.step:07d}-", dir=results) as temporary:
            staged = write_evaluation_results(
                report,
                Path(temporary),
                snapshot.step,
                {
                    "checkpoint_reference": str(snapshot.path),
                    "source_run": None if source is None else asdict(source),
                    "evaluation_identity": identity,
                    "eval_bundle": bundle,
                    "suites": {name: suite["__meta__"] for name, suite in suites.items()},
                },
            )
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
    plot_steps = cfg.plot_steps or (PAPER_STEPS if cfg.paper else steps)
    paths = plot_retention_trajectory(
        root,
        plot_steps,
        str(cfg.source_run),
        cmap=str(cfg.cmap),
        show_ci=bool(cfg.show_ci),
        paper=bool(cfg.paper),
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
        "--paper", action="store_true", help="Five selected checkpoints in separate LaTeX panels"
    )
    parser.add_argument(
        "--plot-steps", nargs="+", type=int, help="Explicit plotted steps in increasing order"
    )
    parser.add_argument(
        "--show-ci", action="store_true", help="Draw saved 95%% bootstrap intervals"
    )
    cfg = parser.parse_args()
    if cfg.batch_size <= 0 or cfg.bootstrap_replicates < 2:
        parser.error("batch size must be positive and bootstrap replicates must be at least two")
    run_retention_trajectory(cfg)


if __name__ == "__main__":
    main()
