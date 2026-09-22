"""Paper panels for within-task and across-task prediction error.

Run from the repository root with an existing evaluation step directory:

    uv run python scripts/plotting/plot_learning_curves.py \
        --results outputs/report-results/6x64x0g0/step_2100000

The default reference configuration is M=8, T=8, D=32. Means and pointwise
bootstrap intervals are read directly from curves.csv, without model inference,
smoothing or recomputing uncertainty from aggregate values. Separate PDF/PNG
panels, a LaTeX subfigure snippet, numerical provenance and an Overleaf ZIP are
written to --out-dir. Displayed demonstration and task indices start at one.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from iccl.evaluation.metrics import METRIC_VERSION
from iccl.evaluation.results import read_rows

PANELS = (
    ("within_task_learning", "icl-within-task", "Demonstration index", "Within-task learning."),
    ("episode_learning", "iccl-across-tasks", "Task position", "Across-task learning."),
)


def select_curves(
    rows: list[dict[str, Any]], *, modules: int, tasks: int, demos: int
) -> dict[str, list[dict[str, Any]]]:
    """Select one complete physical evaluation cell without mixing checkpoints."""
    if min(modules, tasks, demos) <= 0:
        raise ValueError("Module, task and demonstration counts must be positive")
    curves = {}
    for kind, count, axis in (
        ("within_task_learning", demos, "demo_index"),
        ("episode_learning", tasks, "task_position"),
    ):
        curve = sorted(
            (
                row
                for row in rows
                if row["capability"] == "icl"
                and row["condition"] == "ordinary"
                and row["curve_type"] == kind
                and (row["M"], row["T"], row["D"]) == (modules, tasks, demos)
            ),
            key=lambda row: row["x_value"],
        )
        if [row["x_value"] for row in curve] != list(range(count)):
            raise ValueError(
                f"Expected one complete {kind} curve for M={modules}, T={tasks}, D={demos}"
            )
        if any(row["x_name"] != axis for row in curve):
            raise ValueError(f"Unexpected axis semantics for {kind}")
        values = np.array([[row[k] for k in ("nmse", "ci_low", "ci_high")] for row in curve])
        if (
            not np.isfinite(values).all()
            or (values < 0).any()
            or (values[:, 1] > values[:, 2]).any()
        ):
            raise ValueError(f"Invalid means or confidence intervals for {kind}")
        curves[kind] = curve
    metadata = {
        tuple(row[k] for k in ("step", "checkpoint_reference", "suite", "cell_id", "n_sequences"))
        for curve in curves.values()
        for row in curve
    }
    if len(metadata) != 1 or next(iter(metadata))[-1] <= 0:
        raise ValueError("Curves must describe the same checkpoint, suite and episode count")
    return curves


def write_latex(root: Path, row: dict[str, Any], *, show_ci: bool) -> Path:
    """Let LaTeX own panel lettering, captions and independent cross-references."""
    lines = [
        r"% Preamble: \usepackage{graphicx,subcaption} and \usepackage{cleveref}.",
        r"\begin{figure}[tbp]",
        r"    \centering",
    ]
    for index, (_, stem, _, caption) in enumerate(PANELS):
        lines += [
            r"    \begin{subfigure}[t]{0.48\textwidth}",
            r"        \centering",
            rf"        \includegraphics[width=\linewidth]{{figures/{stem}.pdf}}",
            rf"        \caption{{{caption}}}",
            rf"        \label{{fig:{stem}}}",
            r"    \end{subfigure}" + (r"\hfill" if index == 0 else ""),
        ]
    step_label = f"{row['step']:,}"
    lines += [
        r"    \caption{Prediction improves within tasks and across the episode",
        rf"        for $M={row['M']}$, $T={row['T']}$ and $D={row['D']}$",
        f"        at {step_label} meta-training steps.",
        r"        (\subref{fig:icl-within-task}) Error at each demonstration index,",
        "        averaged over tasks.",
        r"        (\subref{fig:iccl-across-tasks}) Error at each task position,",
        "        averaged over demonstrations.",
        f"        Lines show means over {row['n_sequences']} frozen evaluation episodes.",
    ]
    if show_ci:
        lines += [
            r"        Shading shows 95\% pointwise bootstrap confidence intervals",
            "        from resampling episodes.",
        ]
    lines += [
        r"    }",
        r"    \label{fig:learning-curves}",
        r"\end{figure}",
        r"% Reference panels with \Cref{fig:icl-within-task,fig:iccl-across-tasks}.",
    ]
    path = root / "learning-figure.tex"
    path.write_text("\n".join(lines) + "\n")
    return path


def render_panels(
    curves: dict[str, list[dict[str, Any]]], root: Path, *, show_ci: bool
) -> list[Path]:
    """Render matched panel sizes for side-by-side LaTeX subfigures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    root.mkdir(parents=True, exist_ok=True)
    figure_root = root / "figures"
    figure_root.mkdir(exist_ok=True)
    paths = []
    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        for kind, stem, xlabel, _ in PANELS:
            rows = curves[kind]
            x = np.array([row["x_value"] + 1 for row in rows])
            mean, low, high = (
                np.array([row[k] for row in rows]) for k in ("nmse", "ci_low", "ci_high")
            )
            fig, ax = plt.subplots(figsize=(3.3, 2.5))
            fig.subplots_adjust(left=0.17, right=0.98, bottom=0.22, top=0.97)
            colour = "#28688C"
            if show_ci:
                ax.fill_between(x, low, high, color=colour, alpha=0.18, linewidth=0)
            ax.plot(
                x,
                mean,
                color=colour,
                linewidth=1.5,
                marker="o",
                markersize=2 if kind == "within_task_learning" else 3,
            )
            ticks = x if len(x) <= 8 else sorted({1, *range(8, len(x) + 1, 8), len(x)})
            ax.set(
                xlabel=xlabel,
                ylabel="Mean nMSE",
                xticks=ticks,
                ylim=(0, max(float((high if show_ci else mean).max()) * 1.08, 1e-6)),
            )
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            ax.tick_params(length=3, width=0.7, labelsize=9)
            ax.grid(axis="y", color="0.91", linewidth=0.5)
            ax.set_axisbelow(True)
            for suffix in ("pdf", "png"):
                path = figure_root / f"{stem}.{suffix}"
                fig.savefig(path, dpi=300)
                paths.append(path)
            plt.close(fig)
    paths.append(write_latex(root, curves["within_task_learning"][0], show_ci=show_ci))
    bundle = root / "learning-panels.zip"
    with ZipFile(bundle, "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            if path.suffix == ".pdf":
                archive.write(path, f"figures/{path.name}")
            elif path.suffix == ".tex":
                archive.write(path, path.name)
    return [*paths, bundle]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Evaluation step directory containing curves.csv and manifest.json",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/report-figures/learning"))
    parser.add_argument("--modules", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--demos", type=int, default=32)
    args = parser.parse_args()
    source = args.results.expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["metric_version"] != METRIC_VERSION:
        raise ValueError("Evaluation metric version does not match this codebase")
    curves = select_curves(
        read_rows(source / "curves.csv"), modules=args.modules, tasks=args.tasks, demos=args.demos
    )
    row = curves["within_task_learning"][0]
    if (
        row["step"] != manifest["step"]
        or row["checkpoint_reference"] != manifest["checkpoint_reference"]
    ):
        raise ValueError("Curve checkpoint provenance disagrees with its manifest")
    show_ci = (
        int(manifest["data_evaluation_config"]["bootstrap_replicates"]) >= 2
        and row["n_sequences"] > 1
    )
    destination = args.out_dir.expanduser().resolve()
    paths = render_panels(curves, destination, show_ci=show_ci)
    provenance = {
        "source_directory": str(source),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("curves.csv", "manifest.json")
        },
        "source_run": manifest.get("source_run"),
        "step": row["step"],
        "checkpoint_reference": row["checkpoint_reference"],
        "metric_version": manifest["metric_version"],
        "M": row["M"],
        "T": row["T"],
        "D": row["D"],
        "n_episodes": row["n_sequences"],
        "bootstrap_replicates": manifest["data_evaluation_config"]["bootstrap_replicates"],
        "displayed_index_base": 1,
        "curves": {
            kind: [{key: r[key] for key in ("x_value", "nmse", "ci_low", "ci_high")} for r in rows]
            for kind, rows in curves.items()
        },
    }
    (destination / "learning-curves.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for kind, rows in curves.items():
        print(f"{kind}: {rows[0]['nmse']:.6f} -> {rows[-1]['nmse']:.6f}")
    print("Saved " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
