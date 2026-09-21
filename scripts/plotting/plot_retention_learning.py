"""Paper retention learning curves and a paired mean-savings decomposition.

    uv run python scripts/plotting/plot_retention_learning.py \
        --results outputs/report-results/6x64x0g0/step_2100000

Reads curves.csv, summary.csv and manifest.json from one evaluation step.
The reference defaults to M=T=8, D=32. Original denotes the first visit to
the task that is subsequently repeated, not the ordinary ICL suite.
Saved estimates give each delay equal weight. Confidence intervals for savings
come from paired episode differences, never from subtracting marginal bounds.
No inference, W&B access, configuration changes or raw arrays are required.
Separate PDF/PNG panels, a PDF-only Overleaf ZIP, source provenance and a
per-configuration audit CSV are written under --out-dir.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from iccl.evaluation.metrics import METRIC_VERSION
from iccl.evaluation.results import read_rows

Row = dict[str, Any]
LEARNING = {
    "original": ("Original", "#666666", "--"),
    "novel": ("Novel", "#0072B2", "-"),
    "shared": ("Shared", "#D55E00", "--"),
    "repeat": ("Repeat", "#6A3D9A", "-"),
}
COMPONENTS = {
    "total": ("savings", "Total", "#333333"),
    "module": ("module_savings", "Module", "#0072B2"),
    "episodic": ("episodic_savings", "Episodic", "#6A3D9A"),
}


def _check_values(rows: list[Row], field: str, *, nonnegative: bool = False) -> None:
    values = np.array([[r[k] for k in (field, "ci_low", "ci_high")] for r in rows])
    if (
        not np.isfinite(values).all()
        or (values[:, 1] > values[:, 2]).any()
        or (nonnegative and (values < 0).any())
    ):
        raise ValueError("Invalid saved estimates or confidence intervals")


def select_retention(
    curve_rows: list[Row], summary_rows: list[Row], *, modules: int, tasks: int, demos: int
) -> tuple[dict[str, list[Row]], dict[str, list[Row]], dict[str, Row]]:
    """Select a complete cell and verify the paired decomposition at every index."""
    if min(modules, tasks, demos) <= 0:
        raise ValueError("Module, task and demonstration counts must be positive")
    cell = (modules, tasks, demos)
    curves = [
        r for r in curve_rows if r["capability"] == "retention" and (r["M"], r["T"], r["D"]) == cell
    ]
    summaries = [
        r
        for r in summary_rows
        if r["capability"] == "retention" and (r["M"], r["T"], r["D"]) == cell
    ]

    def curve(kind: str, condition: str, component: str = "") -> list[Row]:
        selected = sorted(
            [r for r in curves if r["curve_type"] == kind and r["condition"] == condition],
            key=lambda r: r["x_value"],
        )
        if [r["x_value"] for r in selected] != list(range(demos)):
            raise ValueError(f"Expected one complete {condition} {kind} curve for {cell}")
        if any(
            r["x_name"] != "demo_index" or (r["retention_component"] or "") != component
            for r in selected
        ):
            raise ValueError(f"Unexpected axis or component for {condition}")
        _check_values(selected, "nmse", nonnegative=kind == "retention_learning")
        return selected

    learning = {condition: curve("retention_learning", condition) for condition in LEARNING}
    savings = {}
    summary = {}
    for component, (condition, _, _) in COMPONENTS.items():
        savings[component] = curve("retention_savings", condition, component)
        selected = [
            r
            for r in summaries
            if r["retention_component"] == component
            and r["metric"] == f"{condition}_mean"
            and r["condition"] == condition
        ]
        if len(selected) != 1:
            raise ValueError(f"Expected one {component} summary for {cell}")
        _check_values(selected, "value")
        summary[component] = selected[0]

    records = [r for group in (*learning.values(), *savings.values()) for r in group]
    records += list(summary.values())
    metadata = {
        tuple(r[k] for k in ("step", "checkpoint_reference", "suite", "cell_id", "n_sequences"))
        for r in records
    }
    if len(metadata) != 1 or next(iter(metadata))[-1] <= 0:
        raise ValueError("Retention rows must describe one checkpoint, suite and episode count")
    means = {key: np.array([r["nmse"] for r in rows]) for key, rows in learning.items()}
    for component, (lhs, rhs) in {
        "total": ("novel", "repeat"),
        "module": ("novel", "shared"),
        "episodic": ("shared", "repeat"),
    }.items():
        values = np.array([r["nmse"] for r in savings[component]])
        if not np.allclose(values, means[lhs] - means[rhs], rtol=1e-8, atol=1e-10):
            raise ValueError(f"Saved {component} curve disagrees with learning-curve differences")
        if not np.isclose(values.mean(), summary[component]["value"], rtol=1e-8, atol=1e-10):
            raise ValueError(f"Saved {component} summary disagrees with its demonstration mean")
    return learning, savings, summary


def audit_configurations(curves: list[Row], summaries: list[Row]) -> list[Row]:
    """Verify the direction and relative size of savings in each physical cell."""
    cells = {(r["M"], r["T"], r["D"]) for r in curves if r["capability"] == "retention"}
    summary_cells = {(r["M"], r["T"], r["D"]) for r in summaries if r["capability"] == "retention"}
    if not cells or cells != summary_cells:
        raise ValueError("Retention curves and summaries must cover the same nonempty cell set")
    audit = []
    for modules, tasks, demos in sorted(cells):
        learning, savings, summary = select_retention(
            curves, summaries, modules=modules, tasks=tasks, demos=demos
        )
        base = summary["total"]
        row = {k: base[k] for k in ("cell_id", "M", "T", "S", "D", "n_sequences")}
        for component, estimate in summary.items():
            for key in ("value", "ci_low", "ci_high"):
                row[f"{component}_{key}"] = estimate[key]
        total = summary["total"]["value"]
        row["module_fraction_of_total"] = summary["module"]["value"] / total if total > 0 else None
        row["module_exceeds_episodic"] = summary["module"]["value"] > summary["episodic"]["value"]
        for condition, rows in learning.items():
            row[f"{condition}_mean_nmse"] = float(np.mean([r["nmse"] for r in rows]))
        for index, estimate in enumerate(savings["total"][:2], start=1):
            for key in ("nmse", "ci_low", "ci_high"):
                row[f"total_demo_{index}_{key}"] = estimate[key]
        audit.append(row)
    return audit


def render_panels(
    learning: dict[str, list[Row]], summary: dict[str, Row], root: Path, *, show_ci: bool
) -> list[Path]:
    """Export separate, consistently sized panels with LaTeX owning their lettering."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    figure_root = root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
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
        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        fig.subplots_adjust(left=0.17, right=0.98, bottom=0.22, top=0.97)
        ymax = 0.0
        for condition, (label, colour, style) in LEARNING.items():
            rows = learning[condition]
            x = np.array([r["x_value"] + 1 for r in rows])
            mean, low, high = (
                np.array([r[k] for r in rows]) for k in ("nmse", "ci_low", "ci_high")
            )
            ymax = max(ymax, float((high if show_ci else mean).max()))
            if show_ci:
                ax.fill_between(x, low, high, color=colour, alpha=0.12, linewidth=0)
            ax.plot(x, mean, color=colour, linestyle=style, linewidth=1.4, label=label)
        demos = len(learning["original"])
        ax.set(
            xlabel="Demonstration index",
            ylabel="Mean nMSE",
            ylim=(0, ymax * 1.08),
            xticks=sorted({1, *range(8, demos + 1, 8), demos}),
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(axis="y", color="0.91", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(length=3, width=0.7, labelsize=9)
        ax.legend(loc="upper right", frameon=False, fontsize=8.5, handlelength=2.2)
        for suffix in ("pdf", "png"):
            path = figure_root / f"retention-learning.{suffix}"
            fig.savefig(path, dpi=300)
            paths.append(path)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(3.3, 2.5))
        fig.subplots_adjust(left=0.27, right=0.97, bottom=0.22, top=0.97)
        xmin, xmax = 0.0, 0.0
        for y, (component, (_, _label, colour)) in zip((2, 1, 0), COMPONENTS.items(), strict=True):
            row = summary[component]
            mean, low, high = (row[k] for k in ("value", "ci_low", "ci_high"))
            xmin, xmax = min(xmin, low if show_ci else mean), max(xmax, high if show_ci else mean)
            if show_ci:
                ax.hlines(y, low, high, color=colour, linewidth=1.5)
                ax.vlines([low, high], y - 0.045, y + 0.045, color=colour, linewidth=1)
            ax.plot(mean, y, "o", color=colour, markersize=4)
            ax.annotate(
                f"{mean:.3f}",
                (mean, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=colour,
                fontsize=9,
            )
        span = max(xmax - xmin, 0.01)
        ax.set(
            xlabel="Mean savings (nMSE)",
            xlim=(xmin - 0.05 * span, xmax + 0.15 * span),
            ylim=(-0.6, 2.65),
            yticks=[2, 1, 0],
            yticklabels=[item[1] for item in COMPONENTS.values()],
        )
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.axvline(0, color="0.7", linewidth=0.7, linestyle="--", zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=3, width=0.7, labelsize=9)
        for suffix in ("pdf", "png"):
            path = figure_root / f"retention-decomposition.{suffix}"
            fig.savefig(path, dpi=300)
            paths.append(path)
        plt.close(fig)
    with ZipFile(root / "retention-learning-panels.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            if path.suffix == ".pdf":
                archive.write(path, f"figures/{path.name}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/report-figures/retention"))
    parser.add_argument("--modules", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--demos", type=int, default=32)
    args = parser.parse_args()
    source = args.results.expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["metric_version"] != METRIC_VERSION:
        raise ValueError("Evaluation metric version does not match this codebase")
    curves, summaries = (read_rows(source / name) for name in ("curves.csv", "summary.csv"))
    for row in (*curves, *summaries):
        if row["capability"] == "retention" and (
            row["step"] != manifest["step"]
            or row["checkpoint_reference"] != manifest["checkpoint_reference"]
        ):
            raise ValueError("Retention checkpoint provenance disagrees with its manifest")
    learning, savings, summary = select_retention(
        curves, summaries, modules=args.modules, tasks=args.tasks, demos=args.demos
    )
    audit = audit_configurations(curves, summaries)
    destination = args.out_dir.expanduser().resolve()
    replicates = int(manifest["data_evaluation_config"]["bootstrap_replicates"])
    show_ci = replicates >= 2 and summary["total"]["n_sequences"] > 1
    paths = render_panels(learning, summary, destination, show_ci=show_ci)
    with (destination / "retention-configuration-audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    provenance = {
        "source_directory": str(source),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("curves.csv", "summary.csv", "manifest.json")
        },
        "source_run": manifest.get("source_run"),
        "step": manifest["step"],
        "checkpoint_reference": manifest["checkpoint_reference"],
        "metric_version": manifest["metric_version"],
        "bootstrap_replicates": replicates,
        "aggregation": "equal-delay mean, bootstrap episodes within delay groups",
        "confidence_intervals": (
            "saved marginal intervals for curves, saved paired intervals for savings"
        ),
        "displayed_index_base": 1,
        "learning": learning,
        "savings": savings,
        "summary": summary,
        "configuration_audit": audit,
    }
    (destination / "retention-learning.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for component, row in summary.items():
        print(f"{component}: {row['value']:.6f} [{row['ci_low']:.6f}, {row['ci_high']:.6f}]")
        positive = sum(r[f"{component}_ci_low"] > 0 for r in audit)
        print(f"  Positive lower CI in {positive}/{len(audit)} configurations")
    fractions = [
        r["module_fraction_of_total"] for r in audit if r["module_fraction_of_total"] is not None
    ]
    if fractions:
        print(
            f"Module fraction of positive total savings: "
            f"{min(fractions):.2%} to {max(fractions):.2%}"
        )
    print("Saved " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
