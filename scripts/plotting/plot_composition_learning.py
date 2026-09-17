"""Paper composition learning curves and their paired exposure benefit.

    uv run python scripts/plotting/plot_composition_learning.py \
        --results outputs/report-results/6x64x0g0/step_2100000

Defaults to the reference M=T=8, D=32. Display labels map constituent to
Exposed and matched_prefix to Unexposed. The benefit is Unexposed minus
Exposed, computed by the evaluator within paired episodes. All means and
pointwise confidence intervals are read from saved results, without inference,
smoothing or reconstruction of paired intervals from marginal intervals.
Exports separate PDF/PNG panels, a PDF-only ZIP with figures/ paths, and JSON
provenance under --out-dir. LaTeX owns panel captions and cross-references.
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

Row = dict[str, Any]
CONDITIONS = {
    "constituent": ("Exposed", "#6A3D9A", "-"),
    "matched_prefix": ("Unexposed", "#0072B2", "--"),
    "no_history": ("No history", "#666666", "-."),
}


def select_composition(
    curve_rows: list[Row],
    summary_rows: list[Row],
    manifest: Row,
    *,
    modules: int,
    tasks: int,
    demos: int,
) -> tuple[dict[str, list[Row]], Row]:
    """Verify complete curves, common paired provenance and the benefit identity."""
    if min(modules, tasks, demos) <= 0:
        raise ValueError("Module, task and demonstration counts must be positive")
    if manifest["metric_version"] != METRIC_VERSION:
        raise ValueError("Evaluation metric version does not match this codebase")
    cell = (modules, tasks, demos)
    rows = [
        r
        for r in curve_rows
        if r["capability"] == "composition" and (r["M"], r["T"], r["D"]) == cell
    ]
    curves = {}
    for condition in (*CONDITIONS, "benefit"):
        kind = "composition_benefit" if condition == "benefit" else "composition_learning"
        selected = sorted(
            [r for r in rows if r["condition"] == condition and r["curve_type"] == kind],
            key=lambda r: r["x_value"],
        )
        if [r["x_value"] for r in selected] != list(range(demos)):
            raise ValueError(f"Expected one complete {condition} curve for {cell}")
        if any(r["x_name"] != "demo_index" for r in selected):
            raise ValueError("Composition curves must use demonstration indices")
        values = np.array([[r[k] for k in ("nmse", "ci_low", "ci_high")] for r in selected])
        if (
            not np.isfinite(values).all()
            or (values[:, 1] > values[:, 2]).any()
            or (condition != "benefit" and (values < 0).any())
        ):
            raise ValueError(f"Invalid saved estimates or confidence intervals for {condition}")
        if len({r["suite"] for r in selected}) != 1:
            raise ValueError(f"Mixed suite provenance in {condition}")
        curves[condition] = selected
    selected_summary = [
        r
        for r in summary_rows
        if r["capability"] == "composition"
        and (r["M"], r["T"], r["D"]) == cell
        and r["condition"] == "benefit"
        and r["metric"] == "benefit_mean"
    ]
    if len(selected_summary) != 1:
        raise ValueError(f"Expected one mean paired-benefit summary for {cell}")
    summary = selected_summary[0]
    if (
        not np.isfinite([summary[k] for k in ("value", "ci_low", "ci_high")]).all()
        or summary["ci_low"] > summary["ci_high"]
    ):
        raise ValueError("Invalid saved benefit summary")
    records = [r for group in curves.values() for r in group] + [summary]
    common = {
        tuple(r[k] for k in ("step", "checkpoint_reference", "cell_id", "n_sequences"))
        for r in records
    }
    if len(common) != 1 or next(iter(common))[-1] <= 0:
        raise ValueError("Curves must share a checkpoint, cell and episode count")
    base = curves["constituent"][0]
    if (
        base["step"] != manifest["step"]
        or base["checkpoint_reference"] != manifest["checkpoint_reference"]
    ):
        raise ValueError("Curve checkpoint provenance disagrees with its manifest")
    pair_groups = set()
    for condition in CONDITIONS:
        row = curves[condition][0]
        meta = manifest["suites"][row["suite"]]
        if (
            meta["condition"] != condition
            or meta["cell_id"] != base["cell_id"]
            or meta["capability"] != "composition"
            or meta["num_sequences"] != base["n_sequences"]
        ):
            raise ValueError(f"Suite manifest disagrees with the {condition} curve")
        pair_groups.add(meta["pair_group"])
    if len(pair_groups) != 1 or not next(iter(pair_groups)):
        raise ValueError("Composition controls must belong to the same paired group")
    if curves["benefit"][0]["suite"] != base["suite"] or summary["suite"] != base["suite"]:
        raise ValueError("Paired benefit must refer to the exposed suite")
    means = {key: np.array([r["nmse"] for r in group]) for key, group in curves.items()}
    if not np.allclose(
        means["benefit"], means["matched_prefix"] - means["constituent"], rtol=1e-8, atol=1e-10
    ):
        raise ValueError("Saved benefit disagrees with unexposed minus exposed errors")
    if not np.isclose(means["benefit"].mean(), summary["value"], rtol=1e-8, atol=1e-10):
        raise ValueError("Saved mean benefit disagrees with its demonstration average")
    return curves, summary


def render_panels(curves: dict[str, list[Row]], root: Path, *, show_ci: bool) -> list[Path]:
    """Draw matching half-width paper panels on linear axes with one-based indices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
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
        for stem, conditions, ylabel in (
            ("composition-learning", tuple(CONDITIONS), "Mean nMSE"),
            ("composition-benefit", ("benefit",), "Paired benefit (nMSE)"),
        ):
            fig, ax = plt.subplots(figsize=(3.3, 2.5))
            fig.subplots_adjust(left=0.17, right=0.98, bottom=0.22, top=0.97)
            ymin, ymax = 0.0, 0.0
            for condition in conditions:
                rows = curves[condition]
                x = np.array([r["x_value"] + 1 for r in rows])
                mean, low, high = (
                    np.array([r[k] for r in rows]) for k in ("nmse", "ci_low", "ci_high")
                )
                ymin = min(ymin, float((low if show_ci else mean).min()))
                ymax = max(ymax, float((high if show_ci else mean).max()))
                if condition == "benefit":
                    label = r"$E^{\mathrm{unexposed}}-E^{\mathrm{exposed}}$"
                    colour, style = "#6A3D9A", "-"
                else:
                    label, colour, style = CONDITIONS[condition]
                if show_ci:
                    ax.fill_between(x, low, high, color=colour, alpha=0.14, linewidth=0)
                ax.plot(x, mean, color=colour, linestyle=style, linewidth=1.4, label=label)
            demos = len(curves["benefit"])
            pad = 0.07 * max(ymax - ymin, 0.01)
            ax.set(
                xlabel="Demonstration index",
                ylabel=ylabel,
                ylim=(ymin - pad if stem == "composition-benefit" else 0, ymax + pad),
                xticks=sorted({1, *range(8, demos + 1, 8), demos}),
            )
            if stem == "composition-benefit":
                ax.axhline(0, color="0.55", linewidth=0.7, linestyle="--", zorder=0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            ax.tick_params(length=3, width=0.7, labelsize=9)
            ax.grid(axis="y", color="0.91", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.legend(loc="upper right", frameon=False, fontsize=8.5, handlelength=2.2)
            for suffix in ("pdf", "png"):
                path = destination / f"{stem}.{suffix}"
                fig.savefig(path, dpi=300)
                paths.append(path)
            plt.close(fig)
    with ZipFile(root / "composition-panels.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in paths:
            if path.suffix == ".pdf":
                archive.write(path, f"figures/{path.name}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/report-figures/composition"))
    parser.add_argument("--modules", type=int, default=8)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--demos", type=int, default=32)
    args = parser.parse_args()
    source = args.results.expanduser().resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    curves, summary = select_composition(
        read_rows(source / "curves.csv"),
        read_rows(source / "summary.csv"),
        manifest,
        modules=args.modules,
        tasks=args.tasks,
        demos=args.demos,
    )
    replicates = int(manifest["data_evaluation_config"]["bootstrap_replicates"])
    show_ci = replicates >= 2 and summary["n_sequences"] > 1
    destination = args.out_dir.expanduser().resolve()
    paths = render_panels(curves, destination, show_ci=show_ci)
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
        "displayed_index_base": 1,
        "display_labels": {k: v[0] for k, v in CONDITIONS.items()},
        "benefit_definition": "Unexposed minus Exposed, paired within each episode",
        "curves": curves,
        "benefit_summary": summary,
        "mean_task_nmse": {
            condition: float(np.mean([r["nmse"] for r in curves[condition]]))
            for condition in CONDITIONS
        },
    }
    (destination / "composition-learning.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for condition, mean in provenance["mean_task_nmse"].items():
        print(f"{CONDITIONS[condition][0]} mean nMSE: {mean:.6f}")
    print(
        f"Paired benefit: {summary['value']:.6f} "
        f"[{summary['ci_low']:.6f}, {summary['ci_high']:.6f}]"
    )
    print("Saved " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
