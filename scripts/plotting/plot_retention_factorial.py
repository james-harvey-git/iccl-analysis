"""Render factorial retention PNG/PDF figures from saved evaluation results.

Score checkpoints with scripts/eval.py and evaluation.suites=retention_position.
This plotting command performs no inference and needs no checkpoint files or GPU:

    uv run python scripts/plotting/plot_retention_factorial.py \
        --results outputs/factorial/evaluation-results --plot-steps 100000 2100000

--results accepts step directories or roots containing step_* directories. Omit
--plot-steps to plot every supplied checkpoint. All steps must share the frozen
factorial experiment and source training run. Existing full-evaluation results
can be used directly. No data generation, downloads or W&B connection occurs.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from iccl.evaluation.results import read_rows, validate_cached_results
from iccl.reporting.factorial import FactorialGrid, factorial_paper_figures


def load_trajectory(
    roots: list[Path],
    steps: list[int] | None = None,
) -> tuple[FactorialGrid, list[dict[str, Any]]]:
    """Read complete compatible numerical artifacts, selecting exact checkpoints."""
    paths = sorted(
        {
            p.resolve()
            for root in roots
            for p in ([root] if (root / "manifest.json").is_file() else root.glob("step_*"))
        }
    )
    if steps is not None and (not steps or steps != sorted(set(steps))):
        raise ValueError("--plot-steps must be unique and increasing")
    selected: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        if steps is not None:
            meta = json.loads((path / "manifest.json").read_text())
            if meta["step"] not in steps:
                continue
        manifest = validate_cached_results(path)
        if manifest is None:
            raise ValueError(f"No complete evaluation results in {path}")
        step = int(manifest["step"])
        if step in selected:
            raise ValueError(f"Multiple result directories for checkpoint {step}")
        selected[step] = (path, manifest)
    if not selected or (steps is not None and set(selected) != set(steps)):
        raise ValueError("Missing requested factorial evaluation checkpoints")
    signature = None
    rows, provenance = [], []
    for step, (path, manifest) in sorted(selected.items()):
        identity = manifest["evaluation_identity"]
        current = {
            k: identity[k]
            for k in (
                "source_run",
                "architecture",
                "implementation_sha256",
                "bootstrap_seed",
                "bootstrap_replicates",
                "backend",
                "device_type",
                "precision",
                "batch_size",
                "torch",
                "numpy",
            )
        }
        current["suite_files"] = {
            k: v
            for k, v in identity["suite_files"].items()
            if k.startswith("retention_factorial__")
        }
        if not current["suite_files"] or (signature is not None and signature != current):
            raise ValueError("Checkpoint results do not share the same factorial data/run/settings")
        if len(selected) > 1 and current["source_run"] is None:
            raise ValueError("Trajectory requires source-run provenance")
        signature = current
        saved = [
            r for r in read_rows(path / "summary.csv") if r["capability"] == "retention_factorial"
        ]
        if not saved or any(r["step"] != step for r in saved):
            raise ValueError("Factorial row steps disagree with the manifest")
        grid = FactorialGrid.from_rows(saved)
        source_suites = [
            m for m in manifest["suites"].values() if m["capability"] == "retention_factorial"
        ]
        if not source_suites or any(
            sorted(m["preceding_tasks"]) != grid.preceding
            or sorted(m["intervening_tasks"]) != grid.delays
            for m in source_suites
        ):
            raise ValueError("Saved surface does not cover the frozen factorial grid")
        rows.extend(saved)
        provenance.append(
            dict(
                results=str(path),
                step=step,
                identity=identity,
                result_files=manifest["result_files"],
            )
        )
    return FactorialGrid.from_rows(rows), provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--plot-steps", type=int, nargs="+")
    parser.add_argument("--fixed-preceding", type=int, default=0)
    parser.add_argument("--fixed-delay", type=int, help="default: largest configured delay")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/retention-factorial-plots"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid, provenance = load_trajectory(args.results, args.plot_steps)
    figures = factorial_paper_figures(grid, preceding=args.fixed_preceding, delay=args.fixed_delay)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, figure in figures.items():
        for extension in ("png", "pdf"):
            figure.savefig(args.out_dir / f"{name}.{extension}", dpi=180)
        plt.close(figure)
    (args.out_dir / "provenance.json").write_text(
        json.dumps(
            dict(
                sources=provenance,
                arguments=vars(args),
                preceding=grid.preceding,
                delays=grid.delays,
                description=grid.title,
                aggregation="all final demonstrations, then equal worlds",
                intervals="pointwise 95% whole-world bootstrap",
            ),
            indent=2,
            default=str,
        )
    )
    print(f"Wrote {len(figures)} PNG/PDF figure pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
