"""Render detailed Plotly HTML curves from a local or W&B evaluation artifact.

Examples:
    uv run python scripts/plot_evaluation.py outputs/<eval-run>/evaluation-results
    uv run python scripts/plot_evaluation.py \
        wandb://entity/project/evaluation-<run-name>:latest --step 100000
"""

import argparse
from pathlib import Path

from iccl.analysis.evaluation_plots import full_evaluation_figures, write_html_figures
from iccl.evaluation.results import read_rows

WANDB_SCHEME = "wandb://"


def resolve_results(reference: str) -> Path:
    """Resolve a local results directory or download an evaluation artifact."""
    if not reference.startswith(WANDB_SCHEME):
        return Path(reference)
    import wandb

    artifact = wandb.Api().artifact(reference.removeprefix(WANDB_SCHEME), type="evaluation")
    return Path(artifact.download())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="local directory or wandb:// evaluation artifact")
    parser.add_argument("--step", type=int, help="render only this source training step")
    parser.add_argument("--out-dir", type=Path, default=Path("evaluation-plots"))
    args = parser.parse_args()

    root = resolve_results(args.results)
    step_dirs = sorted(root.glob("step_*"))
    if args.step is not None:
        step_dirs = [root / f"step_{args.step:07d}"]
    missing = [path for path in step_dirs if not (path / "summary.csv").exists()]
    if not step_dirs or missing:
        detail = f": {missing[0]}" if missing else ""
        raise SystemExit(f"no evaluation step results found under {root}{detail}")

    total = 0
    for step_dir in step_dirs:
        figures = full_evaluation_figures(
            read_rows(step_dir / "summary.csv"),
            read_rows(step_dir / "curves.csv"),
        )
        total += write_html_figures(figures.items(), args.out_dir / step_dir.name)
    print(f"wrote {total} interactive plots to {args.out_dir}")


if __name__ == "__main__":
    main()
