"""Summarize paired performance trials without treating episode tokens as independent."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def series(history: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in history if key in row["metrics"]]
    return (
        np.array([row["step"] for row in selected]),
        np.array([row["metrics"][key] for row in selected]),
    )


def validation_pair(baseline: Path, candidate: Path) -> dict:
    """Bootstrap whole paired episodes and retain the token-pooled MSE denominator."""
    left = np.load(next(baseline.glob("final-evaluation/step_*/raw_errors.npz")))
    right = np.load(next(candidate.glob("final-evaluation/step_*/raw_errors.npz")))
    counts = left["validation.episode_token_count"]
    assert np.array_equal(counts, right["validation.episode_token_count"])
    a, b = left["validation.episode_error_sum"], right["validation.episode_error_sum"]
    rng = np.random.default_rng(0)
    indices = rng.integers(0, len(a), size=(10000, len(a)))
    bootstrap = 100 * (b[indices].sum(axis=1) - a[indices].sum(axis=1)) / a[indices].sum(axis=1)
    relative = float(100 * (b.sum() - a.sum()) / a.sum())
    return {
        "baseline_token_mse": float(a.sum() / counts.sum()),
        "candidate_token_mse": float(b.sum() / counts.sum()),
        "candidate_minus_baseline_percent": relative,
        "paired_episode_percentile_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "episodes": len(a),
        "prediction_tokens": int(counts.sum()),
        "within_one_percent": abs(relative) <= 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--candidate", default="optimized")
    args = parser.parse_args()
    destination = args.root / "analysis"
    destination.mkdir(exist_ok=True)
    labels = [f"fresh-seed{seed}" for seed in range(3)] + ["resumed-seed0"]
    figures = {
        key: plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for key in ("training", "validation", "throughput")
    }
    quantities = {
        "training": ("train/token_mse", "Training token MSE (logged batches; unsmoothed)"),
        "validation": ("validation/token_mse", "Frozen validation token MSE"),
        "throughput": ("train/sequences_per_s", "Training sequences / second (50-step windows)"),
    }
    pairs = []
    for pair_index, label in enumerate(labels):
        paths = {
            variant: args.root / "trajectories" / f"{label}-{variant}"
            for variant in ("baseline", args.candidate)
        }
        pair = {"label": label, "validation": validation_pair(*paths.values()), "runs": {}}
        histories = {variant: records(path / "history.jsonl") for variant, path in paths.items()}
        for variant, path in paths.items():
            provenance = json.loads((path / "trajectory.json").read_text())
            steps, rates = series(histories[variant], "train/sequences_per_s")
            pair["runs"][variant] = {
                "elapsed_seconds": provenance["elapsed_seconds"],
                "median_sequences_per_second_after_first_window": float(np.median(rates[1:])),
                "wandb_run": provenance["wandb_run"],
                "revision": provenance["revision"],
                "job_id": provenance["slurm_job_id"],
                "step": provenance["step"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
            }
            for key, (metric, y_label) in quantities.items():
                figure, axes = figures[key]
                axis = axes.flat[pair_index]
                steps, values = series(histories[variant], metric)
                axis.plot(steps, values, label=variant, alpha=0.9, linewidth=1.4)
                axis.set(title=label, xlabel="Optimizer step", ylabel=y_label)
                axis.ticklabel_format(axis="x", style="plain", useOffset=False)
                axis.grid(alpha=0.2)
                axis.legend()
        baseline = pair["runs"]["baseline"]
        candidate = pair["runs"][args.candidate]
        pair["end_to_end_speedup"] = baseline["elapsed_seconds"] / candidate["elapsed_seconds"]
        pair["median_window_speedup"] = (
            candidate["median_sequences_per_second_after_first_window"]
            / baseline["median_sequences_per_second_after_first_window"]
        )
        baseline_steps, baseline_values = series(histories["baseline"], "validation/token_mse")
        candidate_steps, candidate_values = series(
            histories[args.candidate], "validation/token_mse"
        )
        assert np.array_equal(baseline_steps, candidate_steps)
        pair["largest_intermediate_validation_difference_percent"] = float(
            np.max(np.abs(100 * (candidate_values - baseline_values) / baseline_values))
        )
        pairs.append(pair)

    for key, (figure, _) in figures.items():
        figure.savefig(destination / f"{key}.png", dpi=180)
        plt.close(figure)
    payload = {
        "candidate": args.candidate,
        "bootstrap": "10000 paired whole-episode draws, seed 0, percentile 95% interval",
        "uncertainty_scope": "frozen episodes, excluding training-seed and long-run uncertainty",
        "pairs": pairs,
    }
    (destination / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
