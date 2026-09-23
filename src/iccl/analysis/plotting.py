"""Saved-result plots for module reconstruction and independently fitted controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iccl.analysis.probe_results import read_results, require_comparable


def plot_probe_results(
    results: list[Path | str], out_dir: Path | str, *, histories: list[Path | str] | None = None
) -> list[Path]:
    """Draw position and control comparisons using checked numerical artifacts only."""
    if not results:
        raise ValueError("provide at least one probe results directory")
    reports = [read_results(path) for path in results]
    require_comparable([report[0] for report in reports])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{meta['control']} · step {meta['step']} · {Path(path).name}"
        for (meta, _, _), path in zip(reports, results, strict=True)
    ]
    saved = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for label, (_, _, summary) in zip(labels, reports, strict=True):
        for ax, metric in zip(axes, ("module_mse_by_task", "functional_nmse_by_task"), strict=True):
            values = summary["decoder"]["all"]["metrics"][metric]
            positions = np.arange(1, 9)
            ax.plot(positions, values["mean"], marker="o", label=label)
            if values["ci_low"] is not None:
                ax.fill_between(positions, values["ci_low"], values["ci_high"], alpha=0.15)
    for ax, metric, title in zip(
        axes,
        ("module_mse_by_task", "functional_nmse_by_task"),
        ("Module parameters (MSE)", "Oracle-coefficient function (nMSE)"),
        strict=True,
    ):
        values = reports[0][2]["zero"]["all"]["metrics"][metric]
        ax.plot(np.arange(1, 9), values["mean"], linestyle="--", color="0.4", label="zero output")
        ax.set(xlabel="Task position", ylabel=title, xticks=np.arange(1, 9))
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    path = out_dir / "reconstruction-by-position.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    estimates = [report[2]["decoder"]["all"]["metrics"]["joint_mse"] for report in reports]
    estimates.append(reports[0][2]["zero"]["all"]["metrics"]["joint_mse"])
    for i, value in enumerate(estimates):
        ax.scatter(i, value["mean"])
        if value["ci_low"] is not None:
            ax.vlines(i, value["ci_low"], value["ci_high"])
    ax.set(
        xticks=np.arange(len(estimates)),
        xticklabels=[*labels, "zero output"],
        ylabel="Joint aligned parameter MSE",
    )
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    path = out_dir / "control-comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)
    if histories:
        saved.append(plot_probe_history(histories, out_dir))
    return saved


def plot_probe_history(histories: list[Path | str], out_dir: Path | str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for source in histories:
        source = Path(source)
        history = json.loads(source.read_text())
        training, validation = history["training"], history["validation"]
        (line,) = ax.plot(
            [r["step"] for r in training],
            [r["loss"] for r in training],
            alpha=0.5,
            label=f"{source.parent.name}: training",
        )
        ax.plot(
            [r["step"] for r in validation],
            [r["joint_mse"] for r in validation],
            marker="o",
            color=line.get_color(),
            label=f"{source.parent.name}: validation",
        )
    ax.set(xlabel="Optimizer update", ylabel="Joint aligned parameter MSE")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    path = out_dir / "learning-curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
