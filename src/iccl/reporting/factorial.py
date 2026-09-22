"""Shared factorial plot data and layouts for interactive and paper figures."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots


@dataclass
class FactorialGrid:
    """Paired estimates indexed by checkpoint, preceding count, delay and statistic."""

    steps: list[int]
    preceding: list[int]
    delays: list[int]
    components: dict[str, np.ndarray]  # Each array ends in (mean, low, high).
    title: str

    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]]) -> "FactorialGrid":
        selected = [
            r
            for r in rows
            if r["capability"] == "retention_factorial" and r.get("retention_component")
        ]
        if not selected:
            raise ValueError("No factorial savings results found")
        steps = sorted({int(r.get("step", 0)) for r in selected})
        ps = sorted({int(r["original_task_position"]) for r in selected})
        ds = sorted({int(r["intervening_tasks"]) for r in selected})
        identity = {(r["M"], r["D"], r["n_sequences"], r["protocol"]) for r in selected}
        if len(identity) != 1 or any(r["sample_scope"] != "full" for r in selected):
            raise ValueError("Factorial plots require compatible full-world results")
        names = {r["retention_component"] for r in selected}
        if names not in ({"total"}, {"total", "module", "episodic"}):
            raise ValueError("Incomplete factorial savings decomposition")
        components = {
            c: np.full((len(steps), len(ps), len(ds), 3), np.nan)
            for c in ("total", "module", "episodic")
            if c in names
        }
        for row in selected:
            component = row["retention_component"]
            if row["metric"] != f"factorial_{component}_mean":
                raise ValueError("Unexpected factorial metric")
            i, p, d = (
                steps.index(int(row.get("step", 0))),
                ps.index(row["original_task_position"]),
                ds.index(row["intervening_tasks"]),
            )
            slot = components[component][i, p, d]
            if np.isfinite(slot).any():
                raise ValueError("Duplicate factorial cell")
            slot[:] = [row[k] for k in ("value", "ci_low", "ci_high")]
        for values in components.values():
            if not np.isfinite(values).all() or np.any(values[..., 1] > values[..., 2]):
                raise ValueError("Incomplete or invalid factorial surface")
        if "module" in components and not np.allclose(
            components["total"][..., 0],
            components["module"][..., 0] + components["episodic"][..., 0],
            atol=1e-8,
            rtol=1e-6,
        ):
            raise ValueError("Factorial savings decomposition does not add up")
        m, demos, worlds, _ = next(iter(identity))
        return cls(steps, ps, ds, components, f"M={m} · D={demos} · {worlds} paired worlds")

    def layouts(
        self, *, preceding: int = 0, delay: int | None = None
    ) -> dict[str, list[list[dict[str, Any]]]]:
        """Describe the four layouts once for both rendering backends."""
        delay = max(self.delays) if delay is None else delay
        if preceding not in self.preceding or delay not in self.delays:
            raise ValueError("Requested key-slice coordinate is absent from the factorial grid")

        def heatmap(component: str, index: int, annotate: bool = False) -> dict[str, Any]:
            data = self.components[component][..., 0]
            return dict(
                title=f"{component.title()} · {step_label(self.steps[index])}",
                image=data[index],
                limits=(min(0.0, float(data.min())), max(0.0, float(data.max()))),
                annotate=annotate,
                x=self.delays,
                y=self.preceding,
                xlabel="Intervening tasks",
                ylabel="Tasks before original",
            )

        def sliced(fixed: int, *, fix_preceding: bool) -> dict[str, Any]:
            values = self.components["total"]
            axis = self.preceding if fix_preceding else self.delays
            index = axis.index(fixed)
            return dict(
                title=f"Fixed {'preceding' if fix_preceding else 'intervening'} tasks = {fixed}",
                lines=values[:, index] if fix_preceding else values[:, :, index],
                x=self.delays if fix_preceding else self.preceding,
                xlabel="Intervening tasks" if fix_preceding else "Tasks before original",
                ylabel="Total savings (nMSE)",
            )

        return {
            "total_savings_overview": [[heatmap("total", i, True) for i in range(len(self.steps))]],
            "key_slices": [
                [sliced(preceding, fix_preceding=True), sliced(delay, fix_preceding=False)]
            ],
            "position_and_delay_slices": [
                [sliced(p, fix_preceding=True) for p in self.preceding],
                [sliced(d, fix_preceding=False) for d in self.delays],
            ],
            "savings_surfaces": [
                [heatmap(c, i) for i in range(len(self.steps))] for c in self.components
            ],
        }


def step_label(step: int) -> str:
    if not step:
        return "Checkpoint"
    return f"{step / 1e6:g}M" if step >= 1_000_000 else f"{step / 1000:g}k"


def factorial_figures(
    grid: FactorialGrid,
    *,
    preceding: int = 0,
    delay: int | None = None,
) -> dict[str, go.Figure]:
    """Render all four layouts as Plotly figures without numerical recomputation."""
    colors = [
        str(c)
        for c in sample_colorscale("Viridis", np.linspace(0.9, 0.05, len(grid.steps)).tolist())
    ]
    result = {}
    for name, panels in grid.layouts(preceding=preceding, delay=delay).items():
        rows, cols = len(panels), max(map(len, panels))
        titles = [
            panels[r][c]["title"] if c < len(panels[r]) else ""
            for r in range(rows)
            for c in range(cols)
        ]
        fig = make_subplots(
            rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=min(0.07, 0.2 / cols)
        )
        for r, row in enumerate(panels, 1):
            for c, panel in enumerate(row, 1):
                if "image" in panel:
                    low, high = panel["limits"]
                    axis = "coloraxis" if r == 1 else f"coloraxis{r}"
                    fig.update_layout(
                        {
                            axis: dict(
                                cmin=low,
                                cmax=high if high > low else low + 1e-9,
                                colorscale="Viridis",
                                colorbar=dict(title="nMSE", len=0.8 / rows, y=1 - (r - 0.5) / rows),
                            )
                        }
                    )
                    fig.add_trace(
                        go.Heatmap(
                            z=panel["image"],
                            x=[str(x) for x in panel["x"]],
                            y=[str(y) for y in panel["y"]],
                            coloraxis=axis,
                            texttemplate="%{z:.2f}" if panel["annotate"] else None,
                        ),
                        row=r,
                        col=c,
                    )
                    fig.update_xaxes(type="category", row=r, col=c)
                    fig.update_yaxes(type="category", row=r, col=c)
                else:
                    for i, (values, color) in enumerate(zip(panel["lines"], colors, strict=True)):
                        x = panel["x"]
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=values[:, 2],
                                mode="lines",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="skip",
                                legendgroup=str(i),
                            ),
                            row=r,
                            col=c,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=values[:, 1],
                                mode="lines",
                                line=dict(width=0),
                                fill="tonexty",
                                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)"),
                                showlegend=False,
                                hoverinfo="skip",
                                legendgroup=str(i),
                            ),
                            row=r,
                            col=c,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=values[:, 0],
                                mode="lines+markers",
                                name=step_label(grid.steps[i]),
                                line=dict(color=color),
                                legendgroup=str(i),
                                showlegend=r == 1 and c == 1,
                            ),
                            row=r,
                            col=c,
                        )
                    fig.update_xaxes(tickvals=panel["x"], row=r, col=c)
                    low = min(0.0, float(grid.components["total"][..., 1].min()))
                    high = max(0.0, float(grid.components["total"][..., 2].max()))
                    margin = max((high - low) * 0.05, 0.005)
                    fig.update_yaxes(range=[low - margin, high + margin], row=r, col=c)
                fig.update_xaxes(title_text=panel["xlabel"], row=r, col=c)
                fig.update_yaxes(title_text=panel["ylabel"], row=r, col=c)
        fig.update_layout(
            template="plotly_white",
            width=max(800, 330 * cols),
            height=380 * rows + 100,
            title=(
                f"{name.replace('_', ' ').title()} · {grid.title}"
                "<br>Pointwise 95% paired-world intervals"
            ),
        )
        result[name] = fig
    return result


def factorial_paper_figures(
    grid: FactorialGrid,
    *,
    preceding: int = 0,
    delay: int | None = None,
) -> dict[str, Any]:
    """Render the same estimates and layouts with Matplotlib for PNG/PDF output."""
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("viridis")(np.linspace(0.9, 0.05, len(grid.steps)))
    figures = {}
    for name, panels in grid.layouts(preceding=preceding, delay=delay).items():
        rows, cols = len(panels), max(map(len, panels))
        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(max(8, 3.4 * cols), 3.5 * rows + 0.7),
            squeeze=False,
            layout="constrained",
        )
        for r, row in enumerate(panels):
            for c, ax in enumerate(axes[r]):
                if c >= len(row):
                    ax.set_visible(False)
                    continue
                panel = row[c]
                if "image" in panel:
                    low, high = panel["limits"]
                    im = ax.imshow(
                        panel["image"],
                        origin="lower",
                        aspect="auto",
                        cmap="viridis",
                        vmin=low,
                        vmax=high if high > low else low + 1e-9,
                    )
                    ax.set_xticks(range(len(panel["x"])), panel["x"])
                    ax.set_yticks(range(len(panel["y"])), panel["y"])
                    if panel["annotate"]:
                        for (p, d), value in np.ndenumerate(panel["image"]):
                            ax.text(
                                d,
                                p,
                                f"{value:.2f}",
                                ha="center",
                                va="center",
                                fontsize=8,
                                color="white" if value < (low + high) / 2 else "black",
                            )
                    if c == len(row) - 1:
                        fig.colorbar(im, ax=list(axes[r]), label="Savings (nMSE)", shrink=0.8)
                else:
                    for i, (values, color) in enumerate(zip(panel["lines"], colors, strict=True)):
                        ax.plot(
                            panel["x"],
                            values[:, 0],
                            "o-",
                            color=color,
                            label=step_label(grid.steps[i]),
                        )
                        ax.fill_between(
                            panel["x"], values[:, 1], values[:, 2], color=color, alpha=0.15
                        )
                    ax.set_xticks(panel["x"])
                    low = min(0.0, float(grid.components["total"][..., 1].min()))
                    high = max(0.0, float(grid.components["total"][..., 2].max()))
                    margin = max((high - low) * 0.05, 0.005)
                    ax.set_ylim(low - margin, high + margin)
                    ax.grid(axis="y", alpha=0.2)
                    if r == 0 and c == 0:
                        ax.legend(title="Training steps", fontsize=8)
                ax.set(title=panel["title"], xlabel=panel["xlabel"], ylabel=panel["ylabel"])
        fig.suptitle(
            f"{name.replace('_', ' ').title()} · {grid.title}\nPointwise 95% paired-world intervals"
        )
        figures[name] = fig
    return figures
