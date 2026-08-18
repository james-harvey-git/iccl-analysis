"""Small Plotly primitives shared by live reporting and offline analysis."""

from typing import Any

import plotly.graph_objects as go


def grouped_figure(
    rows: list[dict[str, Any]],
    *,
    title: str,
    x_field: str,
    y_field: str,
    x_title: str,
    y_title: str,
    group_fields: tuple[str, ...],
    trace_names: dict[Any, str] | None = None,
    hover_fields: tuple[str, ...] = (),
) -> go.Figure:
    """Plot tabular rows as one trace per declared group."""

    def sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
        value = row.get(x_field)
        return (0, float(value), "") if isinstance(value, int | float) else (1, 0.0, str(value))

    figure = go.Figure()
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    for group, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        ordered = sorted(group_rows, key=sort_key)
        if len(group) == 1 and trace_names is not None:
            label = trace_names.get(group[0], str(group[0]))
        elif group_fields:
            label = ", ".join(
                f"{field}={value}" for field, value in zip(group_fields, group, strict=True)
            )
        else:
            label = title
        values = [float(row[y_field]) for row in ordered]
        customdata = [[row.get(field) for field in hover_fields] for row in ordered]
        hovertemplate = "<br>".join(
            f"{field}=%{{customdata[{index}]}}" for index, field in enumerate(hover_fields)
        )
        if hovertemplate:
            hovertemplate += "<extra></extra>"
        figure.add_trace(
            go.Scatter(
                x=[row.get(x_field) for row in ordered],
                y=values,
                mode="lines+markers",
                name=label,
                error_y={
                    "type": "data",
                    "array": [
                        max(0.0, float(row["ci_high"]) - value)
                        for row, value in zip(ordered, values, strict=True)
                    ],
                    "arrayminus": [
                        max(0.0, value - float(row["ci_low"]))
                        for row, value in zip(ordered, values, strict=True)
                    ],
                    "symmetric": False,
                    "visible": True,
                },
                customdata=customdata or None,
                hovertemplate=hovertemplate or None,
            )
        )
    figure.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_white",
    )
    return figure
