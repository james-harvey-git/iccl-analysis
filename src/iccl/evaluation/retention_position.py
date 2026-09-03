"""Whole-world estimands for the paired retention-position diagnostic."""

from typing import Any

import numpy as np


def _matrix(
    values: np.ndarray, groups: np.ndarray, coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align one value per coordinate into complete paired-group rows."""
    group_values, coordinate_values = np.unique(groups), np.unique(coordinates)
    order = np.lexsort((coordinates, groups))
    if len(order) != len(group_values) * len(coordinate_values) or not np.array_equal(
        coordinates[order], np.tile(coordinate_values, len(group_values))
    ):
        raise ValueError("every position group must contain each coordinate exactly once")
    if not np.array_equal(groups[order], np.repeat(group_values, len(coordinate_values))):
        raise ValueError("position-group rows are incomplete")
    return values[order].reshape(len(group_values), len(coordinate_values)), coordinate_values


def _curve(
    report: Any,
    descriptor: dict[str, Any],
    mse: np.ndarray,
    nmse: np.ndarray,
    groups: np.ndarray,
    positions: np.ndarray,
    *,
    seed: int,
    component: str,
    curve_type: str,
    mode: str,
    statuses: np.ndarray | None = None,
) -> None:
    mse_matrix, coordinates = _matrix(mse.mean(axis=1), groups, positions)
    nmse_matrix, _ = _matrix(nmse.mean(axis=1), groups, positions)
    extras = {
        int(position): {
            "original_task_position": int(position),
            "intervening_tasks": int(descriptor["T"]) - 1 - int(position),
            "rehearsal_mode": None if curve_type == "retention_position" else mode,
            "support_status": (
                None if statuses is None else str(np.unique(statuses[positions == position]).item())
            ),
        }
        for position in coordinates
    }
    report.curve(
        descriptor,
        "savings" if curve_type == "retention_position" else mode,
        curve_type,
        mse_matrix,
        nmse_matrix,
        seed=seed,
        x_name="original_task_position",
        x_values=coordinates,
        component=component,
        row_extras=extras,
    )


def evaluate_position_diagnostic(
    report: Any,
    descriptor: dict[str, Any],
    values: dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray]],
    raw_errors: dict[str, np.ndarray],
    *,
    seed: int,
) -> None:
    """Add paired position/rehearsal rows and arrays to an evaluation report."""
    suite, repeat_mse, repeat_nmse = values["repeat"]
    novel = values["novel"]
    components = {"total": (novel[1] - repeat_mse, novel[2] - repeat_nmse)}
    if "shared" in values:
        shared = values["shared"]
        components |= {
            "episodic": (shared[1] - repeat_mse, shared[2] - repeat_nmse),
            "module": (novel[1] - shared[1], novel[2] - shared[2]),
        }

    family = str(descriptor["diagnostic_family"])
    positions = np.asarray(suite["original_task_position"], dtype=np.int64)
    groups = np.asarray(suite["position_group_id"], dtype=np.int64)
    for key in (
        "pair_id",
        "position_group_id",
        "world_index",
        "sequence_index",
        "logical_task_id",
        "original_task_position",
        "intervening_tasks",
        "target_support",
        "target_modules_seen_before",
        "target_module_pre_exposures",
        "target_module_post_exposures",
        "prior_target_latent_count",
        "prior_target_support_count",
        "rehearsal_mode",
        "support_status",
        "designated_constituent",
    ):
        if key in suite:
            raw_errors[f"retention_position/{family}/{key}"] = suite[key]

    for component_index, (component, (mse, nmse)) in enumerate(components.items()):
        prefix = f"retention_position/{family}/{component}"
        raw_errors[f"{prefix}_mse"], raw_errors[f"{prefix}_nmse"] = mse, nmse
        component_seed = seed + 100 * component_index
        if family == "paired_permutation":
            _curve(
                report,
                descriptor,
                mse,
                nmse,
                groups,
                positions,
                seed=component_seed,
                component=component,
                curve_type="retention_position",
                mode="natural",
            )
            matrix, _ = _matrix(nmse.mean(axis=1), groups, positions)
            interior = matrix[:, 1:-1].mean(axis=1)
            contrasts = {
                "primacy_excess": matrix[:, 0] - interior,
                "recency_excess": matrix[:, -1] - interior,
                "edge_excess": (matrix[:, 0] + matrix[:, -1]) / 2 - interior,
            }
            for contrast_index, (name, contrast) in enumerate(contrasts.items()):
                report.summary(
                    descriptor,
                    name,
                    f"{name}_mean",
                    contrast,
                    seed=component_seed + 20 + contrast_index,
                    component=component,
                    key_suffix=f"/{component}",
                )
            continue

        modes = np.asarray(suite["rehearsal_mode"]).astype(str)
        statuses = np.asarray(suite["support_status"]).astype(str)
        scalar_values = nmse.mean(axis=1)
        for mode_index, mode in enumerate(("none", "one", "both")):
            selected = modes == mode
            _curve(
                report,
                descriptor,
                mse[selected],
                nmse[selected],
                groups[selected],
                positions[selected],
                seed=component_seed + 10 * mode_index,
                component=component,
                curve_type="retention_rehearsal",
                mode=mode,
                statuses=statuses[selected],
            )
        for position in np.unique(positions):
            selected = positions == position
            matrix, labels = _matrix(scalar_values[selected], groups[selected], modes[selected])
            baseline = matrix[:, np.flatnonzero(labels == "none")[0]]
            for mode_index, mode in enumerate(("one", "both")):
                difference = matrix[:, np.flatnonzero(labels == mode)[0]] - baseline
                coordinate = int(position)
                report.summary(
                    descriptor,
                    "rehearsal_effect",
                    "rehearsal_effect_mean",
                    difference,
                    seed=component_seed + 40 + coordinate * 2 + mode_index,
                    component=component,
                    extra={
                        "original_task_position": coordinate,
                        "intervening_tasks": int(descriptor["T"]) - 1 - coordinate,
                        "rehearsal_mode": mode,
                        "support_status": (
                            "includes_disconnected_ood" if coordinate == 0 else "connected_id"
                        ),
                    },
                    key_suffix=f"/{component}/p{coordinate}/{mode}",
                )
