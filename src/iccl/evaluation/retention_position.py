"""Paired retention and rehearsal estimates over fixed worlds."""

from typing import Any

import numpy as np

Duo = tuple[np.ndarray, np.ndarray]
Values = dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray]]


def _matrix(
    values: np.ndarray, groups: np.ndarray, coordinates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Align complete world-coordinate groups, retaining any trailing axes."""
    group_values, coordinate_values = np.unique(groups), np.unique(coordinates)
    order = np.lexsort((coordinates, groups))
    if (
        len(order) != len(group_values) * len(coordinate_values)
        or not np.array_equal(coordinates[order], np.tile(coordinate_values, len(group_values)))
        or not np.array_equal(groups[order], np.repeat(group_values, len(coordinate_values)))
    ):
        raise ValueError("every position group must contain each coordinate exactly once")
    return values[order].reshape(
        len(group_values), len(coordinate_values), *values.shape[1:]
    ), coordinate_values


def _components(values: Values) -> dict[str, Duo]:
    repeat, unexposed = values["repeat"], values["unexposed"]
    result = {"total": (unexposed[1] - repeat[1], unexposed[2] - repeat[2])}
    if "shared" in values:
        shared = values["shared"]
        result.update(
            episodic=(shared[1] - repeat[1], shared[2] - repeat[2]),
            module=(unexposed[1] - shared[1], unexposed[2] - shared[2]),
        )
    return result


def _save_provenance(raw: dict[str, np.ndarray], prefix: str, values: Values) -> None:
    for condition, (suite, _, _) in values.items():
        for key in (
            "pair_id",
            "position_group_id",
            "world_index",
            "sequence_index",
            "logical_task_id",
            "original_task_position",
            "intervening_tasks",
            "target_support",
            "target_module_pre_exposures",
            "target_module_post_exposures",
            "constituent_task_exposures",
            "prior_target_latent_count",
            "prior_target_support_count",
            "rehearsal_mode",
            "rehearsal_positions",
            "support_status",
            "designated_constituent",
        ):
            if key in suite:
                raw[f"{prefix}/{condition}/{key}"] = suite[key]
    for key in ("position_group_id", "original_task_position", "intervening_tasks"):
        if key in values["repeat"][0]:
            raw[f"{prefix}/{key}"] = values["repeat"][0][key]


def evaluate_retention(
    report: Any,
    descriptor: dict[str, Any],
    values: Values,
    raw_errors: dict[str, np.ndarray],
    *,
    seed: int,
    original_errors: Duo,
) -> None:
    """Use whole-world estimates for full suites and delay strata for monitoring."""
    suite = values["repeat"][0]
    positions, delays = suite["original_task_position"], suite["intervening_tasks"]
    full = descriptor["sample_scope"] == "full"
    groups = suite["position_group_id"]
    prefix = f"retention/{descriptor['cell_id']}"
    _save_provenance(raw_errors, prefix, values)
    strata = None if full else delays

    def collapse(errors: np.ndarray) -> np.ndarray:
        return _matrix(errors, groups, delays)[0].mean(axis=1) if full else errors

    indices = np.arange(len(positions))
    original = tuple(errors[indices, positions, : descriptor["D"]] for errors in original_errors)
    learning = {"original": original} | {condition: (v[1], v[2]) for condition, v in values.items()}
    for condition, (mse, nmse) in learning.items():
        report.curve(
            descriptor,
            condition,
            "retention_learning",
            collapse(mse),
            collapse(nmse),
            seed=seed,
            x_name="demo_index",
            strata=strata,
        )
        if condition != "original":
            report.delay_curve(
                descriptor,
                condition,
                mse,
                nmse,
                delays,
                seed=seed,
                curve_type="retention_error_delay",
            )

    for component, (mse, nmse) in _components(values).items():
        condition = "savings" if component == "total" else f"{component}_savings"
        report.summary(
            descriptor,
            condition,
            f"{condition}_mean",
            collapse(nmse).mean(axis=1),
            seed=seed,
            strata=strata,
            component=component,
        )
        report.curve(
            descriptor,
            condition,
            "retention_savings",
            collapse(mse),
            collapse(nmse),
            seed=seed,
            x_name="demo_index",
            strata=strata,
            component=component,
        )
        report.delay_curve(descriptor, condition, mse, nmse, delays, seed=seed, component=component)
        raw_errors[f"{prefix}/{component}_mse"], raw_errors[f"{prefix}/{component}_nmse"] = (
            mse,
            nmse,
        )
        if full:
            matrix, _ = _matrix(nmse.mean(axis=1), groups, positions)
            if matrix.shape[1] > 2:
                interior = matrix[:, 1:-1].mean(axis=1)
                contrasts = {
                    "primacy_excess": matrix[:, 0] - interior,
                    "recency_excess": matrix[:, -1] - interior,
                    "edge_excess": (matrix[:, 0] + matrix[:, -1]) / 2 - interior,
                }
                for name, contrast in contrasts.items():
                    report.summary(
                        descriptor,
                        name,
                        f"{name}_mean",
                        contrast,
                        seed=seed,
                        component=component,
                        key_suffix=f"/{component}",
                    )


def evaluate_rehearsal(
    report: Any,
    descriptor: dict[str, Any],
    values: Values,
    raw_errors: dict[str, np.ndarray],
    *,
    seed: int,
) -> None:
    """Original-encounter benefits conditional on common subsequent rehearsal."""
    suite = values["repeat"][0]
    positions, groups, modes = (
        suite[key] for key in ("original_task_position", "position_group_id", "rehearsal_mode")
    )
    prefix = f"rehearsal/{descriptor['cell_id']}"
    _save_provenance(raw_errors, prefix, values)
    quantities = {(condition, None): (v[1], v[2]) for condition, v in values.items()} | {
        ("savings", component): errors for component, errors in _components(values).items()
    }
    for (condition, component), (mse, nmse) in quantities.items():
        name = component or condition
        raw_errors[f"{prefix}/{name}_mse"], raw_errors[f"{prefix}/{name}_nmse"] = mse, nmse
        for mode in ("none", "one", "both"):
            selected = modes == mode
            matrices = [
                _matrix(e[selected].mean(axis=1), groups[selected], positions[selected])
                for e in (mse, nmse)
            ]
            coordinates = matrices[0][1]
            scoped = dict(descriptor, rehearsal_mode=mode)
            report.curve(
                scoped,
                condition,
                "retention_rehearsal" if component else "rehearsal_error",
                matrices[0][0],
                matrices[1][0],
                seed=seed,
                x_name="original_task_position",
                x_values=coordinates,
                component=component,
                row_extras={
                    int(p): {
                        "original_task_position": int(p),
                        "intervening_tasks": descriptor["T"] - 1 - int(p),
                    }
                    for p in coordinates
                },
            )
        for position in np.unique(positions):
            selected = positions == position
            matrix, labels = _matrix(nmse[selected].mean(axis=1), groups[selected], modes[selected])
            baseline = matrix[:, np.flatnonzero(labels == "none")[0]]
            for mode in ("one", "both"):
                difference = matrix[:, np.flatnonzero(labels == mode)[0]] - baseline
                report.summary(
                    descriptor,
                    condition,
                    "rehearsal_effect_mean" if component else "rehearsal_error_effect_mean",
                    difference,
                    seed=seed,
                    component=component,
                    extra={
                        "original_task_position": int(position),
                        "intervening_tasks": descriptor["T"] - 1 - int(position),
                        "rehearsal_mode": mode,
                    },
                    key_suffix=f"/{name}/p{position}/{mode}",
                )
