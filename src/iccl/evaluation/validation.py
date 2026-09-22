"""Semantic validation of paired retention archives at the loading boundary."""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from iccl.data.curriculum import check_compositional, check_connected, task_categories
from iccl.data.retention_factorial import FACTORIAL_PROTOCOL, factorial_axis
from iccl.data.retention_position import REHEARSAL_PROTOCOL, RETENTION_PROTOCOL


def _require(valid: bool | np.bool_, message: str) -> None:
    if not valid:
        raise ValueError(
            f"invalid retention bundle: {message}; regenerate with scripts/make_eval_sets.py"
        )


def _equal(left: ArrayLike, right: ArrayLike, message: str) -> None:
    _require(np.array_equal(left, right), message)


def _row(suite: dict[str, Any], index: int, condition: str, rehearsal: bool) -> None:
    latents = suite["latents"][index]
    history, target = latents[:-1], latents[-1]
    tasks, modules = history.shape
    position = int(suite["original_task_position"][index])
    _require(0 <= position < tasks, "original position out of range")
    _equal(suite["intervening_tasks"][index], tasks - position - 1, "delay/position mismatch")
    active = history != 0
    support, remaining = np.flatnonzero(target), np.flatnonzero(target == 0)
    _require(len(support) == 2 and np.all(active.sum(axis=1) == 2), "requires two-hot tasks")
    _equal(suite["target_support"][index], support, "target support mismatch")
    original = history[position]
    if condition == "repeat":
        _equal(original, target, "repeat original latent differs from final task")
    elif condition == "shared":
        _equal(original != 0, target != 0, "shared support mismatch")
        _require(not np.array_equal(original, target), "shared equals repeat")
    else:
        _require(not original[support].any(), "unexposed original has target exposure")
        _equal(
            np.sort(original[original != 0]), np.sort(target[support]), "unexposed weight multiset"
        )
    pre, post = active[:position, support].sum(axis=0), active[position + 1 :, support].sum(axis=0)
    _require(not pre.any(), "target pre-exposure")
    mode = str(suite["rehearsal_mode"][index])
    slots = suite["rehearsal_positions"][index]
    expected_post = np.zeros(2, dtype=np.int64)
    if rehearsal:
        _require(mode in {"none", "one", "both"}, "unknown rehearsal mode")
        _require(
            len(slots) == 2
            and len(np.unique(slots)) == 2
            and np.all((slots > position) & (slots < tasks)),
            "rehearsal slots must be distinct and after encounter",
        )
        designated = int(suite["designated_constituent"][index])
        _require(designated in support, "designated constituent is not a target")
        if mode in {"one", "both"}:
            expected_post[support == designated] = 1
        if mode == "both":
            expected_post[:] = 1
        _equal(
            active[slots][:, support].sum(axis=1),
            [int(mode != "none"), int(mode == "both")],
            "incorrect rehearsal slot exposures",
        )
        if mode != "none":
            _require(bool(active[slots[0], designated]), "incorrect first rehearsed constituent")
    else:
        _require(mode == "natural" and len(slots) == 0, "rehearsal leaked into standard retention")
    _equal(post, expected_post, "incorrect target post-exposure")
    core_indices = np.array(
        [i for i in range(tasks) if i != position and i not in slots], dtype=int
    )
    _equal(suite["background_task_indices"][index], core_indices, "incorrect background indices")
    _require(
        not active[core_indices][:, support].any(), "target exposure outside encounter/rehearsal"
    )
    core = active[core_indices][:, remaining]
    if suite["__meta__"]["capability"] != "retention_factorial":
        _require(
            check_compositional(core, modules - 2) and check_connected(core),
            "background must cover and connect all non-target modules independently",
        )
    exposed = int(condition != "unexposed")
    expected = {
        "prior_target_latent_count": int(condition == "repeat"),
        "prior_target_support_count": exposed,
        "target_module_pre_exposures": pre,
        "target_modules_seen_before": pre > 0,
        "target_module_post_exposures": post,
        "constituent_task_exposures": exposed + post,
        "constituent_demo_exposures": (
            active[:, support] * suite["demo_counts"][index, :-1, None]
        ).sum(axis=0),
        "background_num_modules": modules - 2,
        "background_covered": check_compositional(core, modules - 2),
        "background_connected": check_connected(core),
        "history_covered": check_compositional(active, modules),
        "history_connected": check_connected(active),
        "presentation_category": task_categories(latents),
        "generation_category": task_categories(latents),
        "support_status": ("connected" if check_connected(active) else "disconnected")
        + ("_covered" if check_compositional(active, modules) else "_partial"),
        "condition": condition,
        "exposure_scope": "original_encounter" if rehearsal else "history",
    }
    _equal(
        np.all(history == target, axis=1).sum(),
        expected["prior_target_latent_count"],
        "target latent count",
    )
    _equal(np.all(active == (target != 0), axis=1).sum(), exposed, "target pair count")
    expected["num_unique_supports_seen"] = np.array(
        [len({tuple(np.flatnonzero(v)) for v in latents[:i]}) for i in range(len(latents))]
    )
    expected["num_modules_covered"] = np.array(
        [np.any(latents[:i] != 0, axis=0).sum() for i in range(len(latents))]
    )
    for key, value in expected.items():
        _equal(suite[key][index], value, f"stale {key} metadata")


def validate_retention_group(conditions: dict[str, dict[str, Any]]) -> None:
    """Validate fixed final tasks, single-slot interventions and paired populations."""
    repeat = conditions["repeat"]
    meta = repeat["__meta__"]
    rehearsal = meta["capability"] == "rehearsal"
    factorial = meta["capability"] == "retention_factorial"
    protocol = (
        FACTORIAL_PROTOCOL if factorial else REHEARSAL_PROTOCOL if rehearsal else RETENTION_PROTOCOL
    )
    _require(meta.get("protocol") == protocol, "obsolete protocol")
    count = len(repeat["tokens"])
    tasks = int(meta["num_tasks"])
    groups = repeat["position_group_id"]
    worlds = np.unique(groups)
    _equal(len(worlds), meta["num_worlds"], "world count mismatch")
    _equal(
        count,
        len(worlds) * (1 if factorial else 6 if rehearsal else tasks),
        "incomplete world-position population",
    )
    if factorial:
        _equal(groups, np.arange(meta["num_worlds"]), "factorial world order/identity mismatch")
        _equal(repeat["intervening_tasks"], np.full(count, meta["delay"]), "cell delay mismatch")
        _require(set(conditions) == set(meta["conditions"]), "missing factorial condition")
    _require(len(np.unique(repeat["pair_id"])) == count, "duplicate pair identifiers")
    for condition, suite in conditions.items():
        _equal(suite["__meta__"].get("protocol"), protocol, "mixed protocols")
        _equal(suite["protocol"], np.full(count, protocol), "stale row protocol")
        if "monitor_indices" in meta:
            _equal(
                suite["__meta__"].get("monitor_indices"),
                meta["monitor_indices"],
                "conditions select different monitor rows",
            )
        for key in (
            "pair_id",
            "position_group_id",
            "original_task_position",
            "intervening_tasks",
            "task_spans",
            "demo_counts",
            "token_type",
            "loss_mask",
            "logical_task_id",
            "rehearsal_mode",
            "rehearsal_positions",
            "background_task_indices",
        ):
            _equal(suite[key], repeat[key], f"conditions disagree on {key}")
        for key in (k for k in repeat if k.startswith("world_")):
            _equal(suite[key], repeat[key], f"conditions disagree on {key}")
        _equal(suite["latents"][:, -1], repeat["latents"][:, -1], "final latent mismatch")
        _equal(suite["base_mse"][:, -1], repeat["base_mse"][:, -1], "final normalization mismatch")
        for index in range(count):
            _row(suite, index, condition, rehearsal)
            position = int(repeat["original_task_position"][index])
            start, end = repeat["task_spans"][index, position]
            mask = np.ones(suite["tokens"].shape[1], dtype=bool)
            mask[start:end] = False
            for key in ("tokens", "targets"):
                _equal(
                    suite[key][index, mask],
                    repeat[key][index, mask],
                    f"{key} changed outside original encounter",
                )
            _equal(
                suite["tokens"][index, start:end:2],
                repeat["tokens"][index, start:end:2],
                "original inputs changed",
            )
            keep = np.arange(tasks + 1) != position
            for key in ("latents", "base_mse"):
                _equal(
                    suite[key][index, keep],
                    repeat[key][index, keep],
                    f"{key} changed outside encounter",
                )
    for world in worlds:
        rows = np.flatnonzero(groups == world)
        positions = repeat["original_task_position"][rows]
        modes = repeat["rehearsal_mode"][rows]
        expected = (
            {(p, r) for p in (0, (tasks - 1) // 2) for r in ("none", "one", "both")}
            if rehearsal
            else {(p, "natural") for p in range(tasks)}
        )
        if factorial:
            expected = {(meta["original_task_position"], "natural")}
        _require(
            set(zip(positions.tolist(), modes.tolist(), strict=True)) == expected,
            "missing or repeated position/mode",
        )
        for suite in conditions.values():
            reference = int(rows[0])
            for index in rows:
                for key in (k for k in suite if k.startswith("world_")):
                    _equal(
                        suite[key][index],
                        suite[key][reference],
                        "world changes across positions/modes",
                    )
                if not rehearsal:
                    for task, logical in enumerate(suite["logical_task_id"][index, :-1]):
                        reference_task = np.flatnonzero(
                            suite["logical_task_id"][reference] == logical
                        )
                        _require(len(reference_task) == 1, "invalid logical task identity")
                        begin, end = suite["task_spans"][index, task]
                        ref_begin, ref_end = suite["task_spans"][reference, int(reference_task[0])]
                        for key in ("tokens", "targets"):
                            _equal(
                                suite[key][index, begin:end],
                                suite[key][reference, ref_begin:ref_end],
                                "logical block changed across delays",
                            )
                final_start = int(suite["task_spans"][index, -1, 0])
                for key in ("tokens", "targets"):
                    _equal(
                        suite[key][index, final_start:],
                        suite[key][reference, final_start:],
                        "final examples changed across positions/modes",
                    )
                _equal(
                    suite["latents"][index, -1],
                    suite["latents"][reference, -1],
                    "target changed within world",
                )
                p, ref_p = (
                    int(suite["original_task_position"][index]),
                    int(suite["original_task_position"][reference]),
                )
                _equal(
                    suite["latents"][index, p],
                    suite["latents"][reference, ref_p],
                    "encounter latent changed within world",
                )
            if rehearsal:
                for position in np.unique(positions):
                    aligned = rows[positions == position]
                    base = int(aligned[np.flatnonzero(modes[positions == position] == "none")[0]])
                    slots = suite["rehearsal_positions"][base]
                    one = int(aligned[np.flatnonzero(modes[positions == position] == "one")[0]])
                    both = int(aligned[np.flatnonzero(modes[positions == position] == "both")[0]])
                    first_start, first_end = suite["task_spans"][base, int(slots[0])]
                    for key in ("tokens", "targets"):
                        _equal(
                            suite[key][one, first_start:first_end],
                            suite[key][both, first_start:first_end],
                            "rehearsal block differs between one and both",
                        )
                    for index in aligned:
                        _equal(suite["rehearsal_positions"][index], slots, "rehearsal slot drift")
                        mask = np.ones(suite["tokens"].shape[1], dtype=bool)
                        for slot in slots:
                            start, end = suite["task_spans"][index, slot]
                            mask[start:end] = False
                            _equal(
                                suite["tokens"][index, start:end:2],
                                suite["tokens"][base, start:end:2],
                                "rehearsal input drift",
                            )
                        for key in ("tokens", "targets"):
                            _equal(
                                suite[key][index, mask],
                                suite[key][base, mask],
                                "mode changed an unaffected block",
                            )


def validate_factorial_grid(suites: list[dict[str, Any]]) -> None:
    """Check complete cells and immutable logical blocks across the factorial grid."""
    if not suites:
        return
    first = suites[0]
    meta = first["__meta__"]
    ps, ds = factorial_axis(meta["preceding_tasks"]), factorial_axis(meta["intervening_tasks"])
    expected = {(p, d, c) for p in ps for d in ds for c in meta["conditions"]}
    seen: set[tuple[int, int, str]] = set()
    blocks: dict[tuple[str, int, int], tuple[np.ndarray, ...]] = {}
    for suite in suites:
        info = suite["__meta__"]
        p, d, condition = info["original_task_position"], info["delay"], info["condition"]
        cell = (p, d, condition)
        _require(cell in expected and cell not in seen, "missing or duplicate factorial cell")
        seen.add(cell)
        for key in (
            "preceding_tasks",
            "intervening_tasks",
            "conditions",
            "num_worlds",
            "num_modules",
            "demos_per_task",
            "seed",
        ):
            _equal(info[key], meta[key], f"factorial grid disagrees on {key}")
        _equal(info["num_tasks"], p + d + 1, "incorrect factorial history length")
        _require(info["sampler"] == "independent", "incorrect factorial sampler")
        for key in (k for k in first if k.startswith("world_")):
            _equal(suite[key], first[key], "teacher/world changed across grid")
        logical = np.array([*(2 + 2 * np.arange(p)), 0, *(3 + 2 * np.arange(d)), 1])
        _equal(
            suite["logical_task_id"],
            np.tile(logical, (info["num_worlds"], 1)),
            "invalid bank block IDs",
        )
        for world in range(info["num_worlds"]):
            for task, identity in enumerate(logical):
                start, end = suite["task_spans"][world, task]
                block = tuple(suite[k][world, start:end] for k in ("tokens", "targets")) + (
                    suite["latents"][world, task],
                    suite["base_mse"][world, task],
                )
                key = (condition, world, int(identity))
                reference = blocks.setdefault(key, block)
                for actual, frozen in zip(block, reference, strict=True):
                    _equal(actual, frozen, "logical block changed across factorial grid")
    _require(seen == expected, "incomplete factorial grid")
