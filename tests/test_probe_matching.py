"""Independent assignment oracles and production native-boundary checks."""

import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from iccl.analysis.probe_matching import AssignmentSolver
from iccl.analysis.probe_targets import MODULE_SHAPE, MODULES, OUTPUT_FEATURES, READOUT_FEATURES


def subset_dp(cost: np.ndarray) -> float:
    n = len(cost)
    dp = np.full(1 << n, np.inf)
    dp[0] = 0
    for mask in range(1, 1 << n):
        row = mask.bit_count() - 1
        dp[mask] = min(dp[mask ^ (1 << j)] + cost[row, j] for j in range(n) if mask & (1 << j))
    return float(dp[-1])


def joint_matrix(cost: np.ndarray, modules: tuple[int, ...] | np.ndarray) -> np.ndarray:
    return cost[0] + sum(cost[1 + MODULES * a + b] for a, b in enumerate(modules))


@pytest.fixture(scope="module")
def solver(tmp_path_factory: pytest.TempPathFactory):
    with AssignmentSolver(2, tmp_path_factory.mktemp("assignment-cache")) as value:
        yield value


@pytest.mark.parametrize("n", [1, 3, 7, 16])
def test_single_against_independent_oracles(solver: AssignmentSolver, n: int) -> None:
    rng = np.random.default_rng(n)
    for matrix in (rng.normal(size=(n, n)), rng.integers(-2, 3, size=(n, n)), np.zeros((n, n))):
        score, permutation = solver.single(matrix)
        np.testing.assert_array_equal(np.sort(permutation), np.arange(n))
        np.testing.assert_allclose(score, matrix[np.arange(n), permutation].sum(), atol=1e-12)
        np.testing.assert_allclose(score, subset_dp(matrix), atol=1e-12)
        if n <= 7:
            brute = min(
                sum(matrix[i, p[i]] for i in range(n)) for p in itertools.permutations(range(n))
            )
            np.testing.assert_allclose(score, brute, atol=1e-12)


@pytest.mark.parametrize("n", [3, 5, 8])
def test_joint_against_all_patterns_with_independent_dp(solver: AssignmentSolver, n: int) -> None:
    rng = np.random.default_rng(40 + n)
    costs = rng.normal(size=(17, 16, 16))
    costs[:, n:, :] = 0
    costs[:, :, n:] = 0
    costs[0, n:, :] = 1e6
    costs[0, :, n:] = 1e6
    costs[0, np.arange(n, 16), np.arange(n, 16)] = 0
    truth = min(
        subset_dp(joint_matrix(costs, q)[:n, :n]) for q in itertools.permutations(range(MODULES))
    )
    if n <= 3:
        brute = min(
            sum(joint_matrix(costs, q)[i, p[i]] for i in range(n))
            for q in itertools.permutations(range(MODULES))
            for p in itertools.permutations(range(n))
        )
        np.testing.assert_allclose(truth, brute, atol=1e-12)
    actual = solver.match_costs(costs[None])
    np.testing.assert_allclose(actual.scores[0], truth, atol=1e-10)
    matrix = joint_matrix(costs, actual.module_permutations[0])
    np.testing.assert_allclose(
        matrix[np.arange(16), actual.hidden_permutations[0]].sum(), truth, atol=1e-10
    )


def numpy_costs(prediction: np.ndarray, target: np.ndarray, weight: float) -> np.ndarray:
    p, y = prediction.astype(np.float64), target.astype(np.float64)
    pb = p[:, READOUT_FEATURES:].reshape(-1, *MODULE_SHAPE)
    yb = y[:, READOUT_FEATURES:].reshape(-1, *MODULE_SHAPE)
    pr = p[:, :READOUT_FEATURES].reshape(-1, 16, 16)
    yr = y[:, :READOUT_FEATURES].reshape(-1, 16, 16)
    costs = np.empty((len(p), 1 + MODULES**2, 16, 16))
    costs[:, 0] = weight * ((pr[:, :, None] - yr[:, None]) ** 2).sum(-1)
    for a in range(MODULES):
        for b in range(MODULES):
            costs[:, 1 + MODULES * a + b] = (
                (pb[:, a, :, :, None] - yb[:, b, :, None, :]) ** 2
            ).sum(1)
    return costs


@pytest.mark.parametrize("weight", [1.0, 17.0])
@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_native_costs_and_warm_enumeration_against_exhaustive(
    solver: AssignmentSolver, weight: float, scale: float
) -> None:
    rng = np.random.default_rng(44)
    target = (rng.normal(size=(6, OUTPUT_FEATURES)) * scale).astype(np.float32)
    prediction = (target + rng.normal(size=target.shape) * scale).astype(np.float32)
    prediction[0] = target[0]
    prediction[1] = 0
    costs = solver.construct_costs(prediction, target, weight)
    np.testing.assert_allclose(
        costs, numpy_costs(prediction, target, weight), rtol=1e-13, atol=1e-30
    )
    actual = solver.match(prediction, target, weight, profile=True)
    truth = solver.match_costs(costs, exhaustive=True)
    np.testing.assert_allclose(actual.scores, truth.scores, rtol=2e-12, atol=scale**2 * 1e-10)
    assert (actual.counters[:, 0] == 24).all()
    assert (actual.counters[:, 1] + actual.counters[:, 2] == 24).all()
    assert (actual.timings >= 0).all()
    for e in range(len(target)):
        matrix = joint_matrix(costs[e], actual.module_permutations[e])
        np.testing.assert_allclose(
            actual.scores[e], matrix[np.arange(16), actual.hidden_permutations[e]].sum(), rtol=1e-12
        )


def test_exact_symmetries_and_threads(solver: AssignmentSolver, tmp_path: Path) -> None:
    rng = np.random.default_rng(73)
    modules = rng.normal(size=MODULE_SHAPE).astype(np.float32)
    readout = rng.normal(size=(16, 16)).astype(np.float32)
    p, q = rng.permutation(16), rng.permutation(MODULES)
    predicted = modules[q][..., p]
    target = np.concatenate((readout.ravel(), modules.ravel()))[None]
    prediction = np.concatenate((readout[p].ravel(), predicted.ravel()))[None]
    result = solver.match(prediction, target)
    assert result.scores[0] == 0
    np.testing.assert_array_equal(result.hidden_permutations[0], p)
    np.testing.assert_array_equal(result.module_permutations[0], q)
    with AssignmentSolver(1, tmp_path) as single:
        other = single.match(prediction, target)
    np.testing.assert_array_equal(result.hidden_permutations, other.hidden_permutations)
    np.testing.assert_array_equal(result.module_permutations, other.module_permutations)
    with ThreadPoolExecutor(3) as executor:
        results = list(executor.map(lambda _: solver.match(prediction, target), range(6)))
    assert all(r.scores[0] == 0 for r in results)


def test_matching_requires_unique_modules_and_one_hidden_basis(solver: AssignmentSolver) -> None:
    rng = np.random.default_rng(10)
    target = rng.normal(size=(1, OUTPUT_FEATURES)).astype(np.float32)
    modules = target[:, READOUT_FEATURES:].reshape(1, *MODULE_SHAPE)
    duplicate = target.copy()
    duplicate[:, READOUT_FEATURES:].reshape(1, *MODULE_SHAPE)[:, 1] = modules[:, 0]
    assert solver.match(duplicate, target).scores[0] > 1
    inconsistent = target.copy()
    inconsistent[:, READOUT_FEATURES:].reshape(1, *MODULE_SHAPE)[:, 0] = modules[:, 0, :, ::-1]
    assert solver.match(inconsistent, target).scores[0] > 1
    # A fixed readout block cannot be reassigned as module content.
    exchanged = target.copy()
    exchanged[:, :256], exchanged[:, 256:512] = target[:, 256:512], target[:, :256]
    assert solver.match(exchanged, target).scores[0] > 1


@pytest.mark.parametrize("regime", ["independent", "small", "noisy", "bias_only", "collapsed"])
def test_prediction_regimes_and_relabelling_invariance(solver: AssignmentSolver, regime: str):
    rng = np.random.default_rng(155)
    target = rng.normal(size=(12, OUTPUT_FEATURES)).astype(np.float32)
    prediction = rng.normal(size=target.shape).astype(np.float32)
    if regime == "small":
        prediction *= 1e-3
    elif regime == "noisy":
        prediction = target + prediction * 0.1
    elif regime == "bias_only":
        prediction[:, READOUT_FEATURES:].reshape(-1, *MODULE_SHAPE)[:, :, :16] = 0
    elif regime == "collapsed":
        modules = prediction[:, READOUT_FEATURES:].reshape(-1, *MODULE_SHAPE)
        modules[:] = modules[:, :1]
    original = solver.match(prediction, target)
    truth = solver.match_costs(solver.construct_costs(prediction, target), exhaustive=True)
    np.testing.assert_allclose(original.scores, truth.scores, rtol=2e-12)
    q, p = rng.permutation(MODULES), rng.permutation(16)
    relabelled = target.copy()
    relabelled[:, READOUT_FEATURES:] = (
        target[:, READOUT_FEATURES:].reshape(-1, *MODULE_SHAPE)[:, q].reshape(len(target), -1)
    )
    transformed = prediction.copy()
    transformed[:, :READOUT_FEATURES] = (
        prediction[:, :READOUT_FEATURES].reshape(-1, 16, 16)[:, p].reshape(len(target), -1)
    )
    transformed[:, READOUT_FEATURES:] = (
        prediction[:, READOUT_FEATURES:]
        .reshape(-1, *MODULE_SHAPE)[:, q][..., p]
        .reshape(len(target), -1)
    )
    np.testing.assert_allclose(solver.match(transformed, relabelled).scores, original.scores)


def test_empty_tied_and_invalid_inputs(solver: AssignmentSolver, tmp_path: Path) -> None:
    empty = np.empty((0, OUTPUT_FEATURES), np.float32)
    assert solver.match(empty, empty).hidden_permutations.shape == (0, 16)
    zero = np.zeros((1, OUTPUT_FEATURES), np.float32)
    result = solver.match(zero, zero)
    assert result.scores[0] == 0
    np.testing.assert_array_equal(result.module_permutations[0], np.arange(MODULES))
    np.testing.assert_array_equal(result.hidden_permutations[0], np.arange(16))
    for invalid in (zero.astype(np.float64), zero[:, :-1], np.full_like(zero, np.nan)):
        with pytest.raises(ValueError):
            solver.match(invalid, zero)
    for weight in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            solver.match(zero, zero, weight)
    with pytest.raises(RuntimeError, match="costs"):
        solver.match_costs(np.full((1, 17, 16, 16), 1e300))
    # A failed native call leaves the persistent pool usable.
    assert solver.match(zero, zero).scores[0] == 0
    closed = AssignmentSolver(1, tmp_path)
    closed.close()
    closed.close()
    with pytest.raises(RuntimeError, match="closed"):
        closed.match(zero, zero)


def test_import_does_not_build_native_code(tmp_path: Path) -> None:
    code = (
        "import iccl.analysis.probe_matching; import pathlib,json; "
        "print(json.dumps(list(map(str,pathlib.Path('.').iterdir()))))"
    )
    output = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert json.loads(output.stdout) == []
