"""Affine multi-output ridge solves with direct and bounded-memory conjugate-gradient paths."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol

import numpy as np
import torch


class Matrix(Protocol):
    shape: tuple[int, int]

    def rows(self, indices: np.ndarray) -> np.ndarray: ...
    def batches(self, size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


class ArrayMatrix:
    """In-memory adapter for small numerical checks and observation features."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = np.asarray(values)
        self.shape = self.values.shape

    def rows(self, indices: np.ndarray) -> np.ndarray:
        return self.values[indices]

    def batches(self, size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, len(self.values), size):
            ids = np.arange(start, min(start + size, len(self.values)))
            yield ids, self.values[ids]


def training_scaling(matrix: Matrix, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge centered moments in float64, including genuinely constant features."""
    mean, moment = np.zeros(matrix.shape[1]), np.zeros(matrix.shape[1])
    count = 0
    for _, raw in matrix.batches(batch_size):
        values = raw.astype(np.float64)
        batch_mean = values.mean(axis=0)
        delta = batch_mean - mean
        total = count + len(values)
        moment += ((values - batch_mean) ** 2).sum(axis=0) + delta**2 * count * len(values) / total
        mean += delta * len(values) / total
        count = total
    variance = moment / count
    scale = np.where(variance > 0, np.sqrt(variance), 1.0)
    return mean, scale, variance


@torch.no_grad()
def predict_ridge(
    bundle: dict[str, Any],
    matrix: Matrix,
    device: torch.device,
    batch_size: int,
    indices: np.ndarray | None = None,
) -> np.ndarray:
    indices = np.arange(matrix.shape[0]) if indices is None else np.asarray(indices)
    if bundle.get("kind") == "training_mean":
        return np.broadcast_to(
            np.asarray(bundle["target_mean"]), (len(indices), len(bundle["target_mean"]))
        ).copy()
    weights = bundle["weights"].to(device)
    mean, scale = bundle["mean"].to(device), bundle["scale"].to(device)
    intercept = bundle["intercept"].to(device)
    values = []
    for start in range(0, len(indices), batch_size):
        raw = matrix.rows(indices[start : start + batch_size])
        x = torch.as_tensor(raw, dtype=weights.dtype, device=device)
        values.append((((x - mean) / scale) @ weights + intercept).cpu().numpy())
    return np.concatenate(values)


@torch.no_grad()
def fit_ridge(
    matrix: Matrix,
    targets: np.ndarray,
    regularizer: float,
    *,
    device: torch.device,
    options: dict[str, Any],
    log_every: int = 10,
    callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    resume: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Minimize ||XW + b - Z||_F²/n + lambda ||W||_F² with an unpenalized intercept.

    The direct dual/primal route never constructs a feature covariance when
    features outnumber trials. Large designs use streamed matrix-vector products.
    Callback state contains a complete resumable CG iterate and its scaling.
    """
    n, width = matrix.shape
    if n < 2 or targets.shape[0] != n or not np.isfinite(targets).all():
        raise ValueError("ridge needs at least two finite, aligned training examples")
    if not np.isfinite(regularizer) or regularizer <= 0:
        raise ValueError("ridge regularization must be finite and positive")
    dtype_name = options["dtype"]
    if dtype_name == "auto":
        dtype_name = "float32" if device.type == "mps" else "float64"
    if dtype_name == "float64" and device.type == "mps":
        raise ValueError("MPS does not support float64; choose float32 or a CPU/CUDA solve")
    dtype = getattr(torch, dtype_name)
    batch_size = options["row_batch_size"]
    if resume is None:
        mean, scale, variance = training_scaling(matrix, batch_size)
        target_mean = targets.astype(np.float64).mean(axis=0)
    else:
        mean = resume["mean"].numpy()
        scale = resume["scale"].numpy()
        variance = resume["variance"].numpy()
        target_mean = resume["intercept"].numpy()
    average = torch.as_tensor(mean, device=device, dtype=dtype)
    scaling = torch.as_tensor(scale, device=device, dtype=dtype)
    intercept = torch.as_tensor(target_mean, device=device, dtype=dtype)
    y = torch.as_tensor(targets, device=device, dtype=dtype) - intercept
    weight_shape = (width, targets.shape[1])

    def batches():
        for ids, values in matrix.batches(batch_size):
            yield ids, (torch.as_tensor(values, device=device, dtype=dtype) - average) / scaling

    size = torch.tensor([], dtype=dtype).element_size()
    dense_bytes = size * (2 * n * width + 3 * min(n, width) ** 2 + 4 * width * targets.shape[1])
    fits = (
        min(n, width) <= options["direct_max_dimension"]
        and dense_bytes <= options["max_working_mb"] * 2**20
    )
    method = options["method"]
    if method == "auto":
        method = "direct" if fits else "cg"
    if method == "direct" and not fits:
        raise ValueError(
            "direct ridge exceeds its dimension/memory bound; choose cg or raise the explicit bound"
        )
    design = torch.cat([x for _, x in batches()]) if method == "direct" else None
    rhs = torch.zeros(weight_shape, dtype=dtype, device=device)
    if design is not None:
        rhs = design.T @ y / n
    else:
        for ids, x in batches():
            rhs += x.T @ y[ids] / n

    def product(value: torch.Tensor) -> torch.Tensor:
        output = regularizer * value
        if design is not None:
            return output + design.T @ (design @ value) / n
        for _, x in batches():
            output += x.T @ (x @ value) / n
        return output

    norm_rhs = torch.linalg.vector_norm(rhs, dim=0)
    safe_norm = torch.where(norm_rhs > 0, norm_rhs, torch.ones_like(norm_rhs))
    weights = (
        torch.zeros_like(rhs)
        if resume is None
        else resume["weights"].to(device=device, dtype=dtype)
    )
    residual = (
        rhs - product(weights)
        if resume is None
        else resume["residual"].to(device=device, dtype=dtype)
    )
    direction = (
        residual.clone() if resume is None else resume["direction"].to(device=device, dtype=dtype)
    )
    iteration = 0 if resume is None else int(resume["iteration"])
    tolerance = options["relative_tolerance"]
    relative = float((torch.linalg.vector_norm(residual, dim=0) / safe_norm).max())

    def bundle() -> dict[str, Any]:
        return {
            "kind": "ridge",
            "weights": weights.detach(),
            "mean": average,
            "scale": scaling,
            "intercept": intercept,
            "regularizer": regularizer,
            "dtype": dtype_name,
        }

    def report(final: bool = False) -> dict[str, Any]:
        objective = float((y.square().sum() / n - (weights * (rhs + residual)).sum()).cpu())
        diagnostics = {
            "method": method,
            "dtype": dtype_name,
            "iteration": iteration,
            "relative_residual": relative,
            "objective": objective,
            "converged": relative <= tolerance,
            "final": final,
            "dense_design_estimate_bytes": dense_bytes,
        }
        if callback is not None:
            state = bundle() | {
                "iteration": iteration,
                "direction": direction,
                "residual": residual,
                "variance": torch.as_tensor(variance),
                "method": method,
            }
            callback(diagnostics, state)
        return diagnostics

    report()
    if method == "direct":
        assert design is not None
        if width <= n:
            system = design.T @ design
            system.diagonal().add_(n * regularizer)
            weights = torch.linalg.solve(system, design.T @ y)
        else:
            system = design @ design.T
            system.diagonal().add_(n * regularizer)
            weights = design.T @ torch.linalg.solve(system, y)
        iteration = 1
        residual = rhs - product(weights)
        relative = float((torch.linalg.vector_norm(residual, dim=0) / safe_norm).max())
    else:
        while relative > tolerance and iteration < options["max_iterations"]:
            applied = product(direction)
            squares = residual.square().sum(dim=0)
            active = torch.sqrt(squares) / safe_norm > tolerance
            denominator = (direction * applied).sum(dim=0)
            if torch.any(denominator[active] <= 0):
                raise RuntimeError(
                    "CG encountered a nonpositive curvature; check numerical precision"
                )
            alpha = torch.where(active, squares / denominator.clamp_min(torch.finfo(dtype).tiny), 0)
            weights += direction * alpha
            residual -= applied * alpha
            updated = residual.square().sum(dim=0)
            beta = torch.where(active, updated / squares.clamp_min(torch.finfo(dtype).tiny), 0)
            direction = residual + direction * beta
            direction[:, ~active] = 0
            iteration += 1
            relative = float((torch.sqrt(updated) / safe_norm).max())
            if relative <= tolerance:
                residual = rhs - product(weights)
                relative = float((torch.linalg.vector_norm(residual, dim=0) / safe_norm).max())
                if relative > tolerance:
                    direction = residual.clone()
            if (
                iteration % log_every == 0
                and relative > tolerance
                and iteration < options["max_iterations"]
            ):
                report()
    residual = rhs - product(weights)
    relative = float((torch.linalg.vector_norm(residual, dim=0) / safe_norm).max())
    diagnostics = report(final=True)
    result = {
        key: value.cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in bundle().items()
    }
    result["diagnostics"] = diagnostics
    return result, diagnostics
