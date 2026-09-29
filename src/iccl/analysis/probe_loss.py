"""Differentiable parameter loss after detached exact discrete alignment."""

import math
from dataclasses import dataclass

import torch

from iccl.analysis.probe_matching import Assignment
from iccl.analysis.probe_targets import (
    MODULE_FEATURES,
    MODULE_SHAPE,
    MODULES,
    OUTPUT_FEATURES,
    READOUT_FEATURES,
)


def unpack_parameters(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return module and readout views of a batch in the readout-first layout."""
    if values.ndim != 2 or values.shape[1] != OUTPUT_FEATURES:
        raise ValueError(f"expected [batch,{OUTPUT_FEATURES}] teacher parameters")
    return values[:, READOUT_FEATURES:].reshape(-1, *MODULE_SHAPE), values[
        :, :READOUT_FEATURES
    ].reshape(-1, 16, 16)


def aligned_targets(targets: torch.Tensor, assignment: Assignment) -> torch.Tensor:
    """Gather targets in prediction coordinates with one shared hidden permutation."""
    batch = len(targets)
    if assignment.module_permutations.shape != (batch, MODULES) or (
        assignment.hidden_permutations.shape != (batch, 16)
    ):
        raise ValueError("assignment shape does not match targets")
    modules, readout = unpack_parameters(targets)
    q = torch.as_tensor(assignment.module_permutations, device=targets.device, dtype=torch.long)
    p = torch.as_tensor(assignment.hidden_permutations, device=targets.device, dtype=torch.long)
    modules = modules.gather(1, q[:, :, None, None].expand(batch, MODULES, 17, 16))
    modules = modules.gather(3, p[:, None, None, :].expand(batch, MODULES, 17, 16))
    readout = readout.gather(1, p[:, :, None].expand(batch, 16, 16))
    return torch.cat((readout.reshape(batch, 256), modules.reshape(batch, MODULE_FEATURES)), dim=1)


@dataclass(frozen=True)
class ParameterErrors:
    joint: torch.Tensor
    weights: torch.Tensor
    biases: torch.Tensor
    readout: torch.Tensor
    modules: torch.Tensor
    by_module: torch.Tensor


def parameter_errors(
    predictions: torch.Tensor, aligned: torch.Tensor, readout_weight: float = 1.0
) -> ParameterErrors:
    """Per-episode FP32 errors; ``joint.mean()`` retains the decoder gradient graph."""
    if not math.isfinite(readout_weight) or readout_weight <= 0:
        raise ValueError("readout_weight must be positive and finite")
    if predictions.shape != aligned.shape:
        raise ValueError("prediction and aligned target shapes differ")
    modules, readout = unpack_parameters((predictions.float() - aligned.float()).square())
    weights = modules[:, :, :16].mean(dim=(1, 2, 3))
    biases = modules[:, :, 16].mean(dim=(1, 2))
    output = readout.mean(dim=(1, 2))
    joint = (
        modules.sum(dim=(1, 2, 3)) + readout_weight * readout.sum(dim=(1, 2))
    ) / OUTPUT_FEATURES
    return ParameterErrors(
        joint, weights, biases, output, modules.mean(dim=(1, 2, 3)), modules.mean(dim=(2, 3))
    )
