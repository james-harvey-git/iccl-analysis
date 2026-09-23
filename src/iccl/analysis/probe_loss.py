"""Differentiable parameter loss after detached exact discrete alignment."""

import math
from dataclasses import dataclass

import torch

from iccl.analysis.probe_matching import Assignment
from iccl.analysis.probe_targets import MODULE_FEATURES, MODULE_SHAPE, OUTPUT_FEATURES


def unpack_parameters(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """View a batch in the versioned module-then-readout layout."""
    if values.ndim != 2 or values.shape[1] != OUTPUT_FEATURES:
        raise ValueError("expected [batch,4608] teacher parameters")
    return values[:, :MODULE_FEATURES].reshape(-1, *MODULE_SHAPE), values[
        :, MODULE_FEATURES:
    ].reshape(-1, 16, 16)


def aligned_targets(targets: torch.Tensor, assignment: Assignment) -> torch.Tensor:
    """Gather targets in prediction coordinates with one shared hidden permutation."""
    batch = len(targets)
    if assignment.masks.shape != (batch,) or assignment.permutations.shape != (batch, 16):
        raise ValueError("assignment shape does not match targets")
    modules, readout = unpack_parameters(targets)
    masks = torch.as_tensor(assignment.masks, device=targets.device, dtype=torch.long)
    p = torch.as_tensor(assignment.permutations, device=targets.device, dtype=torch.long)
    bits = (masks[:, None] >> torch.arange(8, device=targets.device)) & 1
    pairs = bits[:, :, None] ^ torch.arange(2, device=targets.device)
    modules = modules.gather(2, pairs[:, :, :, None, None].expand(batch, 8, 2, 17, 16))
    modules = modules.gather(4, p[:, None, None, None, :].expand(batch, 8, 2, 17, 16))
    readout = readout.gather(1, p[:, :, None].expand(batch, 16, 16))
    return torch.cat((modules.reshape(batch, MODULE_FEATURES), readout.reshape(batch, 256)), dim=1)


@dataclass(frozen=True)
class ParameterErrors:
    joint: torch.Tensor
    weights: torch.Tensor
    biases: torch.Tensor
    readout: torch.Tensor
    by_task: torch.Tensor


def parameter_errors(
    predictions: torch.Tensor, aligned: torch.Tensor, readout_weight: float = 1.0
) -> ParameterErrors:
    """Per-episode FP32 errors; ``joint.mean()`` retains the decoder gradient graph."""
    if not math.isfinite(readout_weight) or readout_weight <= 0:
        raise ValueError("readout_weight must be positive and finite")
    if predictions.shape != aligned.shape:
        raise ValueError("prediction and aligned target shapes differ")
    modules, readout = unpack_parameters((predictions.float() - aligned.float()).square())
    weights = modules[:, :, :, :16].mean(dim=(1, 2, 3, 4))
    biases = modules[:, :, :, 16].mean(dim=(1, 2, 3))
    output = readout.mean(dim=(1, 2))
    joint = (
        modules.sum(dim=(1, 2, 3, 4)) + readout_weight * readout.sum(dim=(1, 2))
    ) / OUTPUT_FEATURES
    return ParameterErrors(joint, weights, biases, output, modules.mean(dim=(2, 3, 4)))
