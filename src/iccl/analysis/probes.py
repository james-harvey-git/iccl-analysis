"""Unrestricted linear decoding of module content and a state-independent control."""

import math

import torch
from jaxtyping import Float
from torch import nn

from iccl.analysis.probe_targets import OUTPUT_FEATURES


class LinearModuleDecoder(nn.Module):
    """One affine map from all terminal fast-weight coordinates to teacher parameters."""

    def __init__(self, input_features: int) -> None:
        super().__init__()
        if input_features < 1:
            raise ValueError("input_features must be positive")
        self.input_features = input_features
        self.linear = nn.Linear(input_features, OUTPUT_FEATURES, bias=True)

    def forward(
        self, states: Float[torch.Tensor, "batch features"]
    ) -> Float[torch.Tensor, "batch 4608"]:
        return self.linear(states)


class ConstantModuleDecoder(nn.Module):
    """A trained 4,608-vector shared across episodes, with no state-dependent output."""

    def __init__(self, input_features: int) -> None:
        super().__init__()
        if input_features < 1:
            raise ValueError("input_features must be positive")
        self.input_features = input_features
        self.output = nn.Parameter(torch.empty(OUTPUT_FEATURES))
        nn.init.uniform_(self.output, -1 / math.sqrt(input_features), 1 / math.sqrt(input_features))

    def forward(
        self, states: Float[torch.Tensor, "batch features"]
    ) -> Float[torch.Tensor, "batch 4608"]:
        return self.output.unsqueeze(0).expand(states.shape[0], -1)


def make_decoder(input_features: int, control: str) -> LinearModuleDecoder | ConstantModuleDecoder:
    """Control selection is explicit; the full map is never reduced after an OOM."""
    if control == "constant":
        return ConstantModuleDecoder(input_features)
    if control in ("none", "shuffled_targets"):
        return LinearModuleDecoder(input_features)
    raise ValueError(f"unknown probe control: {control}")
