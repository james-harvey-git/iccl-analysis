"""Frozen GDN continuation for independent queries from a shared demonstration prefix."""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F

from iccl.models.blocks import GDNBlock
from iccl.models.model import GDNModel
from iccl.models.ops import Backend, gated_delta_rule


@dataclass
class GDNCache:
    """Every recurrent value: fast-weight matrices and raw short-convolution tails."""

    states: list[torch.Tensor]
    convolutions: list[list[torch.Tensor]]

    def select(self, indices: torch.Tensor) -> "GDNCache":
        """Copy selected rows, including repeated rows for independent query branches."""
        return GDNCache(
            [value.index_select(0, indices) for value in self.states],
            [[value.index_select(0, indices) for value in layer] for layer in self.convolutions],
        )


@torch.inference_mode()
def continue_gdn(
    model: GDNModel,
    tokens: torch.Tensor,
    token_type: torch.Tensor,
    cache: GDNCache | None = None,
    *,
    backend: Backend = "auto",
) -> tuple[torch.Tensor, GDNCache]:
    """Resume before the first supplied token, leaving the input cache untouched.

    Mirrors the model's frozen forward path; convolution histories accompany
    the matrices because the matrices alone are not a sufficient continuation
    state. Segmented/monolithic prediction parity is covered by tests.
    """
    if tokens.shape[1] < 1:
        raise ValueError("continuation requires at least one token")
    if cache is not None and (
        len(cache.states) != len(model.blocks) or len(cache.convolutions) != len(model.blocks)
    ):
        raise ValueError("cache must contain every model layer")
    h = model.embed(tokens, token_type)
    states, convolutions = [], []
    for layer, block in enumerate(cast(list[GDNBlock], list(model.blocks))):
        mixer = block.mixer
        x = block.mixer_norm(h)
        values, tails = [], []
        for coordinate, kind in enumerate(("q", "k", "v")):
            raw = getattr(mixer, f"{kind}_proj")(x)
            if mixer.use_short_conv:
                convolution = getattr(mixer, f"{kind}_conv")
                joined = (
                    raw
                    if cache is None
                    else torch.cat((cache.convolutions[layer][coordinate], raw), dim=1)
                )
                values.append(convolution(joined)[:, -raw.shape[1] :])
                width = convolution.conv.kernel_size[0] - 1
                tails.append(joined[:, -width:].clone() if width else joined[:, :0])
            else:
                values.append(F.silu(raw))
                tails.append(raw[:, :0])
        q, k, v = [value.reshape(*value.shape[:2], mixer.n_heads, -1) for value in values]
        mixed, state = gated_delta_rule(
            q,
            k,
            v,
            mixer.a_proj(x),
            mixer.b_proj(x),
            mixer.A_log,
            mixer.dt_bias,
            allow_neg_eigval=mixer.allow_neg_eigval,
            backend=backend,
            initial_state=None if cache is None else cache.states[layer],
            return_final_state=True,
        )
        assert state is not None
        if mixer.use_gate:
            gate = mixer.g_proj(x).reshape(*x.shape[:2], mixer.n_heads, mixer.head_v_dim)
            mixed = mixer.o_norm(mixed, gate)
        else:
            mixed = mixer.o_norm(mixed)
        h = h + mixer.o_proj(mixed.flatten(-2))
        h = h + block.mlp(block.mlp_norm(h))
        states.append(state)
        convolutions.append(tails)
    return model.head(model.final_norm(h)), GDNCache(states, convolutions)
