"""Gated-delta-net building blocks: projections, short conv, norms, and MLP.

All modules here are backend-agnostic PyTorch; the sequence-mixing itself goes
through ``iccl.models.ops.gated_delta_rule``. Component semantics follow fla's
``GatedDeltaNet`` layer, but the conv and gated norm are reimplemented here (the
``fla`` package cannot be imported off Linux) so they are literally the same
code under both backends — only the recurrence differs.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from iccl.models.ops import Backend, gated_delta_rule


class ShortConvolution(nn.Module):
    """Depthwise causal conv followed by silu, matching fla's ``ShortConvolution``
    (bias-free, left-padded so position t sees tokens t-3..t at kernel size 4)."""

    def __init__(self, dim: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size, groups=dim, padding=kernel_size - 1, bias=False
        )

    def forward(
        self, x: Float[torch.Tensor, "batch seq dim"]
    ) -> Float[torch.Tensor, "batch seq dim"]:
        y = self.conv(x.transpose(1, 2))[..., : x.shape[1]]
        return F.silu(y.transpose(1, 2))


class GatedRMSNorm(nn.Module):
    """``RMSNorm(x) * silu(g)`` — normalize first, then gate, matching fla's
    ``FusedRMSNormGated`` with its default swish activation."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * F.silu(g)


class GDNLayer(nn.Module):
    """Gated-delta-net sequence-mixing layer.

    Bias-free q/k/v projections (optionally short-convolved, else silu), tiny
    ``a_proj``/``b_proj`` heads for the decay gate and write strength, the
    Mamba2 gate parameters ``A_log``/``dt_bias`` (flagged ``_no_weight_decay``),
    the gated-delta-rule recurrence, and a (optionally silu-gated) RMSNorm
    before the output projection. Value heads are ``expand_v`` times wider than
    key heads.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        expand_v: float = 2.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_k_dim = d_model // n_heads
        head_v_dim = int(self.head_k_dim * expand_v)
        if not math.isclose(head_v_dim, self.head_k_dim * expand_v):
            raise ValueError(f"expand_v={expand_v} gives a non-integer value head dim")
        self.head_v_dim = head_v_dim
        value_dim = n_heads * head_v_dim
        self.use_short_conv = use_short_conv
        self.use_gate = use_gate
        self.allow_neg_eigval = allow_neg_eigval

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, value_dim, bias=False)
        self.a_proj = nn.Linear(d_model, n_heads, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)

        # Mamba2 gate parameter init, matching fla: A ~ Uniform(0, 16) stored in
        # log space; dt_bias the softplus-inverse of dt ~ LogUniform[1e-3, 1e-1].
        A = torch.empty(n_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True  # pyright: ignore[reportAttributeAccessIssue]
        self.dt_bias._no_weight_decay = True  # pyright: ignore[reportAttributeAccessIssue]

        if use_short_conv:
            self.q_conv = ShortConvolution(d_model, conv_size)
            self.k_conv = ShortConvolution(d_model, conv_size)
            self.v_conv = ShortConvolution(value_dim, conv_size)
        if use_gate:
            self.g_proj = nn.Linear(d_model, value_dim, bias=False)
            self.o_norm: nn.Module = GatedRMSNorm(head_v_dim, eps=norm_eps)
        else:
            self.o_norm = nn.RMSNorm(head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(value_dim, d_model, bias=False)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq d_model"],
        *,
        backend: Backend = "auto",
        return_states: bool = False,
    ) -> tuple[
        Float[torch.Tensor, "batch seq d_model"],
        Float[torch.Tensor, "batch seq heads value_dim key_dim"] | None,
    ]:
        if self.use_short_conv:
            q, k, v = (
                self.q_conv(self.q_proj(x)),
                self.k_conv(self.k_proj(x)),
                self.v_conv(self.v_proj(x)),
            )
        else:
            q, k, v = F.silu(self.q_proj(x)), F.silu(self.k_proj(x)), F.silu(self.v_proj(x))
        q = rearrange(q, "b t (h d) -> b t h d", d=self.head_k_dim)
        k = rearrange(k, "b t (h d) -> b t h d", d=self.head_k_dim)
        v = rearrange(v, "b t (h d) -> b t h d", d=self.head_v_dim)

        o, states = gated_delta_rule(
            q,
            k,
            v,
            self.a_proj(x),
            self.b_proj(x),
            self.A_log,
            self.dt_bias,
            allow_neg_eigval=self.allow_neg_eigval,
            backend=backend,
            return_states=return_states,
        )
        if self.use_gate:
            g = rearrange(self.g_proj(x), "b t (h d) -> b t h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        return self.o_proj(rearrange(o, "b t h d -> b t (h d)")), states


class SwiGLU(nn.Module):
    """Bias-free gated MLP: ``down(silu(gate(x)) * up(x))``."""

    def __init__(self, d_model: int, d_ffw: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ffw, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffw, bias=False)
        self.down_proj = nn.Linear(d_ffw, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GDNBlock(nn.Module):
    """Pre-norm residual block: RMSNorm -> GDN -> residual, RMSNorm -> SwiGLU -> residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffw: int,
        *,
        expand_v: float = 2.0,
        use_short_conv: bool = True,
        conv_size: int = 4,
        use_gate: bool = True,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.mixer_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.mixer = GDNLayer(
            d_model,
            n_heads,
            expand_v=expand_v,
            use_short_conv=use_short_conv,
            conv_size=conv_size,
            use_gate=use_gate,
            allow_neg_eigval=allow_neg_eigval,
            norm_eps=norm_eps,
        )
        self.mlp_norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.mlp = SwiGLU(d_model, d_ffw)

    def forward(
        self,
        x: Float[torch.Tensor, "batch seq d_model"],
        *,
        backend: Backend = "auto",
        return_states: bool = False,
    ) -> tuple[
        Float[torch.Tensor, "batch seq d_model"],
        Float[torch.Tensor, "batch seq heads value_dim key_dim"] | None,
    ]:
        mixed, states = self.mixer(self.mixer_norm(x), backend=backend, return_states=return_states)
        x = x + mixed
        x = x + self.mlp(self.mlp_norm(x))
        return x, states
