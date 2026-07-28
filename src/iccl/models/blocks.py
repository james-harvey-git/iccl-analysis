"""Gated-delta-net block: q/k/v/beta/gate projections, norms, and MLP.

All modules here are backend-agnostic PyTorch; the sequence-mixing itself goes
through ``iccl.models.ops.gated_delta_rule``.
"""
