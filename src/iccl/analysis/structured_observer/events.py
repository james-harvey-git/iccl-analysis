"""Causal observation events derived from a frozen token stream."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from iccl.data.sequences import TOKEN_BOUNDARY, TOKEN_PAD, TOKEN_X, TOKEN_Y


@dataclass(frozen=True)
class BoundaryEvent:
    """An explicit task boundary; positions are one-based for reporting."""

    task_position: int


@dataclass(frozen=True)
class InputEvent:
    """A revealed input that requires a prediction before its output arrives."""

    task_position: int
    demo_index: int
    value: np.ndarray


@dataclass(frozen=True)
class OutputEvent:
    """The output revealed immediately after its corresponding prediction."""

    task_position: int
    demo_index: int
    value: np.ndarray


ObservationEvent = BoundaryEvent | InputEvent | OutputEvent


def iter_observation_events(
    tokens: np.ndarray,
    token_types: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
) -> Iterator[ObservationEvent]:
    """Yield only the information available to a causal sequence observer.

    Ground-truth latents, sampled module worlds, normalization statistics,
    targets at future x-token positions, and future demonstration counts never
    enter this interface.
    """
    if tokens.ndim != 2 or token_types.ndim != 1 or len(tokens) != len(token_types):
        raise ValueError("tokens and token_types must describe one aligned sequence")
    task_position = 0
    demo_index = 0
    waiting_for_output = False
    for position, token_type in enumerate(token_types):
        kind = int(token_type)
        if kind == TOKEN_PAD:
            if waiting_for_output:
                raise ValueError("sequence padding begins before a pending output")
            break
        if kind == TOKEN_BOUNDARY:
            if waiting_for_output:
                raise ValueError("task boundary occurs before a pending output")
            task_position += 1
            demo_index = 0
            yield BoundaryEvent(task_position=task_position)
        elif kind == TOKEN_X:
            if task_position == 0:
                raise ValueError("x-token appears before the first task boundary")
            if waiting_for_output:
                raise ValueError("consecutive x-tokens violate demonstration timing")
            waiting_for_output = True
            yield InputEvent(
                task_position=task_position,
                demo_index=demo_index,
                value=tokens[position, :input_dim].astype(np.float64, copy=True),
            )
        elif kind == TOKEN_Y:
            if not waiting_for_output:
                raise ValueError("y-token has no preceding x-token")
            yield OutputEvent(
                task_position=task_position,
                demo_index=demo_index,
                value=tokens[position, :output_dim].astype(np.float64, copy=True),
            )
            demo_index += 1
            waiting_for_output = False
        else:
            raise ValueError(f"unknown token type {kind} at position {position}")
    if waiting_for_output:
        raise ValueError("sequence ends before a pending output")
