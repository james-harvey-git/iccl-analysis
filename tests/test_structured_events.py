import numpy as np
import pytest

from iccl.analysis.structured_observer.events import (
    BoundaryEvent,
    InputEvent,
    OutputEvent,
    iter_observation_events,
)
from iccl.data.sequences import TOKEN_BOUNDARY, TOKEN_PAD, TOKEN_X, TOKEN_Y


def test_event_stream_preserves_predict_then_reveal_timing() -> None:
    tokens = np.zeros((7, 3), dtype=np.float32)
    tokens[1, :2] = [0.2, -0.4]
    tokens[2, :1] = [0.7]
    tokens[4, :2] = [0.8, 0.1]
    tokens[5, :1] = [-0.3]
    token_types = np.array(
        [TOKEN_BOUNDARY, TOKEN_X, TOKEN_Y, TOKEN_BOUNDARY, TOKEN_X, TOKEN_Y, TOKEN_PAD]
    )
    events = list(iter_observation_events(tokens, token_types, input_dim=2, output_dim=1))
    assert [type(event) for event in events] == [
        BoundaryEvent,
        InputEvent,
        OutputEvent,
        BoundaryEvent,
        InputEvent,
        OutputEvent,
    ]
    assert events[1].task_position == 1
    assert isinstance(events[1], InputEvent)
    assert isinstance(events[2], OutputEvent)
    assert events[1].demo_index == 0
    assert np.allclose(events[1].value, [0.2, -0.4])
    assert np.allclose(events[2].value, [0.7])


def test_event_stream_rejects_output_before_input() -> None:
    tokens = np.zeros((2, 2), dtype=np.float32)
    token_types = np.array([TOKEN_BOUNDARY, TOKEN_Y])
    with pytest.raises(ValueError, match="no preceding x-token"):
        list(iter_observation_events(tokens, token_types, input_dim=1, output_dim=1))
