"""Torch dataset wrapper with deterministic, order-independent seeding.

Sequences are generated on-the-fly: sequence ``i`` under base seed ``s`` is
produced from an RNG keyed on ``(s, i)`` on CPU, so the stream is independent of
dataloader workers, batch size, and device. Together with the golden-stream
regression tests (``tests/test_data_golden.py``), this makes the training data
fully determined by config + seed + code version.
"""
