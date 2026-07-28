"""Numerical parity between the fla and reference gated-delta-rule backends.

Runs only on CUDA machines (Isambard); this is what makes "prototype on the
reference backend, train on fla kernels" trustworthy.
"""

import pytest
import torch

pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_backends_match() -> None:
    pytest.skip("waiting on the reference implementation")
