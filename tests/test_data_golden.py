"""Golden-stream regression tests for the on-the-fly data pipeline.

Checksums the first sequences generated under a fixed seed, so any code change
that alters the random stream fails loudly instead of silently changing the
training distribution.
"""

import pytest

pytest.skip("waiting on the HyperTeacher port", allow_module_level=True)
