"""Unit contracts for D3Q19 Gloo transport preconditions."""

import pytest

from tensorlbm.multi_gpu import D3Q19GlooTransport


def test_gloo_transport_requires_real_initialized_process_group() -> None:
    with pytest.raises(RuntimeError, match="initialized"):
        D3Q19GlooTransport()
