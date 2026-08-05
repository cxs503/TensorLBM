"""Moment equivalence for the bounded-memory D3Q19 observer."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import macroscopic3d, macroscopic3d_low_memory


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_low_memory_macroscopic_fields_match_vector_form(
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(20260802)
    state = 0.02 + 0.1 * torch.rand((19, 4, 5, 6), dtype=dtype)

    expected = macroscopic3d(state)
    actual = macroscopic3d_low_memory(state)

    tolerance = 2.0e-7 if dtype is torch.float32 else 3.0e-16
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            observed,
            reference,
            rtol=tolerance,
            atol=tolerance,
        )


def test_low_memory_macroscopic_fields_validate_d3q19_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        macroscopic3d_low_memory(torch.zeros((27, 2, 2, 2)))
