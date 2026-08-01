from __future__ import annotations

import pytest
import torch

from tensorlbm.cuda_memory_budget import (
    assess_cuda_memory_budget,
    require_cuda_memory_budget,
)


def test_memory_budget_accounts_for_live_free_memory_and_reserve() -> None:
    gib = 2**30
    safe = assess_cuda_memory_budget(
        free_bytes=20 * gib, total_bytes=24 * gib,
        estimated_peak_gib=18.0, reserve_gib=1.0,
    )
    unsafe = assess_cuda_memory_budget(
        free_bytes=19 * gib, total_bytes=24 * gib,
        estimated_peak_gib=18.5, reserve_gib=1.0,
    )

    assert safe.admitted is True
    assert safe.headroom_gib == pytest.approx(1.0)
    assert unsafe.admitted is False
    assert unsafe.headroom_gib == pytest.approx(-0.5)


def test_cpu_run_needs_no_cuda_preflight() -> None:
    assert require_cuda_memory_budget(
        torch.device("cpu"), estimated_peak_gib=1.0,
    ) is None


@pytest.mark.parametrize(
    ("free_bytes", "total_bytes"), ((-1, 10), (11, 10)),
)
def test_invalid_device_inventory_is_rejected(
    free_bytes: int, total_bytes: int,
) -> None:
    with pytest.raises(ValueError, match="invalid CUDA"):
        assess_cuda_memory_budget(
            free_bytes=free_bytes, total_bytes=total_bytes,
            estimated_peak_gib=1.0,
        )
