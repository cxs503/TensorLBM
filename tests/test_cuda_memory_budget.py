from __future__ import annotations

import pytest
import torch

from tensorlbm.cuda_memory_budget import (
    assess_cuda_memory_budget,
    assess_cuda_runtime_reserve,
    plan_hierarchy_device_memory,
    require_cuda_memory_budget,
    require_cuda_runtime_reserve,
)


def test_hierarchy_memory_is_aggregated_per_owning_device() -> None:
    plan = plan_hierarchy_device_memory(
        (100, 20, 40, 200),
        ("cuda:0", "cuda:0", "cuda:1", "cuda:2"),
        bytes_per_cell=1024.0,
    )

    assert [entry.device for entry in plan] == ["cuda:0", "cuda:1", "cuda:2"]
    assert [entry.level_indices for entry in plan] == [(0, 1), (2,), (3,)]
    assert [entry.allocated_cells for entry in plan] == [120, 40, 200]
    assert plan[-1].estimated_peak_gib == pytest.approx(200 / 2**20)


def test_hierarchy_memory_rejects_an_incomplete_assignment() -> None:
    with pytest.raises(ValueError, match="one entry per level"):
        plan_hierarchy_device_memory((100, 20), ("cpu",), bytes_per_cell=1000)


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
    assert require_cuda_runtime_reserve(torch.device("cpu")) is None


def test_post_allocation_reserve_is_measured_not_estimated() -> None:
    gib = 2**30
    admitted = assess_cuda_runtime_reserve(
        free_bytes=2 * gib,
        total_bytes=24 * gib,
        required_reserve_gib=1.0,
    )
    rejected = assess_cuda_runtime_reserve(
        free_bytes=gib // 12,
        total_bytes=24 * gib,
        required_reserve_gib=1.0,
    )

    assert admitted.admitted is True
    assert admitted.free_gib == pytest.approx(2.0)
    assert rejected.admitted is False
    assert rejected.free_gib == pytest.approx(1.0 / 12.0)


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
