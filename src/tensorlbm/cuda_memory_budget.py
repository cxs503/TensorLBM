"""Fail-fast CUDA memory budgeting for production LBM campaigns."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class CUDAMemoryBudget:
    free_gib: float
    total_gib: float
    estimated_peak_gib: float
    reserve_gib: float
    headroom_gib: float
    admitted: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class CUDARuntimeReserve:
    """Measured free-memory reserve after persistent solver allocation."""

    free_gib: float
    total_gib: float
    required_reserve_gib: float
    admitted: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


@dataclass(frozen=True)
class HierarchyDeviceAllocation:
    """Conservative persistent-memory assignment for hierarchy levels."""

    device: str
    level_indices: tuple[int, ...]
    allocated_cells: int
    estimated_peak_gib: float

    def to_dict(self) -> dict[str, str | int | float | list[int]]:
        return {
            "device": self.device,
            "level_indices": list(self.level_indices),
            "allocated_cells": self.allocated_cells,
            "estimated_peak_gib": self.estimated_peak_gib,
        }


def plan_hierarchy_device_memory(
    allocated_cells_by_level: Sequence[int],
    level_devices: Sequence[torch.device | str],
    *,
    bytes_per_cell: float,
) -> tuple[HierarchyDeviceAllocation, ...]:
    """Aggregate a per-level hierarchy memory model by execution device.

    The estimate remains deliberately conservative: levels sharing a device
    retain the same empirical bytes-per-cell coefficient and are summed.  A
    distributed hierarchy can therefore be admitted per GPU without hiding
    the impossible aggregate allocation behind total cluster memory.
    """
    if len(allocated_cells_by_level) != len(level_devices) or not level_devices:
        raise ValueError("cells and devices must contain one entry per level")
    if bytes_per_cell <= 0.0:
        raise ValueError("bytes_per_cell must be positive")
    levels_by_device: dict[str, list[int]] = {}
    cells_by_device: dict[str, int] = {}
    for level, (cells, raw_device) in enumerate(
        zip(allocated_cells_by_level, level_devices, strict=True),
    ):
        if isinstance(cells, bool) or int(cells) != cells or cells <= 0:
            raise ValueError("every hierarchy level must contain positive cells")
        device = str(torch.device(raw_device))
        levels_by_device.setdefault(device, []).append(level)
        cells_by_device[device] = cells_by_device.get(device, 0) + int(cells)
    return tuple(
        HierarchyDeviceAllocation(
            device=device,
            level_indices=tuple(levels),
            allocated_cells=cells_by_device[device],
            estimated_peak_gib=(
                cells_by_device[device] * bytes_per_cell / 2**30
            ),
        )
        for device, levels in levels_by_device.items()
    )


def assess_cuda_memory_budget(
    *,
    free_bytes: int,
    total_bytes: int,
    estimated_peak_gib: float,
    reserve_gib: float = 1.0,
) -> CUDAMemoryBudget:
    """Assess an empirical peak against currently free device memory."""
    if min(free_bytes, total_bytes) < 0 or free_bytes > total_bytes:
        raise ValueError("invalid CUDA free/total byte counts")
    if estimated_peak_gib <= 0.0 or reserve_gib < 0.0:
        raise ValueError("peak must be positive and reserve non-negative")
    free_gib = free_bytes / 2**30
    total_gib = total_bytes / 2**30
    headroom = free_gib - estimated_peak_gib - reserve_gib
    return CUDAMemoryBudget(
        free_gib=free_gib,
        total_gib=total_gib,
        estimated_peak_gib=estimated_peak_gib,
        reserve_gib=reserve_gib,
        headroom_gib=headroom,
        admitted=headroom >= 0.0,
    )


def require_cuda_memory_budget(
    device: torch.device,
    *,
    estimated_peak_gib: float,
    reserve_gib: float = 1.0,
    label: str = "LBM run",
) -> CUDAMemoryBudget | None:
    """Query live free memory and reject an unsafe CUDA allocation early."""
    if device.type != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    budget = assess_cuda_memory_budget(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        estimated_peak_gib=estimated_peak_gib,
        reserve_gib=reserve_gib,
    )
    if not budget.admitted:
        raise MemoryError(
            f"{label} needs about {estimated_peak_gib:.2f} GiB plus "
            f"{reserve_gib:.2f} GiB reserve, but only {budget.free_gib:.2f} "
            "GiB is currently free",
        )
    return budget


def assess_cuda_runtime_reserve(
    *,
    free_bytes: int,
    total_bytes: int,
    required_reserve_gib: float = 1.0,
) -> CUDARuntimeReserve:
    """Assess measured free memory after persistent fields are allocated."""
    if min(free_bytes, total_bytes) < 0 or free_bytes > total_bytes:
        raise ValueError("invalid CUDA free/total byte counts")
    if required_reserve_gib < 0.0:
        raise ValueError("required CUDA reserve must be non-negative")
    free_gib = free_bytes / 2**30
    return CUDARuntimeReserve(
        free_gib=free_gib,
        total_gib=total_bytes / 2**30,
        required_reserve_gib=required_reserve_gib,
        admitted=free_gib >= required_reserve_gib,
    )


def require_cuda_runtime_reserve(
    device: torch.device,
    *,
    required_reserve_gib: float = 1.0,
    label: str = "LBM run",
) -> CUDARuntimeReserve | None:
    """Reject a run whose actual persistent allocation consumed its reserve."""
    if device.type != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    reserve = assess_cuda_runtime_reserve(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        required_reserve_gib=required_reserve_gib,
    )
    if not reserve.admitted:
        raise MemoryError(
            f"{label} left only {reserve.free_gib:.2f} GiB after persistent "
            f"allocation; at least {required_reserve_gib:.2f} GiB is required "
            "for collision and diagnostic temporaries",
        )
    return reserve


__all__ = [
    "CUDAMemoryBudget",
    "CUDARuntimeReserve",
    "HierarchyDeviceAllocation",
    "assess_cuda_memory_budget",
    "assess_cuda_runtime_reserve",
    "plan_hierarchy_device_memory",
    "require_cuda_memory_budget",
    "require_cuda_runtime_reserve",
]
