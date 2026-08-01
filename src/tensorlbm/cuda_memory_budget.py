"""Fail-fast CUDA memory budgeting for production LBM campaigns."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


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


__all__ = [
    "CUDAMemoryBudget",
    "assess_cuda_memory_budget",
    "require_cuda_memory_budget",
]
