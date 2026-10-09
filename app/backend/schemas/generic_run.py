"""Pydantic request schemas for the generic simulation endpoints.

The body is deliberately *generic*: the case type selects a registry
entry (see :mod:`backend.services.generic_run`) that declares the
accepted grid/physics parameters, their defaults and minima.  This keeps
one endpoint for every case instead of one Pydantic model per case.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenericSimRequest(BaseModel):
    """Request body for ``POST /api/sim/generic``."""

    case: str = Field(
        ...,
        description=(
            "Case type, resolved through the case registry: "
            "'cavity' | 'poiseuille' | 'couette' | 'shear_wave' | 'cylinder'"
        ),
    )
    grid: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Grid parameters for the case (e.g. {'nx': 64} for cavity, "
            "{'H': 20} for poiseuille, {'n': 64} for shear_wave, "
            "{'D': 12} for cylinder); omitted keys take the case default"
        ),
    )
    physics: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Physics parameters for the case (e.g. {'Re': 100, 'u_lid': 0.06} "
            "for cavity); omitted keys take the case default"
        ),
    )
    steps: int = Field(
        0,
        ge=0,
        description="Total steps (0 = case default)",
    )
    collision: str = Field(
        "auto",
        description="Collision model: 'auto' (case default), 'bgk' or 'mrt'",
    )
    compile_mode: str | None = Field(
        "default",
        description=(
            "Whole-step routing through tensorlbm.compile_utils, identical to "
            "the benchmark suite: 'eager' (= None passthrough), 'default', or "
            "'max-autotune-no-cudagraphs' (cudagraph-class modes are rejected)"
        ),
    )
    device: str = Field("cpu", description="Torch device (e.g. 'cpu', 'cuda:6')")
    seed: int = Field(0, description="Torch RNG seed for reproducible runs")
    monitor_interval: int = Field(
        0,
        ge=0,
        description="Diagnostics push interval in steps (0 = auto)",
    )
