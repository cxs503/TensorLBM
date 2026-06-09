"""Pydantic request schemas for benchmark endpoints."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

class MarineBenchmarkParams(BaseModel):
    cases: list[
        Literal[
            "cylinder",
            "sloshing",
            "pipeline",
            "turbulent_channel",
            "wigley",
            "suboff",
            "geometry_library",
        ]
    ] = Field(
        default=[
            "cylinder",
            "sloshing",
            "pipeline",
            "turbulent_channel",
            "wigley",
            "suboff",
            "geometry_library",
        ],
        description="Which benchmark cases to run",
    )
    fast: bool = Field(True, description="Use reduced step counts for quick validation")
    device: str = "cpu"

class MultiphaseBenchmarkParams(BaseModel):
    fast: bool = True
    device: str = "cpu"

class GhiaBenchmarkParams(BaseModel):
    nx: int = Field(64, ge=16, description="Grid size (square)")
    re: Literal[100, 400, 1000] = 100
    n_steps: int = Field(5000, ge=1)
    device: str = "cpu"

class MLUPSParams(BaseModel):
    sizes: list[int] = Field(
        default=[128, 256, 512],
        description="Grid sizes to benchmark (nx = ny = size)",
    )
    steps: int = Field(100, ge=10, description="Steps per size")
    device: str = "cpu"

class PorousBenchmarkParams(BaseModel):
    fast: bool = True
    device: str = "cpu"

class AccuracyBenchmarkParams(BaseModel):
    cases: list[
        Literal["cavity", "bfs", "rotating_cylinder"]
    ] = Field(
        default=["cavity", "bfs", "rotating_cylinder"],
        description="Which accuracy benchmark cases to run",
    )
    fast: bool = Field(True, description="Use reduced step counts for quick validation")
    device: str = "cpu"
