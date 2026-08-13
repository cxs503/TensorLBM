"""Additive marine engineering endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from ..schemas.marine import MarinePreflightResult, ShipCase
from ..services.marine_preflight import evaluate_marine_preflight

router = APIRouter()


@router.post("/preflight", response_model=MarinePreflightResult)
async def marine_preflight(case: ShipCase) -> MarinePreflightResult:
    """Return a versioned ship-case readiness assessment without running a solver."""
    return evaluate_marine_preflight(case)
