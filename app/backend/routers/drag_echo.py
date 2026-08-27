"""CAD slider streaming echo API — B4-P3a first end-to-end interactive loop.

Thin FastAPI layer over
``tensorlbm.ai.geometry_pipeline.GeometryEchoPipeline`` (which wraps any
``DragSurrogateService`` from ``backend.routers.drag_surrogate``): design
sliders (suboff_cad hull-form axes + appendage scales + u_in/Re) are
echoed as C_D curves with deep-ensemble UQ and a guardrail verdict, in a
single process.  The router owns only HTTP concerns:

- request validation (pydantic) -> pipeline call;
- ``DependencyOverride``-able ``get_echo_service`` so tests (and deploys
  without /nfs artifacts) can inject a fixture pipeline;
- JSON shaping of :class:`EchoResult` (``params / re / cd / uq / guard /
  unsupported_channels``).

Service construction (``build_default_echo_service``) wraps the drag
service built from the PR #239 environment (``TENSORLBM_DRAG_CKPT_DIR`` /
``TENSORLBM_DRAG_RUN_DIR`` ...); the ``TENSORLBM_DRAG_ECHO_*`` variables
override device, channel-count device and geometry-cache size.  With
nothing configured the
endpoints answer 503 with the reason — serving is opt-in, never silently
degraded.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from backend.routers.drag_surrogate import build_default_service
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from tensorlbm.ai.geometry_pipeline import (
    PARAM_AXIS_NAMES,
    SWEEP_AXIS_NAMES,
    EchoResult,
    GeometryEchoPipeline,
)
from tensorlbm.ai.inference_service import BackendQueryError

router = APIRouter()

_HULL_TYPES = ("bare_hull", "with_sail", "full")


class EchoParams(BaseModel):
    """One CAD design point (defaults = mother full-appendage hull)."""

    hull_type: str = Field(default="full", description="bare_hull | with_sail | full")
    sail_scale: float = Field(default=1.0, gt=0, description="sail size multiplier")
    fin_scale: float = Field(default=1.0, gt=0, description="fin size multiplier")
    u_in: float = Field(default=0.1, gt=0, description="inlet speed (lattice units)")
    l_over_d_mult: float = Field(default=1.0, gt=0, description="L/D axis (length x m)")
    nose_len_mult: float = Field(default=1.0, gt=0, description="bow segment length x m")
    stern_len_mult: float = Field(default=1.0, gt=0, description="stern taper+cap length x m")
    sail_x_mult: float = Field(default=1.0, gt=0, description="sail axial centre x m")

    @field_validator("hull_type")
    @classmethod
    def _check_hull(cls, v: str) -> str:
        if v not in _HULL_TYPES:
            raise ValueError(f"hull_type must be one of {_HULL_TYPES}, got {v!r}")
        return v

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"hull_type": self.hull_type, "u_in": self.u_in}
        for axis in PARAM_AXIS_NAMES:
            out[axis] = float(getattr(self, axis))
        return out


class EchoRequest(BaseModel):
    """Single-geometry prediction (one slider move)."""

    params: EchoParams = Field(default_factory=EchoParams)
    re_list: list[float] = Field(min_length=1, description="Reynolds numbers to evaluate")

    @field_validator("re_list")
    @classmethod
    def _check_re(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(r) and r > 0 for r in v):
            raise ValueError("re_list entries must be finite and positive")
        if len(v) > 4096:
            raise ValueError("re_list capped at 4096 points per request")
        return v


class EchoSweepRequest(BaseModel):
    """Slider curve: sweep one axis, N geometries, one batched call."""

    axis: str = Field(description=f"one of {SWEEP_AXIS_NAMES}")
    values: list[float] = Field(min_length=1, max_length=256, description="axis values")
    base_params: EchoParams = Field(default_factory=EchoParams)
    re_list: list[float] = Field(min_length=1, description="Reynolds numbers per geometry")

    @field_validator("axis")
    @classmethod
    def _check_axis(cls, v: str) -> str:
        if v not in SWEEP_AXIS_NAMES:
            raise ValueError(f"axis must be one of {SWEEP_AXIS_NAMES}, got {v!r}")
        return v

    @field_validator("values")
    @classmethod
    def _check_values(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(x) and x > 0 for x in v):
            raise ValueError("values entries must be finite and positive")
        return v

    @field_validator("re_list")
    @classmethod
    def _check_re(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(r) and r > 0 for r in v):
            raise ValueError("re_list entries must be finite and positive")
        if len(v) > 4096:
            raise ValueError("re_list capped at 4096 points per request")
        return v


class UQOut(BaseModel):
    lo: list[float]
    hi: list[float]
    mean_std: float
    std: list[float]


class GuardOut(BaseModel):
    flag: str
    score: float
    reasons: list[str]


class EchoResponse(BaseModel):
    params: dict[str, Any]
    re: list[float]
    cd: list[float]
    uq: UQOut
    guard: GuardOut
    confident: bool
    backend: str
    members: list[str]
    unsupported_channels: list[str] = Field(default_factory=list)
    info: dict[str, Any] = Field(default_factory=dict)


class EchoSweepResponse(BaseModel):
    axis: str
    values: list[float]
    results: list[EchoResponse]


class EchoHealthResponse(BaseModel):
    status: str
    backend: str
    members: list[str]
    guard_features: list[str]
    guard_n_fit: int
    grid: dict[str, int]
    device: str
    counts_device: str
    cache_entries: int


_pipeline: GeometryEchoPipeline | None = None
_pipeline_error: str | None = None


def build_default_echo_service() -> GeometryEchoPipeline:
    """Wrap the PR #239 drag service into the echo pipeline (env-driven).

    The underlying service follows ``TENSORLBM_DRAG_*`` (see
    ``backend.routers.drag_surrogate.build_default_service``); the echo
    layer adds ``TENSORLBM_DRAG_ECHO_DEVICE`` (default: the drag device or
    cpu), ``TENSORLBM_DRAG_ECHO_CACHE`` (geometry LRU slots, default 16)
    and ``TENSORLBM_DRAG_ECHO_COUNTS_DEVICE`` (channel-count device:
    ``auto`` — the echo device when it is CUDA, else any CUDA device
    visible to the process, else CPU — or an explicit torch device
    string; the integer counts are bit-identical on both sides, see
    ``docs/geo_fast_20260827.md``).
    """
    device = os.environ.get(
        "TENSORLBM_DRAG_ECHO_DEVICE", os.environ.get("TENSORLBM_DRAG_DEVICE", "cpu")
    )
    counts_device = os.environ.get("TENSORLBM_DRAG_ECHO_COUNTS_DEVICE", "auto")
    cache = int(os.environ.get("TENSORLBM_DRAG_ECHO_CACHE", "16"))
    service = build_default_service()
    return GeometryEchoPipeline(
        service, device=device, counts_device=counts_device, cache_size=cache
    )


def get_echo_service() -> Iterator[GeometryEchoPipeline]:
    """FastAPI dependency yielding the process-wide pipeline (lazily built)."""
    global _pipeline, _pipeline_error
    if _pipeline is None and _pipeline_error is None:
        try:
            _pipeline = build_default_echo_service()
        except Exception as exc:  # noqa: BLE001 — surfaced as HTTP 503 detail
            _pipeline_error = f"{type(exc).__name__}: {exc}"
    if _pipeline is None:
        raise HTTPException(status_code=503, detail=f"drag echo unavailable: {_pipeline_error}")
    yield _pipeline


def set_echo_service(pipeline: GeometryEchoPipeline | None, error: str | None = None) -> None:
    """Test/deploy hook: inject a pipeline (or a failure reason) directly."""
    global _pipeline, _pipeline_error
    _pipeline = pipeline
    _pipeline_error = error


def _to_response(res: EchoResult) -> EchoResponse:
    return EchoResponse(**res.as_dict())


@router.post("/params", response_model=EchoResponse)
def echo_params(
    req: EchoRequest, pipeline: GeometryEchoPipeline = Depends(get_echo_service)
) -> EchoResponse:
    """One design point swept over ``re_list`` — the slider-move echo."""
    try:
        res = pipeline.predict_from_params(req.params.to_dict(), req.re_list)
    except BackendQueryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _to_response(res)


@router.post("/sweep", response_model=EchoSweepResponse)
def echo_sweep(
    req: EchoSweepRequest, pipeline: GeometryEchoPipeline = Depends(get_echo_service)
) -> EchoSweepResponse:
    """Slider curve over one axis (one batched ensemble forward per call)."""
    try:
        results = pipeline.sweep_axis(req.axis, req.values, req.base_params.to_dict(), req.re_list)
    except BackendQueryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return EchoSweepResponse(
        axis=req.axis,
        values=req.values,
        results=[_to_response(r) for r in results],
    )


@router.post("/stl", response_model=EchoResponse)
async def echo_stl(
    file: UploadFile = File(..., description="binary or ASCII STL of the geometry"),
    re_list: str = Form(..., description="JSON list of Reynolds numbers"),
    u_in: float = Form(0.1),
    hull_type: str = Form("full"),
    pipeline: GeometryEchoPipeline = Depends(get_echo_service),
) -> EchoResponse:
    """Arbitrary-STL demo: voxelize -> mask -> honest downgrade (see docs).

    The v3 geometry channels are not derivable from an arbitrary mask, so
    the answer always carries ``unsupported_channels`` and a ``reject``
    guard verdict — out-of-family geometry never gets confident numbers.
    """
    try:
        re_values = [float(r) for r in json.loads(re_list)]
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"re_list must be a JSON list: {exc}") from exc
    suffix = Path(file.filename or "upload.stl").suffix or ".stl"
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        res = pipeline.predict_from_stl(tmp_path, re_values, u_in=u_in, hull_type=hull_type)
    except BackendQueryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return _to_response(res)


@router.get("/health", response_model=EchoHealthResponse)
def health(pipeline: GeometryEchoPipeline = Depends(get_echo_service)) -> EchoHealthResponse:
    """Echo status: backend kind, members, guard space, grid, cache depth."""
    return EchoHealthResponse(
        status="ok",
        backend=pipeline.service.backend.kind,
        members=list(pipeline.service.backend.member_labels()),
        guard_features=list(pipeline.service.guard.feature_names),
        guard_n_fit=getattr(pipeline.service.guard, "n_fit", -1),
        grid={
            "nz": pipeline.grid.nz,
            "ny": pipeline.grid.ny,
            "nx": pipeline.grid.nx,
        },
        device=pipeline.device,
        counts_device=pipeline.counts_device,
        cache_entries=len(pipeline._cache),  # noqa: SLF001 — process introspection
    )
