"""Drag-surrogate API — B4 conditional-FNO C_D curves with UQ + guardrails.

Thin FastAPI layer over ``tensorlbm.ai.inference_service.DragSurrogateService``
(the B4-P1d serving layer: deep-ensemble UQ, extrapolation guardrails,
replay/model backends).  The router owns only HTTP concerns:

- request validation (pydantic) -> service call;
- ``DependencyOverride``-able ``get_drag_service`` so tests (and deploys
  without /nfs artifacts) can inject a fixture service;
- JSON shaping of :class:`DragCurveResult` (``re / cd / uq / guard``).

Service construction (``build_default_service``) prefers real member
checkpoints under ``TENSORLBM_DRAG_CKPT_DIR`` and falls back to the
replay backend over ``TENSORLBM_DRAG_RUN_DIR`` (archived preds + cache).
With neither configured the endpoints answer 503 with the reason —
serving is opt-in, never silently degraded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from tensorlbm.ai.inference_service import (
    BackendQueryError,
    DragSurrogateService,
)

router = APIRouter()

_HULL_TYPES = ("bare_hull", "with_sail", "full")


class DragCurveRequest(BaseModel):
    """One design swept over a batch of Reynolds numbers."""

    hull_type: str = Field(description="bare_hull | with_sail | full")
    sail_scale: float = Field(gt=0, description="sail scale factor")
    fin_scale: float = Field(gt=0, description="fin scale factor")
    re_grid: list[float] = Field(min_length=1, description="Reynolds numbers to evaluate")
    u_in: float = Field(default=0.1, gt=0, description="inlet speed (lattice units)")
    field_point: int | None = Field(
        default=None, ge=0, description="optional corpus row for the reference field"
    )

    @field_validator("hull_type")
    @classmethod
    def _check_hull(cls, v: str) -> str:
        if v not in _HULL_TYPES:
            raise ValueError(f"hull_type must be one of {_HULL_TYPES}, got {v!r}")
        return v

    @field_validator("re_grid")
    @classmethod
    def _check_re(cls, v: list[float]) -> list[float]:
        if not all(np.isfinite(r) and r > 0 for r in v):
            raise ValueError("re_grid entries must be finite and positive")
        if len(v) > 4096:
            raise ValueError("re_grid capped at 4096 points per request")
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


class DragCurveResponse(BaseModel):
    hull_type: str
    sail_scale: float
    fin_scale: float
    u_in: float
    backend: str
    members: list[str]
    re: list[float]
    cd: list[float]
    uq: UQOut
    guard: GuardOut
    info: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    backend: str
    members: list[str]
    guard_features: list[str]
    guard_n_fit: int


_service: DragSurrogateService | None = None
_service_error: str | None = None


def build_default_service() -> DragSurrogateService:
    """Build the process service from environment configuration.

    Priority: explicit checkpoints (``TENSORLBM_DRAG_CKPT_DIR`` / ``_FILES``)
    over the replay backend (``TENSORLBM_DRAG_RUN_DIR``, defaulting to the
    archived B4-v4 run).  Raises with a descriptive message when neither
    path exists.
    """


    run_dir = os.environ.get("TENSORLBM_DRAG_RUN_DIR", "/nfs/wangxi/runs/b4_v4_20260824")
    arm = os.environ.get("TENSORLBM_DRAG_ARM", "C_full")
    fold = os.environ.get("TENSORLBM_DRAG_FOLD", "loho::full")
    device = os.environ.get("TENSORLBM_DRAG_DEVICE", "cpu")

    ckpt_files_env = os.environ.get("TENSORLBM_DRAG_CKPT_FILES")
    ckpt_dir_env = os.environ.get("TENSORLBM_DRAG_CKPT_DIR")
    ckpt_paths: list[Path] = []
    if ckpt_files_env:
        ckpt_paths = [Path(p) for p in ckpt_files_env.split(",") if p.strip()]
    elif ckpt_dir_env:
        ckpt_paths = sorted(Path(ckpt_dir_env).glob("*.pt"))
    if ckpt_paths and all(p.is_file() for p in ckpt_paths):
        from tensorlbm.ai.inference_service import load_corpus_index

        index = load_corpus_index(run_dir)
        return DragSurrogateService.from_checkpoints(
            ckpt_paths,
            index.cond,
            corpus_cache=index.fields,
            cache_re=index.re,
            cache_designs=list(index.designs),
            device=device,
        )
    if not ckpt_paths and Path(run_dir).is_dir():
        return DragSurrogateService.from_run_dir(run_dir, arm=arm, fold=fold)
    raise RuntimeError(
        "no drag-surrogate artifacts configured: set TENSORLBM_DRAG_CKPT_DIR/"
        f"TENSORLBM_DRAG_CKPT_FILES or point TENSORLBM_DRAG_RUN_DIR at a valid "
        f"run directory (tried {run_dir!r})"
    )


def get_drag_service() -> Iterator[DragSurrogateService]:
    """FastAPI dependency yielding the process-wide service (lazily built)."""
    global _service, _service_error
    if _service is None and _service_error is None:
        try:
            _service = build_default_service()
        except Exception as exc:  # noqa: BLE001 — surfaced as HTTP 503 detail
            _service_error = f"{type(exc).__name__}: {exc}"
    if _service is None:
        raise HTTPException(status_code=503, detail=f"drag surrogate unavailable: {_service_error}")
    yield _service


def set_drag_service(service: DragSurrogateService | None, error: str | None = None) -> None:
    """Test/deploy hook: inject a service (or a failure reason) directly."""
    global _service, _service_error
    _service = service
    _service_error = error


@router.post("", response_model=DragCurveResponse)
def drag_curve(
    req: DragCurveRequest, service: DragSurrogateService = Depends(get_drag_service)
) -> DragCurveResponse:
    """C_D(re) curve for one design with deep-ensemble UQ and guard verdict."""
    try:
        res = service.predict(
            req.hull_type,
            req.sail_scale,
            req.fin_scale,
            req.re_grid,
            u_in=req.u_in,
            field_point=req.field_point,
        )
    except BackendQueryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DragCurveResponse(
        hull_type=res.hull_type,
        sail_scale=res.sail_scale,
        fin_scale=res.fin_scale,
        u_in=res.u_in,
        backend=res.backend,
        members=list(res.members),
        re=res.re.tolist(),
        cd=res.cd.tolist(),
        uq=UQOut(
            lo=res.lo.tolist(),
            hi=res.hi.tolist(),
            mean_std=float(np.mean(res.std)),
            std=res.std.tolist(),
        ),
        guard=GuardOut(**res.guard.as_dict()),
        info={k: v for k, v in res.info.items() if isinstance(v, (str, int, float, bool, list))},
    )


@router.get("/health", response_model=HealthResponse)
def health(service: DragSurrogateService = Depends(get_drag_service)) -> HealthResponse:
    """Serving status: backend kind, ensemble members, guard feature space."""
    return HealthResponse(
        status="ok",
        backend=service.backend.kind,
        members=list(service.backend.member_labels()),
        guard_features=list(service.guard.feature_names),
        guard_n_fit=getattr(service.guard, "n_fit", -1),
    )
