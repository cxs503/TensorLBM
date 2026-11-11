"""Cross-validation API: XVAL-LBM-TAYLORGREEN (three-way Taylor-Green decay).

``POST /api/v1/xval/taylor-green`` runs the case spec declared by
:mod:`tensorlbm.xval.case_spec` on every available tool (in-house TensorLBM,
palabos, OpenFOAM icoFoam), compares each tool with the exact Taylor-Green
solution and with the peers, and returns the cross-validation table together
with the machine-checked criteria and the bug attribution.

The endpoint is deliberately synchronous and capped (``n <= 64``,
``steps <= 2000``) so an API call can never monopolise the control plane; the
published configuration (N = 128, 4900 steps) is reproduced with
``python benchmarks/xval_taylor_green/run_xval.py`` (or ``python -m
tensorlbm.xval``), which writes the same artefacts under ``runs/``.

Every number in the response comes from the run -- nothing is hard-coded.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_JOB_TAG = "xval_taylor_green"

_N_MIN, _N_MAX = 16, 64
_STEPS_MIN, _STEPS_MAX = 50, 2000
_TOOLS = ("tensorlbm", "palabos", "openfoam")


class TaylorGreenRequest(BaseModel):
    n: int = Field(32, ge=_N_MIN, le=_N_MAX, description="grid size (square, periodic)")
    re: float = Field(100.0, gt=1.0, le=5000.0, description="Reynolds number U0 L / nu")
    u0: float = Field(0.05, gt=0.0, le=0.1, description="peak initial velocity (low Mach)")
    steps: int = Field(600, ge=_STEPS_MIN, le=_STEPS_MAX)
    sample_every: int = Field(50, ge=1, le=500)
    boost: bool = Field(False, description="superpose a uniform translation (phase criterion)")
    tools: list[str] = Field(
        default_factory=lambda: list(_TOOLS),
        description="tools to run; unavailable ones are reported as frictions",
    )


@router.post("/taylor-green")
def cross_validate_taylor_green(req: TaylorGreenRequest) -> dict:
    """Run the three-way Taylor-Green cross-validation and return the table."""
    unknown = [t for t in req.tools if t not in _TOOLS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown tool(s): {unknown}")
    if "tensorlbm" not in req.tools:
        raise HTTPException(status_code=422, detail="the in-house tool must be included")

    from tensorlbm.xval.campaign import run_and_report
    from tensorlbm.xval.case_spec import TG2DSpec, boosted_spec

    spec = TG2DSpec(
        n=req.n,
        re=req.re,
        u0=req.u0,
        steps=req.steps,
        sample_every=req.sample_every,
        field_steps=(min(req.steps // 2, req.steps - req.sample_every), req.steps),
    )
    if req.boost:
        spec = boosted_spec(spec)

    out = Path(tempfile.mkdtemp(prefix=f"{_JOB_TAG}_"))
    t0 = time.time()
    try:
        payload = run_and_report(spec, out, tools=tuple(req.tools), threads=4)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"cross-validation failed: {exc}") from exc
    payload["job_tag"] = _JOB_TAG
    payload["output_dir"] = str(out)
    payload["elapsed_sec"] = time.time() - t0
    return payload


@router.get("/taylor-green/spec")
def taylor_green_spec(n: int = 128, re: float = 100.0, u0: float = 0.05, boost: bool = False) -> dict:
    """Return the case spec that all three tools consume (``result.case_spec``)."""
    from tensorlbm.xval.case_spec import TG2DSpec, boosted_spec

    spec = boosted_spec(TG2DSpec(n=n, re=re, u0=u0)) if boost else TG2DSpec(n=n, re=re, u0=u0)
    return {"case_spec": spec.to_dict(), "palabos_argv": spec.palabos_argv("<out_dir>")}
