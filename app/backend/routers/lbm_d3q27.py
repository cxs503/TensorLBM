# ruff: noqa: TC001 — request model is needed at runtime by FastAPI
"""D3Q27 activation validation API (gaps G7/G8).

``POST /api/v1/calculations/lbm/d3q27_validation`` runs the D3Q27 lattice
ladder of :mod:`tensorlbm.d3q27_validation` through the same
:class:`tensorlbm.general_sim.GeneralSimEngine` path the D3Q19 benchmarks use,
and returns the machine-readable report (lattice isotropy moments, shear-wave
dissipation isotropy, Taylor–Green L2 convergence, sphere Cd at Re = 100/200
for both lattices and the D3Q19→D3Q27 comparison).

Sizing: the endpoint is synchronous and caps the work it will accept
(``resolution ≤ 32``, ``steps ≤ 20000``, at most 2 Reynolds numbers × 2
lattices).  The default request (``resolution=16, steps=3000``) is the
published accuracy-ladder configuration and takes a few minutes on CPU;
callers that need the production-sized grid should run
``benchmarks/d3q27_validation_suite.py`` instead.

Every number in the response is produced by the run — nothing is hard-coded.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_JOB_TAG = "lbm_d3q27_validation"
_DEFAULT_OUT = "/tmp/tensorlbm_api/d3q27_validation"

_RESOLUTION_MIN = 8
_RESOLUTION_MAX = 32
_STEPS_MIN = 10
_STEPS_MAX = 20000
_MAX_REYNOLDS = 2


class D3Q27ValidationRequest(BaseModel):
    """Request body for the D3Q27 accuracy-ladder validation."""

    re: list[float] = Field(
        default_factory=lambda: [100.0, 200.0],
        description="Reynolds numbers (1–2 values, e.g. [100, 200]).",
    )
    resolution: int = Field(
        16, description="Sphere diameter in lattice cells (8–32)."
    )
    steps: int = Field(3000, description="Time steps per sphere case (10–20000).")
    lattices: list[int] = Field(
        default_factory=lambda: [19, 27],
        description="Lattices to run: any subset of [19, 27].",
    )
    tg_sizes: list[int] = Field(
        default_factory=lambda: [16, 32],
        description="Taylor-Green grid sizes for the L2 convergence study.",
    )
    isotropy_n: int = Field(32, description="Box size for the shear-wave isotropy study.")
    figures: bool = Field(False, description="Render PNG figures (needs matplotlib).")
    out_dir: str = Field(
        _DEFAULT_OUT, description="Directory for the report JSON and figures."
    )


def _validate(req: D3Q27ValidationRequest) -> None:
    if not req.re:
        raise HTTPException(status_code=422, detail="re must contain at least one value")
    if len(req.re) > _MAX_REYNOLDS:
        raise HTTPException(
            status_code=422,
            detail=f"at most {_MAX_REYNOLDS} Reynolds numbers per request",
        )
    if any(v <= 0 for v in req.re):
        raise HTTPException(status_code=422, detail="re values must be > 0")
    if not (_RESOLUTION_MIN <= req.resolution <= _RESOLUTION_MAX):
        raise HTTPException(
            status_code=422,
            detail=(
                f"resolution must be in [{_RESOLUTION_MIN}, {_RESOLUTION_MAX}] "
                f"(got {req.resolution})"
            ),
        )
    if not (_STEPS_MIN <= req.steps <= _STEPS_MAX):
        raise HTTPException(
            status_code=422,
            detail=f"steps must be in [{_STEPS_MIN}, {_STEPS_MAX}] (got {req.steps})",
        )
    bad = [q for q in req.lattices if q not in (19, 27)]
    if bad or not req.lattices:
        raise HTTPException(
            status_code=422,
            detail=f"lattices must be a non-empty subset of [19, 27] (got {req.lattices})",
        )
    if req.isotropy_n < 8:
        raise HTTPException(status_code=422, detail="isotropy_n must be >= 8")
    if any(n < 8 for n in req.tg_sizes):
        raise HTTPException(status_code=422, detail="tg_sizes must all be >= 8")


@router.post("/d3q27_validation")
def run_d3q27_validation(req: D3Q27ValidationRequest) -> dict:
    """Run the D3Q27 accuracy ladder and return the report.

    Response keys: ``status``, ``wall_seconds``, ``settings``, ``references``,
    ``isotropy_moments``, ``isotropy_lattice_difference``, ``isotropy_dissipation``,
    ``taylor_green``, ``sphere``, ``comparison``, ``verdicts``,
    ``report_json``, ``figures``.
    """
    _validate(req)
    try:
        from tensorlbm.d3q27_validation import (
            isotropy_lattice_difference,
            run_validation_suite,
        )
    except Exception as exc:  # pragma: no cover - import guard
        raise HTTPException(
            status_code=503, detail=f"tensorlbm.d3q27_validation unavailable: {exc}"
        ) from exc

    out = Path(req.out_dir)
    t0 = time.time()
    try:
        report = run_validation_suite(
            re_values=tuple(req.re),
            lattices=tuple(req.lattices),
            resolution=req.resolution,
            steps=req.steps,
            out_dir=out,
            tg_sizes=tuple(req.tg_sizes),
            isotropy_n=req.isotropy_n,
            make_figures=req.figures,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    report.pop("_fields", None)
    for key, case in report.get("sphere", {}).items():
        for res in case.values():
            res.pop("_fields", None)
    report["isotropy_lattice_difference"] = isotropy_lattice_difference()
    report["status"] = "completed"
    report["job_tag"] = _JOB_TAG
    report["wall_seconds"] = round(time.time() - t0, 2)
    return report