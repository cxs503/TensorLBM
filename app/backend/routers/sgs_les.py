# ruff: noqa: TC001
"""LBM-LES sub-grid-closure verification API (P465W1-N05).

Two endpoints expose the four-closure verification surface built in
:mod:`tensorlbm.turbulence.sgs_verification`:

``POST /api/v1/calculations/les/sgs_verification``
    Runs a *bounded* verification on the caller's grid: the laminar
    degeneracy matrix (frozen analytic profiles) for the four closures and,
    optionally, a short Taylor-Green run whose dissipation peak is compared
    with the stored spectral-DNS reference when that reference covers the same
    Reynolds number.  Every number in the response is produced by the run.

``GET /api/v1/calculations/les/sgs_verification/report``
    Returns the committed campaign artefact
    (``runs/sgs_les/report.json``) with the acceptance table, or 404 when the
    campaign has not been assembled in this checkout.

Sizing: the endpoint caps the work it accepts (``resolution <= 32``,
``tgv_steps <= 4000``); production-scale campaigns belong to
``scripts/run_sgs_les_campaign.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

_REPO = Path(__file__).resolve().parents[3]
_REPORT = _REPO / "runs" / "sgs_les" / "report.json"
if not _REPORT.exists():  # committed copy (runs/ is gitignored)
    _REPORT = _REPO / "artifacts" / "sgs_les" / "report.json"

_MODELS = ("smagorinsky", "dynamic_smagorinsky", "wale", "vreman")
_RES_MIN, _RES_MAX = 8, 32
_STEPS_MIN, _STEPS_MAX = 10, 4000


class SGSVerificationRequest(BaseModel):
    """Request body for the bounded SGS verification."""

    models: list[Literal["smagorinsky", "dynamic_smagorinsky", "wale", "vreman"]] = Field(
        default_factory=lambda: list(_MODELS),
        description="Sub-grid closures to exercise (subset of the four).",
    )
    resolution: int = Field(
        16, description="Cubic grid size for the frozen-field probes (8-32)."
    )
    tau: float = Field(0.55, description="BGK relaxation time of the probes (>0.5).")
    include_tgv: bool = Field(
        False, description="Also run a short LBM Taylor-Green case (costs seconds)."
    )
    tgv_size: int = Field(16, description="Taylor-Green grid size (8-32).")
    tgv_steps: int = Field(400, description="Taylor-Green steps (10-4000).")
    re: float = Field(400.0, description="Taylor-Green Reynolds number.")
    amplitude: float = Field(0.06, description="Taylor-Green lattice velocity amplitude.")


def _validate(req: SGSVerificationRequest) -> None:
    if not req.models:
        raise HTTPException(status_code=422, detail="models must contain at least one closure")
    if not (_RES_MIN <= req.resolution <= _RES_MAX):
        raise HTTPException(
            status_code=422,
            detail=f"resolution must lie in [{_RES_MIN}, {_RES_MAX}]",
        )
    if not (_RES_MIN <= req.tgv_size <= _RES_MAX):
        raise HTTPException(
            status_code=422, detail=f"tgv_size must lie in [{_RES_MIN}, {_RES_MAX}]"
        )
    if not (_STEPS_MIN <= req.tgv_steps <= _STEPS_MAX):
        raise HTTPException(
            status_code=422,
            detail=f"tgv_steps must lie in [{_STEPS_MIN}, {_STEPS_MAX}]",
        )
    if not req.tau > 0.5:
        raise HTTPException(status_code=422, detail="tau must be > 0.5")
    if not req.re > 0.0:
        raise HTTPException(status_code=422, detail="re must be positive")
    if not abs(req.amplitude) < 0.2:
        raise HTTPException(status_code=422, detail="amplitude must satisfy |a| < 0.2")


@router.post("/sgs_verification")
def run_sgs_verification(req: SGSVerificationRequest) -> dict:
    """Bounded verification of the four SGS closures on the caller's grid."""
    _validate(req)
    from tensorlbm.turbulence import sgs_verification as sv

    t0 = time.time()
    deg = sv.laminar_degeneracy(n=req.resolution, tau=req.tau)
    fields = {}
    for name, entry in deg["fields"].items():
        fields[name] = {
            model: entry["models"][model]
            for model in req.models
            if model in entry["models"]
        }
    out: dict = {
        "laminar_degeneracy": {
            "n": deg["n"],
            "tau": deg["tau"],
            "nu": deg["nu"],
            "threshold": deg["threshold"],
            "fields": fields,
        },
        "checks": {},
    }
    # model-by-model verdicts on the cases where the closure *must* vanish
    for model in req.models:
        zero_cases = [
            f for f, entry in fields.items()
            if entry[model]["expectation"] in ("zero", "zero_frozen_field")
        ]
        ok = all(fields[f][model]["nut_max"] < 1e-9 for f in zero_cases)
        out["checks"][model] = {
            "zero_expected_in": zero_cases,
            "nut_max_in_those_cases": max(
                (fields[f][model]["nut_max"] for f in zero_cases), default=0.0
            ),
            "pass": bool(ok),
        }
    if req.include_tgv:
        tgv = sv.run_tgv(
            req.models[0],
            req.tgv_size,
            re=req.re,
            t_end=req.tgv_steps * (2.0 * 3.141592653589793 / req.tgv_size) * req.amplitude,
            amplitude_lat=req.amplitude,
            record_every=max(1, req.tgv_steps // 100),
        )
        out["tgv"] = {
            "model": req.models[0],
            "n": tgv["n"],
            "re": tgv["re"],
            "nu_lat": tgv["nu_lat"],
            "tau": tgv["tau"],
            "eps_max": tgv["eps_max"],
            "t_peak": tgv["t_peak"],
            "eps_sgs_max": tgv["eps_sgs_max"],
            "blew_up_at_step": tgv["blew_up_at_step"],
            "n_samples": len(tgv["t"]),
        }
        if _REPORT.exists():
            try:
                rep = json.loads(_REPORT.read_text())
                ref = rep.get("tgv_dns") or {}
                if ref.get("eps_max") and abs(float(ref.get("re", req.re)) - req.re) < 1e-9:
                    rel = abs(tgv["eps_max"] - ref["eps_max"]) / ref["eps_max"]
                    out["tgv"]["dns_reference"] = {
                        "eps_max": ref["eps_max"],
                        "rel_error": rel,
                        "within_20pct": bool(rel <= 0.20),
                    }
            except Exception:  # pragma: no cover - artefact is optional
                pass
    out["wall_sec"] = time.time() - t0
    out["artefact"] = str(_REPORT) if _REPORT.exists() else None
    return out


@router.get("/sgs_verification/report")
def get_sgs_verification_report() -> dict:
    """Return the committed four-closure campaign artefact."""
    if not _REPORT.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "campaign artefact missing: run "
                "`python scripts/run_sgs_les_campaign.py --arm assemble`"
            ),
        )
    return json.loads(_REPORT.read_text())
