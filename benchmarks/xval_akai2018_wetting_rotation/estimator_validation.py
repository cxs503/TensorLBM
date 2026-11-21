#!/usr/bin/env python3
"""Independent validation of the sessile-drop angle estimators on *analytic* caps.

Round 4 of ``XVAL-LBM-AKAI2018-GEOMETRICAL-WETTING`` decided pass/fail with a
Kasa circle fit of the ``phi = 0`` contour, and its only "analytic verification"
was a wall-boundary-condition identity: it fed an analytic ``tanh`` interface
through the wetting kernel and read the angle back out with the inverse of the
same construction.  That is an algebraic identity, not a validation (the audit
``development/audit/m17444/verdict_m17444.md`` says so explicitly).

This script validates the *measurement* instead, with fields that contain no
boundary condition at all: an exact ``tanh`` circular cap

    phi(x, y) = tanh((R - |r - r_c|) / xi)

whose static contact angle is ``theta`` by construction (``y_c = y_wall -
R cos theta``), on a grid, with the vapour region below the wall plane set to
-1.  Reported for every ``(theta, R, xi)``:

* ``theta_cap``  -- cap area + contact-width inversion (decision metric),
* ``theta_vol``  -- exact volume + contact-width inversion,
* ``theta_circle_sym`` -- symmetry-pinned circle fit (used by round 4).

Output: ``estimator_validation.json`` in this directory.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from run import (  # noqa: E402
    Y_WALL,
    cap_phi,
    liquid_volume,
    theta_circle_sym,
    theta_from_volume_width,
)
from tensorlbm.wetting.metrics import spherical_cap_inversion  # noqa: E402

THETAS = [30, 45, 60, 75, 90, 105, 120, 135, 150]
RADII = [12, 18, 30]
XIS = [1.5, 2.0, 3.0]
NX, NY = 192, 128


def one(theta: float, R: float, xi: float) -> dict:
    import torch

    phi = cap_phi(NY, NX, theta, R, xi, torch.float64).numpy()
    rec = {"theta_true": float(theta), "R": R, "xi": xi}
    try:
        cap = spherical_cap_inversion(phi, Y_WALL, level=0.0)
        rec["theta_cap"] = cap["theta_deg"]
        rec["contact_width_cap"] = cap["contact_width"]
        rec["extrapolation"] = cap.get("extrapolation")
    except Exception as exc:  # pragma: no cover
        rec["theta_cap"] = float("nan")
        rec["cap_error"] = repr(exc)
        rec["contact_width_cap"] = None
    rec["theta_circle_sym"] = theta_circle_sym(phi)["theta_deg"]
    V = liquid_volume(phi)
    w = rec["contact_width_cap"]
    rec["theta_vol"] = theta_from_volume_width(V, w)["theta_deg"] if w else float("nan")
    for key in ("theta_cap", "theta_vol", "theta_circle_sym"):
        v = rec[key]
        rec[key + "_err_deg"] = (v - theta) if v is not None and math.isfinite(v) else float("nan")
    return rec


def summarise(rows: list[dict], key: str) -> dict:
    errs = np.array([r[key + "_err_deg"] for r in rows], dtype=float)
    ok = np.isfinite(errs)
    return {
        "n": int(len(rows)),
        "n_finite": int(ok.sum()),
        "max_abs_err_deg": float(np.abs(errs[ok]).max()) if ok.any() else float("nan"),
        "median_abs_err_deg": float(np.median(np.abs(errs[ok]))) if ok.any() else float("nan"),
        "p95_abs_err_deg": float(np.percentile(np.abs(errs[ok]), 95)) if ok.any() else float("nan"),
    }


def main() -> int:
    rows = [one(th, R, xi) for th in THETAS for R in RADII for xi in XIS]
    out = {
        "grid": {"theta_deg": THETAS, "R": RADII, "xi": XIS, "nx": NX, "ny": NY,
                 "level": 0.0, "y_wall": Y_WALL},
        "summary": {k: summarise(rows, k)
                    for k in ("theta_cap", "theta_vol", "theta_circle_sym")},
        "rows": rows,
    }
    # per-angle table for the report
    per_angle = {}
    for th in THETAS:
        sel = [r for r in rows if r["theta_true"] == th and math.isfinite(r["theta_cap"])]
        per_angle[str(th)] = {
            "theta_cap_mean": float(np.mean([r["theta_cap"] for r in sel])) if sel else float("nan"),
            "theta_cap_max_abs_err": float(max(abs(r["theta_cap_err_deg"]) for r in sel)) if sel else float("nan"),
        }
    out["per_angle"] = per_angle
    (HERE / "estimator_validation.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))
    print(json.dumps(per_angle, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
