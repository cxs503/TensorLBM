"""Phase-0 diagnostic: staircase cylinder mask geometry vs R_eff conventions.

Machine-measures, for the OLD Schaefer-Turek channel configs (D=40/80) and the
NEW free-stream ladder (D=40/80/160), using the library cylinder_mask exactly:
  - N_solid (Gauss-circle count), R_area = sqrt(N/pi)
  - row-range R_row = (y_max - y_min + 1)/2 over rows containing solid
  - col-range R_col
  - drift of each convention vs the nominal R = D/2
  - achievable area-lock tolerance: best |R_area - R| over sub-cell r_mask sweep

Pure CPU. Output: diag/geom.json
"""

from __future__ import annotations

import json
import math
import sys

import torch

sys.path.insert(0, "/nfs/wangxi/worktrees/bm_w11/src")
from tensorlbm.boundaries import cylinder_mask  # noqa: E402

DEV = torch.device("cpu")


def measure(nx: int, ny: int, cx: float, cy: float, r_mask: float) -> dict:
    solid = cylinder_mask(nx, ny, cx, cy, r_mask, DEV)
    n = int(solid.sum())
    rows = torch.any(solid, dim=1).nonzero().flatten()
    cols = torch.any(solid, dim=0).nonzero().flatten()
    y0, y1 = int(rows.min()), int(rows.max())
    x0, x1 = int(cols.min()), int(cols.max())
    return {
        "r_mask": r_mask,
        "n_solid": n,
        "R_area": math.sqrt(n / math.pi),
        "row_span": [y0, y1],
        "R_row": (y1 - y0 + 1) / 2.0,
        "col_span": [x0, x1],
        "R_col": (x1 - x0 + 1) / 2.0,
    }


def area_lock_radius(nx, ny, cx, cy, R_target: float) -> tuple[float, dict]:
    """Bisect r_mask so that R_area is as close as possible to R_target."""
    lo, hi = R_target - 1.0, R_target + 1.0
    m = measure(nx, ny, cx, cy, (lo + hi) / 2)
    for _ in range(80):
        mid = (lo + hi) / 2
        m = measure(nx, ny, cx, cy, mid)
        if m["R_area"] > R_target:
            hi = mid
        else:
            lo = mid
    # N is a step function of r_mask; report the closest achievable R_area
    best = min(
        (measure(nx, ny, cx, cy, lo + (hi - lo) * k / 64.0) for k in range(65)),
        key=lambda d: abs(d["R_area"] - R_target),
    )
    return best["r_mask"], best


def main() -> None:
    out = {"old_channel": {}, "new_freestream": {}}

    # ---- OLD channel config (Schaefer-Turek, run.py geometry) ----
    for D in (40, 80):
        dx = 0.1 / D
        nx = int(round(2.2 / dx))
        ny = int(round(0.41 / dx))
        cx = round(0.2 / dx)
        cy = round(0.2 / dx)
        R = 0.05 / dx
        m = measure(nx, ny, cx, cy, R)
        m["R_nominal"] = R
        m["drift_R_row_pct"] = (m["R_row"] / R - 1) * 100
        m["drift_R_area_pct"] = (m["R_area"] / R - 1) * 100
        m["drift_R_col_pct"] = (m["R_col"] / R - 1) * 100
        # effective-Re mismatch if Re were defined with nominal D
        m["Re_eff_over_Re_nominal_row"] = 2 * m["R_row"] / (2 * R)
        m["Re_eff_over_Re_nominal_area"] = 2 * m["R_area"] / (2 * R)
        out["old_channel"][f"D{D}"] = {
            "nx": nx,
            "ny": ny,
            "cx": cx,
            "cy": cy,
            "mask_R": R,
            **m,
        }

    # ---- NEW free-stream ladder ----
    for D in (20, 40, 80, 160):
        R = D / 2.0
        cx, cy = float(D + 10), float(D)  # placeholder centre, symmetric int
        # big enough window so mask is unaffected by edges
        nx, ny = int(cx + 4 * D), int(cy + 4 * D)
        m = measure(nx, ny, cx, cy, R)
        rm, best = area_lock_radius(nx, ny, cx, cy, R)
        out["new_freestream"][f"D{D}"] = {
            "R_nominal": R,
            "mask_R_equals_nominal": m,
            "drift_R_row_pct": (m["R_row"] / R - 1) * 100,
            "drift_R_area_pct": (m["R_area"] / R - 1) * 100,
            "area_lock": {
                "r_mask_found": rm,
                "n_solid": best["n_solid"],
                "R_area": best["R_area"],
                "residual_pct": (best["R_area"] / R - 1) * 100,
                "R_row_at_lock": best["R_row"],
                "drift_R_row_at_lock_pct": (best["R_row"] / R - 1) * 100,
            },
            # row-range lock: r_mask = R - 0.5 exactly -> D_eff = D - 1
            "row_lock": measure(nx, ny, cx, cy, R - 0.5),
        }

    path = "/nfs/wangxi/runs/bm_widen_w11_20261006/a_cyl_steady/diag/geom.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
