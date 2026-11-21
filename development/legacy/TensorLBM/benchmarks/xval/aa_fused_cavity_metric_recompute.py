"""Re-evaluate the cavity acceptance metric from stored velocity fields.

Why this exists
---------------
``benchmarks/xval/aa_fused_cavity_acceptance.py`` scores the AA-pattern fused
kernel against Ghia et al. (1982).  Two harness defects made that score
meaninglessly bad on the horizontal centre line (``v`` profile):

1. the module compared against a **scrambled hand-embedded copy of Ghia's
   Table II** (values attached to the wrong ``x/L`` stations) instead of the
   provenance-bearing table in
   ``tensorlbm.verification.lid_driven_cavity.reference_data`` -- the residual
   plateaued at ~4.5 % and looked grid independent, i.e. like a solver defect;
2. the horizontal centre line was sampled on cell row ``j = L // 2`` of the
   padded array, which for even ``L`` sits half a cell off ``y/H = 1/2``.

Both are fixed in ``aa_fused_stream_collide_ordering_audit.py`` (single source of
truth for the tables, exact centre-line interpolation).  This script re-reads the
stored converged velocity fields and reports the corrected numbers side by side
with the retired ones, so the size of the harness error is on the record and the
old numbers stay reproducible.

Usage
-----
``python benchmarks/xval/aa_fused_cavity_metric_recompute.py``
scans ``artifacts/xval_aa_pattern/cavity_accept_field_L*.npz`` and any
``cavity_accept_<i>.json`` that carries ``profiles``, and writes
``artifacts/xval_aa_pattern/cavity_accept_corrected.json``.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))

import aa_fused_stream_collide_ordering_audit as A  # noqa: E402
import aa_fused_cavity_acceptance as ACC  # noqa: E402  (metric + floor)

ART = ACC.ART
BOUND_U = A.GHIA_U
BOUND_V = A.GHIA_V
RETIRED_V = A.GHIA_V_RETIRED_SCRAMBLED


def floor(L: int, U: float, ref_v) -> dict:
    """Metric floor for an ``L`` grid, computed exactly the way the metric does."""
    ys = (np.arange(L) + 0.5) / L
    out = {}
    for name, ref in (("u", BOUND_U), ("v", ref_v)):
        exact = np.interp(ys, ref[:, 0], ref[:, 1]) * U
        out[f"floor_{name}"] = ACC.profile_error(exact, ys, ref, U)
        out[f"floor_{name}_interior"] = ACC.profile_error(
            exact, ys, ref, U, interior=True)
    return out


def score(uxc, uyc, ys, U: float) -> dict:
    """Every variant of the metric on one pair of centre-line profiles."""
    out = {}
    for tag, ref_v in (("bound", BOUND_V), ("retired", RETIRED_V)):
        out[f"{tag}_L2_u_full"] = ACC.profile_error(uxc, ys, BOUND_U, U)
        out[f"{tag}_L2_u_interior"] = ACC.profile_error(uxc, ys, BOUND_U, U,
                                                        interior=True)
        out[f"{tag}_L2_v_full"] = ACC.profile_error(uyc, ys, ref_v, U)
        out[f"{tag}_L2_v_interior"] = ACC.profile_error(uyc, ys, ref_v, U,
                                                        interior=True)
    out["worst_full"] = max(out["bound_L2_u_full"], out["bound_L2_v_full"])
    out["gate_15pct"] = bool(out["worst_full"] < 0.15)
    out["gate_3pct"] = bool(out["worst_full"] < 0.03)
    return out


def from_field(path: str) -> dict:
    """Recompute the metric from a stored ``ux, uy`` field (converged snapshots)."""
    d = np.load(path)
    L, wt, U = int(d["L"]), int(d["wt"]), float(d["U"])
    ux = torch.as_tensor(d["ux"])
    uy = torch.as_tensor(d["uy"])
    uxc, uyc, ys = A._sample_profiles_from_moments(ux, uy, L, wt)
    # the retired sampling (half-cell offset) for the record
    js = (np.arange(L) + 0.5) / L
    j = L // 2
    uxc_off = ux[wt:wt + L, wt:wt + L][:, j].numpy()
    uyc_off = uy[wt:wt + L, wt:wt + L][j, :].numpy()
    return {
        "L": L, "U": U, "source": os.path.basename(path),
        "centre_line_weights": list(A._centre_line_weights(L)),
        "corrected": score(uxc, uyc, ys, U),
        "retired_sampling_AND_retired_table": score(uxc_off, uyc_off, ys, U),
        "metric_floor_bound": floor(L, U, BOUND_V),
        "metric_floor_retired": floor(L, U, RETIRED_V),
        "profiles_corrected": {"y_star": ys.tolist(), "ux": np.asarray(uxc).tolist(),
                               "uy": np.asarray(uyc).tolist()},
    }


def from_json(path: str) -> dict:
    """Fallback: re-score the profiles a run stored in its JSON (no field on disk)."""
    d = json.load(open(path))
    p = d.get("profiles")
    if not p:
        return {"source": os.path.basename(path), "skipped": "no profiles"}
    L, U = int(d["L"]), float(d["U"])
    ys = np.asarray(p["y_star"])
    return {
        "L": L, "U": U, "source": os.path.basename(path),
        "note": "profiles as sampled by the retired code (half-cell offset)",
        "corrected_table_on_retired_profiles": score(
            np.asarray(p["ux"]), np.asarray(p["uy"]), ys, U),
        "metric_floor_bound": floor(L, U, BOUND_V),
    }


def main() -> dict:
    out = {"tables": {
        "bound_v": BOUND_V.tolist(),
        "retired_v": RETIRED_V.tolist(),
        "bound_source": "tensorlbm.verification.lid_driven_cavity.reference_data.GHIA_RE100 (provenance BOUND)",
    }, "grids": {}}
    for path in sorted(glob.glob(os.path.join(ART, "cavity_accept_field_L*.npz"))):
        r = from_field(path)
        out["grids"][f"L{r['L']}_field"] = r
    for path in sorted(glob.glob(os.path.join(ART, "cavity_accept_*.json"))):
        r = from_json(path)
        if "skipped" in r:
            continue
        out["grids"][f"L{r['L']}_{r['source'].split('.')[0]}"] = r
    dst = os.path.join(ART, "cavity_accept_corrected.json")
    with open(dst, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {dst}")
    for k, r in out["grids"].items():
        c = r.get("corrected") or r.get("corrected_table_on_retired_profiles")
        print(f"  {k:28s} L2u={c['bound_L2_u_full']:.4f} L2v={c['bound_L2_v_full']:.4f} "
              f"(retired table: L2u={c['retired_L2_u_full']:.4f} L2v={c['retired_L2_v_full']:.4f})")
    return out


if __name__ == "__main__":
    main()