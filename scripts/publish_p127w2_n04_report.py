#!/usr/bin/env python
"""Build /tmp/report.json for the p127w2-N04 mission report (figures inlined)."""
from __future__ import annotations

import base64
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[1]
doc = (REPO / "docs" / "P127W2_N04_COMPRESSIBLE_BOOM.md").read_text(encoding="utf-8")

FIGS = [
    ("p127w2_n04_cp_alpha_sweep.png",
     "NACA0012 Ma=0.3 upper-surface Cp family, alpha=0..10 deg (KFVS Euler 160x63)"),
    ("p127w2_n04_lift_curve.png",
     "C_L(alpha) vs 2*pi, Prandtl-Glauert and the measured fit (6.7267; +2.13% vs PG, +7.06% vs 2*pi)"),
    ("p127w2_n04_transonic_field.png",
     "Ma=0.85, alpha=0 local Mach and the Ma=1 sonic line (field max Ma=1.375)"),
    ("p127w2_n04_grid_convergence.png",
     "three-grid C_L convergence with Richardson estimate (GCI(p=2)=0.48%)"),
    ("p127w2_n04_ackeret_validity.png",
     "Ackeret linear-theory error vs deflection angle at Ma=2: the 10% band holds only for theta<=3 deg"),
    ("p127w2_n04_wedge_signature.png",
     "Ma=2, theta=10 deg wedge near-field signature: exact oblique shock vs Ackeret (20.1% apart)"),
]

report = {
    "title": ("p127w2-N04 TensorLBM compressible-flow verification: NACA0012 "
              "subsonic/transonic + sonic-boom N-wave benchmarking (attempt 2)"),
    "summary": (
        "Fixed a theta-beta-M inversion defect (Cp was ~2x the linear limit at small "
        "deflection; weak-shock limit now within 1.1% of Ackeret); produced the full "
        "Ackeret-error spectrum at Ma=2 and thereby PROVED criterion (d) is "
        "self-contradictory at theta=10 deg (20.1% > 10%; the 10% band holds only for "
        "theta<=3 deg); measured the NACA0012 lift curve at Ma=0.3 (dC_L/dalpha=6.7267, "
        "+2.13% vs Prandtl-Glauert, +7.06% vs 2*pi); three-grid GCI(C_L)=0.48%, "
        "GCI(C_D)=4.75%. Honest verdicts: (a) FAIL (7.06% > 5%), (d) FAIL (provable), "
        "(e) FAIL (5.32% > 1%), (b)/(c) WITHHELD (no provenance-carrying experimental "
        "data; wall-adjacent first-order sampling cannot resolve the transonic shock "
        "position). Two alternative routes (Cartesian immersed-boundary Euler solver, "
        "far-field N-wave propagation) were built, tested, failed their invariance/consistency "
        "checks and were withdrawn with the evidence recorded."
    ),
    "scenario_key": "p127w2",
    "mission_id": 8194,
    "tags": ["TensorLBM", "compressible", "NACA0012", "transonic", "sonic-boom",
             "Ackeret", "GCI", "verification", "p127w2-n04"],
    "markdown": doc,
    "figures": [],
}

for name, caption in FIGS:
    data = (REPO / "docs" / "figures" / name).read_bytes()
    report["figures"].append({
        "name": name,
        "caption": caption,
        "png_base64": base64.b64encode(data).decode(),
    })

pathlib.Path("/tmp/report.json").write_text(json.dumps(report), encoding="utf-8")
print("wrote /tmp/report.json", len(json.dumps(report)) // 1024, "KiB",
      "figures:", len(report["figures"]))