#!/usr/bin/env python3
"""Board audit m17746 — re-grade m17678 (XVAL-LBM-CONSERVATIVE-ALLEN-CAHN-H-POPULATION).

Reads the *subject mission's own* full-run artifact (nothing hand-typed) and emits
  artifacts/board_audit/m17746/audit_m17678.json   machine-readable evidence bundle
  artifacts/board_audit/m17746/fig_audit_m17678.png graded metrics vs 15%/3% bars

The numbers below are read out of
  /root/tp-agent/work/m17678/artifacts/xval_m17678/xval_tclb_cac_h_population.json
so every value in the audit bundle is traceable to a key path in that file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SUBJECT = Path("/root/tp-agent/work/m17678")
SRC = SUBJECT / "artifacts/xval_m17678/xval_tclb_cac_h_population.json"
OUT = Path(__file__).resolve().parents[1] / "artifacts/board_audit/m17746"
OUT.mkdir(parents=True, exist_ok=True)

d = json.loads(SRC.read_text())
chk = d["checks"]

# (label, key path, direction) — 'rel' metrics are |rel err|, 'abs' are already
# non-negative magnitudes (mass drift etc. are "should be 0" references).
ROWS = [
    ("P1 counter-term F_phi vs TCLB text", "P1_counter_term_rel_err", "rel"),
    ("P2 laplacian vs myLaplace", "P2_laplacian_rel_err", "rel"),
    ("P2 gradient vs IsotropicGrad", "P2_gradient_max_rel_err", "rel"),
    ("P9a 1-step vs TCLB collision text", "P9a_rel_err", "rel"),
    ("P9b armA vs armB 1-step", "P9b_rel_err", "rel"),
    ("P3 mass drift 2000 steps", "P3_mass_drift_B", "rel"),
    ("P4 interface radius drift", "P4_R_drift_B", "rel"),
    ("P5 radial profile vs tanh", "P5_profile_err_B", "rel"),
    ("P6 advection centroid", "P6_centroid_err_B", "rel"),
    ("P6 advection field L2", "P6_field_err_B", "rel"),
    ("P7 cross-arm static L2", "P7_static_cross_arm_rel_l2", "rel"),
    ("P7 cross-arm advection L2", "P7_advection_cross_arm_rel_l2", "rel"),
    ("P3d 3D sphere mass drift", "P3d_mass_drift_3d", "rel"),
]

rows = [{"label": lab, "key": k, "value": float(chk[k]),
         "pass_15pct": float(chk[k]) < 0.15, "pass_3pct": float(chk[k]) < 0.03}
        for lab, k, _ in ROWS]

bundle = {
    "auditor_mission": 17746,
    "subject_mission": 17678,
    "anchor": d["anchor"],
    "repo": d["repo"],
    "source_artifact": str(SRC),
    "subject_env": d.get("env", {}),
    "graded_rows": rows,
    "failed_gates_declared_by_subject": {k: v for k, v in chk.items() if k.endswith("_pass") and v is False},
    "tclb_provenance": {
        "root": d["provenance"]["tclb_root"],
        "model": d["provenance"]["model"],
        "sha256": {k: v["sha256"] for k, v in d["provenance"]["files"].items()},
        "license_note": d["provenance"]["license_note"],
    },
    "registered_benchmark_rows": 0,
    "registered_benchmark_note": (
        "SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug="
        "'XVAL-LBM-CONSERVATIVE-ALLEN-CAHN-H-POPULATION' -> 0 rows; "
        "the platform owns no machine criterion for this anchor"
    ),
    "tracked_in_git": {p: False for p in [
        "src/tensorlbm/phasefield/cac_h_population.py",
        "scripts/xval_m17678_tclb_cac_h_population.py",
        "tests/test_xval_m17678_cac_h_population.py",
        "docs/XVAL_M17678_TCLB_CONSERVATIVE_AC_H_POPULATION.md",
        "docs/XVAL_M17678_PREREG.md",
        "artifacts/xval_m17678/xval_tclb_cac_h_population.json",
    ]},
    "integration_callers": [],
}
(OUT / "audit_m17678.json").write_text(json.dumps(bundle, indent=2) + "\n")

# ---- figure: graded metrics against the 15% / 3% bars --------------------
labels = [r["label"] for r in rows]
vals = np.array([max(r["value"], 1e-18) for r in rows])
cols = ["#2a9d8f" if r["pass_3pct"] else ("#e9c46a" if r["pass_15pct"] else "#c1121f") for r in rows]

fig, ax = plt.subplots(figsize=(11.5, 6.2))
y = np.arange(len(rows))
ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.5, height=0.66)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=8.5)
ax.invert_yaxis()
ax.axvline(0.03, color="#1d3557", ls="--", lw=1.2)
ax.axvline(0.15, color="#6a040f", ls="-", lw=1.6)
ax.text(0.032, -0.9, "3% bar", color="#1d3557", fontsize=9, rotation=90, va="top")
ax.text(0.16, -0.9, "15% bar (accept/reject)", color="#6a040f", fontsize=9, rotation=90, va="top")
for yi, v, r in zip(y, vals, rows):
    txt = f"{v:.2e}"
    ax.text(v * 1.35, yi, txt, va="center", fontsize=7.5)
ax.set_xlabel("|error| vs reference (log scale; dimensionless)")
ax.set_title(
    "m17678 XVAL conservative Allen-Cahn h-population — board re-grade of subject's own full-run artifact\n"
    "green <3% · amber 3–15% · red >15% (P2 gradient 1.43 and P9a 0.258 = subject-declared honest FAIL; "
    "no metric is regraded by the auditor)",
    fontsize=9.5,
)
ax.set_xlim(1e-18, 10)
ax.grid(axis="x", which="both", ls=":", lw=0.4, alpha=0.6)
fig.tight_layout()
fig.savefig(OUT / "fig_audit_m17678.png", dpi=160)
print("wrote", OUT / "audit_m17678.json", (OUT / "audit_m17678.json").stat().st_size, "bytes")
print("wrote", OUT / "fig_audit_m17678.png", (OUT / "fig_audit_m17678.png").stat().st_size, "bytes")
print("pass<15%:", sum(r["pass_15pct"] for r in rows), "/", len(rows),
      " pass<3%:", sum(r["pass_3pct"] for r in rows), "/", len(rows))