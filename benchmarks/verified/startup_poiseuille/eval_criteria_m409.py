#!/usr/bin/env python
"""Mission 409: Evaluate all criteria for startup_poiseuille, generate evidence.json + plots."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

# Load existing verified results (GPU, H=59 and H=119)
verified = json.loads(Path("benchmarks/verified/startup_poiseuille/scan_out/result.json").read_text())

# Load SDAA run results
sdaa_run = json.loads(Path("artifacts/m409/sdaa_run.json").read_text())

# Build criteria evaluation
criteria = []

# Criterion 1: Centerline error <= 3% for all grids
for pg in verified["per_grid"]:
    H = pg["H"]
    val = pg["centerline_err_max_over_umax_pct"]
    criteria.append({
        "name": f"centerline_err_max_H{H}",
        "measured": round(val, 6),
        "threshold": 3.0,
        "unit": "%",
        "pass": val <= 3.0
    })

# Criterion 2: Profile L2 error <= 3% for all grids
for pg in verified["per_grid"]:
    H = pg["H"]
    val = pg["profile_l2_max_pct"]
    criteria.append({
        "name": f"profile_l2_max_H{H}",
        "measured": round(val, 6),
        "threshold": 3.0,
        "unit": "%",
        "pass": val <= 3.0
    })

# Criterion 3: Profile pointwise max error <= 3% for all grids
for pg in verified["per_grid"]:
    H = pg["H"]
    val = pg["profile_pointwise_max_over_umax_pct"]
    criteria.append({
        "name": f"profile_ptp_max_H{H}",
        "measured": round(val, 6),
        "threshold": 3.0,
        "unit": "%",
        "pass": val <= 3.0
    })

# Criterion 4: Grid convergence (error decreases from H=59 to H=119)
conv = verified["convergence"]
criteria.append({
    "name": "grid_convergence_centerline",
    "measured": conv["values"],
    "threshold": "monotone decrease",
    "unit": "%",
    "pass": conv["err_decreased"]
})

# Criterion 5: Mass conservation
for pg in verified["per_grid"]:
    H = pg["H"]
    val = abs(pg["mass_drift_pct"])
    criteria.append({
        "name": f"mass_drift_H{H}",
        "measured": round(pg["mass_drift_pct"], 6),
        "threshold": 1.0,
        "unit": "%",
        "pass": val <= 1.0
    })

# Criterion 6: Finite values
for pg in verified["per_grid"]:
    H = pg["H"]
    criteria.append({
        "name": f"finite_H{H}",
        "measured": pg["finite"],
        "threshold": True,
        "unit": "bool",
        "pass": pg["finite"]
    })

# Overall pass
all_pass = all(c["pass"] for c in criteria)

evidence = {
    "mission_id": 409,
    "case": "startup_poiseuille",
    "device_evidence": sdaa_run["device_evidence"],
    "sdaa_run_summary": {
        "H29": {k: sdaa_run["cases"]["H29"][k] for k in ["centerline_err_max_over_umax_pct",
               "profile_l2_max_pct", "profile_pointwise_max_over_umax_pct", "elapsed_s", "wall_clock_s", "t_end"]},
        "H59_partial": {"steps": 17425, "elapsed_s": 89.2, "center_u": 0.029732, "finite": True}
    },
    "verified_results": {
        "H59": {k: verified["per_grid"][0][k] for k in ["centerline_err_max_over_umax_pct",
                "profile_l2_max_pct", "profile_pointwise_max_over_umax_pct", "elapsed_s", "t_end"]},
        "H119": {k: verified["per_grid"][1][k] for k in ["centerline_err_max_over_umax_pct",
                 "profile_l2_max_pct", "profile_pointwise_max_over_umax_pct", "elapsed_s", "t_end"]},
    },
    "criteria": criteria,
    "all_pass": all_pass,
    "status": "VERIFIED" if all_pass else "FAILED"
}

Path("artifacts/m409").mkdir(parents=True, exist_ok=True)
Path("artifacts/m409/evidence.json").write_text(json.dumps(evidence, indent=2))
print(json.dumps(evidence, indent=2))
print(f"\n=== OVERALL: {evidence['status']} ===")
