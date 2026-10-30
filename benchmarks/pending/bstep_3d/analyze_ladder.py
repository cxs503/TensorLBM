#!/usr/bin/env python3
"""Generate comprehensive report for bstep_3d SDAA resolution ladder study."""
import json
import numpy as np
from pathlib import Path

# Collected data from runs
results = [
    {"h": 10, "nx": 100, "ny": 20, "nz": 20, "er": 2.0, "er_dev_pct": 3.09, 
     "xr_h": 2.8861, "err_pct": -3.80, "steps": 3000, "wall_s": 75.2},
    {"h": 16, "nx": 160, "ny": 33, "nz": 32, "er": 1.9375, "er_dev_pct": 0.13,
     "xr_h": 2.9082, "err_pct": -3.06, "steps": 3000, "wall_s": 78.0},
    {"h": 24, "nx": 240, "ny": 49, "nz": 48, "er": 1.9583, "er_dev_pct": 0.94,
     "xr_h": 2.3802, "err_pct": -20.66, "steps": 3000, "wall_s": 84.5},
]

X_R_REF = 3.0
ER_REF = 1.94

# Analysis
print("="*70)
print("BSTEP_3D SDAA RESOLUTION LADDER DIAGNOSTIC REPORT")
print("="*70)

print("\n1. CASE INTRODUCTION")
print("-" * 70)
print("Physical problem: 3D backward-facing step with spanwise periodicity")
print("Reference: Armaly et al. JFM 127:473-496 (1983)")
print("Re = 100 (based on U_max and step height h)")
print("Reference X_r/h = 3.0, ER_ref = 1.94")
print("Lattice: D3Q19, BGK collision, Zou/He BCs")
print("Device: SDAA (torch.sdaa), /dev/tcaicard4")

print("\n2. METHOD")
print("-" * 70)
print("Resolution ladder: H = 10, 16, 24 (3 levels)")
print("Grid scaling: nx=10H, ny=round(2.05H), nz=2H, x_step=2H")
print("Steps: 3000 per case (insufficient for full convergence)")
print("Measurement: wall u_x zero-crossing (linear interpolation)")
print("  - Spanwise-averaged field for main X_r/h")
print("  - Individual z-planes for uniformity check")

print("\n3. RESULTS TABLE")
print("-" * 70)
print(f"{'H':>3} {'Grid':>15} {'ER_eff':>8} {'ER_err%':>8} {'X_r/h':>8} {'err%':>8} {'wall_s':>8}")
print("-" * 70)
for r in results:
    grid_str = f"({r['nx']},{r['ny']},{r['nz']})"
    print(f"{r['h']:>3} {grid_str:>15} {r['er']:>8.4f} {r['er_dev_pct']:>8.2f} "
          f"{r['xr_h']:>8.4f} {r['err_pct']:>8.2f} {r['wall_s']:>8.1f}")

print("\n4. KEY FINDINGS")
print("-" * 70)

print("\n4.1 ER Discretization Error")
print("  H=10: ny=20, ER=(20-2)/(20-1-10) = 18/9 = 2.000")
print(f"        ER deviation from 1.94: +3.09%")
print("  H=16: ny=33, ER=(33-2)/(33-1-16) = 31/16 = 1.9375")
print(f"        ER deviation from 1.94: -0.13%")
print("  H=24: ny=49, ER=(49-2)/(49-1-24) = 47/24 = 1.9583")
print(f"        ER deviation from 1.94: +0.94%")
print("\n  ⚠ CRITICAL: ER is NOT consistent across resolutions!")
print("  The formula ny=round(2.05H) does NOT produce geometrically similar grids.")
print("  H=10 has 3.1% ER error, which directly affects X_r/h.")

print("\n4.2 X_r/h Convergence")
print("  H=10: X_r/h = 2.8861 (err = -3.80%)")
print("  H=16: X_r/h = 2.9082 (err = -3.06%)")
print("  H=24: X_r/h = 2.3802 (err = -20.66%) ← NOT CONVERGED")
print("\n  The trend is NOT monotonic. H=24 is much worse than H=10/16.")
print("  This indicates 3000 steps is insufficient for H=24 convergence.")

print("\n4.3 ER Contribution to X_r/h Error")
print("  For H=10: ER error = +3.09%, X_r/h error = -3.80%")
print("  For H=16: ER error = -0.13%, X_r/h error = -3.06%")
print("\n  If we assume X_r/h ∝ ER^α (typical α ≈ 1 for BFS):")
print("    H=10: ER contribution ≈ +3.09% → X_r/h should be +3.09% higher")
print("          But observed is -3.80%, so other errors dominate")
print("    H=16: ER contribution ≈ -0.13% → negligible")
print("          Observed -3.06% is from numerical errors (grid resolution)")

print("\n5. ERROR BUDGET ANALYSIS")
print("-" * 70)
print("H=10 error budget:")
print("  - ER discretization: +3.09% (geometry wrong)")
print("  - Grid resolution: ~-6.9% (coarse grid bias)")
print("  - Net: -3.80%")
print("\nH=16 error budget:")
print("  - ER discretization: -0.13% (negligible)")
print("  - Grid resolution: ~-2.9% (moderate grid bias)")
print("  - Net: -3.06%")
print("\nConclusion: The -3.83% SDAA error is a combination of:")
print("  1. ER discretization error (H=10 has 3.1% ER error)")
print("  2. Grid resolution bias (coarse grid overpredicts separation bubble)")
print("  3. Insufficient convergence (3000 steps)")

print("\n6. REATTACHMENT EXTRACTION METHOD")
print("-" * 70)
print("Method: Wall u_x zero-crossing with linear interpolation")
print("  - Sample at y=1 (first fluid node above bottom wall)")
print("  - Find first sign change from negative to positive")
print("  - Linear interpolation: x_zc = x_step + (pos-1) + (0-a)/(b-a)")
print("  - Normalize: X_r/h = (x_zc - (x_step - 0.5)) / h")
print("\nSensitivity: ±0.5 grid units in x → ±0.5/h in X_r/h")
print("  H=10: ±0.05 in X_r/h (±1.7%)")
print("  H=16: ±0.03 in X_r/h (±1.0%)")
print("  This is within the observed error range.")

print("\n7. VERDICT")
print("-" * 70)
print("❌ FAIL: Resolution ladder does NOT meet criteria")
print("\nCriteria check:")
print("  □ Finest grid error ≤ 3%: NO (H=16: -3.06%, H=24: -20.66%)")
print("  □ Monotonic convergence: NO (H=24 << H=10 < H=16)")
print("  □ Finest two grids differ ≤ 1%: NO (H=16 vs H=24: 22%)")
print("\nRoot causes:")
print("  1. ER inconsistency across resolutions (geometry not similar)")
print("  2. Insufficient convergence steps (3000 too few for H=24)")
print("  3. Grid resolution bias (coarse grid overpredicts X_r)")

print("\n8. HONEST-FAIL STATEMENT")
print("-" * 70)
print("The SDAA bstep_3d case with H=10 gives X_r/h = 2.8861 (err = -3.80%),")
print("which matches the reported -3.83% within measurement uncertainty.")
print("\nThis error is REAL and comes from:")
print("  - ER discretization: H=10 has ER=2.0 instead of 1.94 (+3.1%)")
print("  - Grid resolution bias: coarse grid effect (~-6.9%)")
print("  - Net: -3.80% (exceeds 3% threshold by 0.8 percentage points)")
print("\nH=16 with better ER (1.9375, -0.13%) gives -3.06%, still >3%.")
print("H=24 is not converged (3000 steps insufficient).")
print("\nExtrapolation (Richardson): Cannot perform due to non-monotonic convergence.")
print("\nRecommendation:")
print("  1. Fix ER consistency: use ny = floor(ER_ref * (H+1)) + 1 + step_h")
print("  2. Increase steps to 20000+ for convergence")
print("  3. Run H=10, 20, 30, 40 ladder with consistent ER")
print("  4. Consider MRT collision for better accuracy")

print("\n" + "="*70)
print("END OF REPORT")
print("="*70)

# Save structured data
report_data = {
    "case": "bstep_3d_sdaa_diagnostic",
    "mission_id": 411,
    "verdict": "HONEST_FAIL",
    "reference": {"xr_h_ref": X_R_REF, "er_ref": ER_REF},
    "results": results,
    "analysis": {
        "er_inconsistency": True,
        "monotonic_convergence": False,
        "converged": False,
        "root_causes": [
            "ER discretization error (H=10: +3.1%)",
            "Grid resolution bias (coarse grid)",
            "Insufficient convergence steps"
        ],
        "error_budget_h10": {
            "er_contribution_pct": 3.09,
            "grid_bias_pct": -6.89,
            "net_pct": -3.80
        }
    },
    "criteria_check": {
        "finest_error_le_3pct": False,
        "monotonic_convergence": False,
        "finest_two_differ_le_1pct": False,
        "pass": False
    }
}

out_dir = Path("/root/tp-agent/work/m411/artifacts/bstep3d_ladder")
out_dir.mkdir(parents=True, exist_ok=True)
with open(out_dir / "diagnostic_report.json", "w") as f:
    json.dump(report_data, f, indent=2)

print(f"\nReport saved to: {out_dir / 'diagnostic_report.json'}")
