#!/usr/bin/env python3
"""Mission 209 verification script - reproduce Cd error and convergence from result.json.

This script:
1. Loads result.json (m198 artifacts)
2. Recomputes Cd error vs Schiller-Naumann reference
3. Checks convergence criteria (monotonic error decrease, span ≤3%)
4. Validates all gates
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
RESULT_JSON = os.path.join(BASE, "result.json")

# LOCKED reference (Schiller-Naumann)
CD_REF = 24.0 / 100.0 * (1.0 + 0.15 * 100.0**0.687)
GATE_PCT = 3.0

def main():
    print("=" * 70)
    print("Mission 209 Verification - sphere_re100")
    print("=" * 70)
    
    if not os.path.exists(RESULT_JSON):
        print(f"ERROR: {RESULT_JSON} not found")
        sys.exit(1)
    
    with open(RESULT_JSON) as f:
        data = json.load(f)
    
    print(f"\nLocked Cd_ref (Schiller-Naumann) = {CD_REF:.10f}")
    print(f"Gate = ±{GATE_PCT}%\n")
    
    # Extract grid results
    grids = data.get("grids", {})
    if not grids:
        print("ERROR: No grids found in result.json")
        sys.exit(1)
    
    print("Grid Results:")
    print("-" * 70)
    
    tiers = []
    for tier_name, tier_data in grids.items():
        D = tier_data["D"]
        cd = tier_data["cd"]
        err_pct = tier_data["err_pct"]
        drift_pct = tier_data["drift_pct"]
        steady = tier_data["steady"]
        steps = tier_data["steps"]
        bfl_links = tier_data["bfl_links"]
        
        # Recompute error
        err_recomputed = (cd - CD_REF) / CD_REF * 100.0
        
        within_gate = abs(err_recomputed) <= GATE_PCT
        
        tiers.append({
            "name": tier_name,
            "D": D,
            "cd": cd,
            "err_pct": err_recomputed,
            "drift_pct": drift_pct,
            "steady": steady,
            "within_gate": within_gate,
            "steps": steps,
            "bfl_links": bfl_links,
        })
        
        print(f"{tier_name:8s} D={D:3d}  Cd={cd:.6f}  err={err_recomputed:+.3f}%  "
              f"drift={drift_pct:+.3f}%  steady={steady}  gate={within_gate}  "
              f"steps={steps}  links={bfl_links}")
    
    # Convergence checks
    print("\n" + "=" * 70)
    print("Convergence Analysis:")
    print("-" * 70)
    
    if len(tiers) >= 2:
        # Sort by D
        tiers_sorted = sorted(tiers, key=lambda t: t["D"])
        
        # Check monotonic error decrease
        errors = [abs(t["err_pct"]) for t in tiers_sorted]
        monotone = all(errors[i] > errors[i+1] for i in range(len(errors)-1))
        
        # Check span
        span = abs(tiers_sorted[-1]["err_pct"] - tiers_sorted[0]["err_pct"])
        span_ok = span <= GATE_PCT
        
        # All within gate
        all_gate = all(t["within_gate"] for t in tiers)
        
        # All steady
        all_steady = all(t["steady"] for t in tiers)
        
        print(f"Monotonic |err| decrease: {monotone}")
        print(f"  Errors: {[f'{e:.3f}%' for e in errors]}")
        print(f"Span |err(D_max) - err(D_min)| = {span:.3f}% (gate ≤{GATE_PCT}%)")
        print(f"Span OK: {span_ok}")
        print(f"All within gate: {all_gate}")
        print(f"All steady: {all_steady}")
        
        # Final judgment
        print("\n" + "=" * 70)
        print("JUDGMENT:")
        print("-" * 70)
        
        pass_all = all_gate and all_steady and monotone and span_ok
        
        if pass_all:
            print("✅ PASS - All criteria satisfied")
            print(f"   - Both grids within ±{GATE_PCT}% gate")
            print(f"   - Monotonic error decrease with resolution")
            print(f"   - Span {span:.3f}% ≤ {GATE_PCT}%")
            print(f"   - Steady state achieved")
        else:
            print("❌ FAIL - One or more criteria not satisfied")
            if not all_gate:
                print("   - Not all grids within gate")
            if not all_steady:
                print("   - Not all grids reached steady state")
            if not monotone:
                print("   - Error not monotonically decreasing")
            if not span_ok:
                print(f"   - Span {span:.3f}% > {GATE_PCT}%")
        
        # Write summary
        summary = {
            "mission": 209,
            "case": "sphere_re100",
            "cd_ref": CD_REF,
            "gate_pct": GATE_PCT,
            "tiers": tiers,
            "convergence": {
                "monotone": monotone,
                "span_pct": span,
                "span_ok": span_ok,
                "all_gate": all_gate,
                "all_steady": all_steady,
            },
            "judgment": "PASS" if pass_all else "FAIL",
        }
        
        summary_path = os.path.join(BASE, "verify_m209_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary written to: {summary_path}")
        
        return 0 if pass_all else 1
    else:
        print("ERROR: Need at least 2 tiers for convergence analysis")
        return 1

if __name__ == "__main__":
    sys.exit(main())
