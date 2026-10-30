#!/usr/bin/env python3
"""m479 证据一致性校验：无需 GPU，纯 JSON 比对。

验证 evidence/<run> 与 result.json 的一致性，以及判据门状态。
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
CASE = REPO / "benchmarks" / "verified" / "aircraft_icing_naca0012_rime"


def main():
    # Load result.json
    with open(CASE / "result.json") as f:
        result = json.load(f)

    # Load all evidence files
    evidence = {}
    ev_dir = CASE / "evidence"
    for f in sorted(ev_dir.iterdir()):
        with open(f) as fh:
            evidence[f.name] = json.load(fh)

    print("=" * 60)
    print("m479: Evidence Consistency Verification")
    print("=" * 60)

    # 1. Evidence vs result.json consistency
    print("\n[1] Evidence vs result.json consistency")
    all_consistent = True
    for run_name in result["runs"]:
        if run_name not in evidence:
            print(f"  ✗ MISSING evidence for {run_name}")
            all_consistent = False
            continue
        ev = evidence[run_name]
        ref = result["runs"][run_name]
        checks = [
            ("beta_pk", ev["beta_e_pk"], ref["beta_pk"]),
            ("t_max_mm", ev["t_max_mm"], ref["t_max_mm"]),
            ("euler_deposited_kg", ev["euler_deposited_kg"], ref["euler_deposited_kg"]),
        ]
        run_ok = True
        for key, ev_val, ref_val in checks:
            if abs(ev_val - ref_val) > 1e-10:
                print(f"  ✗ {run_name}.{key}: evidence={ev_val}, result.json={ref_val}")
                all_consistent = False
                run_ok = False
        if run_ok:
            print(f"  ✓ {run_name}: all values match")

    # 2. Gate verification
    print("\n[2] Gate verification")
    all_gates_pass = True
    for gate_name, gate_data in result["gates"].items():
        status = "✓ PASS" if gate_data["pass"] else "✗ FAIL"
        if not gate_data["pass"]:
            all_gates_pass = False
        print(f"  {status} {gate_name}: value={gate_data['value']}")

    # 3. Determinism check
    print("\n[3] Determinism (det_run1 vs det_run2)")
    det1 = evidence.get("det_run1", {})
    det2 = evidence.get("det_run2", {})
    if det1 and det2:
        sha1 = {a["key"]: a["sha256_first8"] for a in det1.get("array_sha256", [])}
        sha2 = {a["key"]: a["sha256_first8"] for a in det2.get("array_sha256", [])}
        match_count = sum(1 for k in sha1 if sha1.get(k) == sha2.get(k))
        total = len(sha1)
        det_ok = match_count == total
        if not det_ok:
            all_gates_pass = False
        print(f"  {'✓' if det_ok else '✗'} {match_count}/{total} arrays match")
        if not det_ok:
            for k in sha1:
                if sha1.get(k) != sha2.get(k):
                    print(f"    ✗ {k}: {sha1.get(k)} vs {sha2.get(k)}")
    else:
        print("  ✗ Missing determinism evidence")
        all_gates_pass = False

    # 4. Figures check
    print("\n[4] Figures check")
    figs_dir = CASE / "figs"
    expected_figs = ["alpha_impact.png", "beta_curve.png", "convergence.png",
                     "g9_lwc_linearity.png", "history.png", "ice_overlay.png"]
    all_figs_ok = True
    for fig in expected_figs:
        exists = (figs_dir / fig).exists()
        if not exists:
            all_figs_ok = False
        print(f"  {'✓' if exists else '✗'} {fig}")

    # Summary
    print("\n" + "=" * 60)
    overall = all_consistent and all_gates_pass and all_figs_ok
    print(f"OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")
    print(f"  Evidence consistency: {'✓' if all_consistent else '✗'}")
    print(f"  All gates pass: {'✓' if all_gates_pass else '✗'}")
    print(f"  All figures present: {'✓' if all_figs_ok else '✗'}")
    print("=" * 60)

    # Save verification result
    out = Path("runs/m479")
    out.mkdir(parents=True, exist_ok=True)
    verify_result = {
        "mission": "m479",
        "case": "aircraft_icing_naca0012_rime",
        "evidence_consistent": all_consistent,
        "all_gates_pass": all_gates_pass,
        "all_figs_present": all_figs_ok,
        "overall_pass": overall,
        "n_runs_checked": len(result["runs"]),
        "n_gates": len(result["gates"]),
        "n_figs": len(expected_figs),
    }
    with open(out / "evidence_verify.json", "w") as f:
        json.dump(verify_result, f, indent=2)
    print(f"\nResult saved to {out / 'evidence_verify.json'}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
