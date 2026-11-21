#!/usr/bin/env python3
"""Run LES channel verification and produce artifacts."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.turbulence.les_channel_verification import run_case, write_artifact

def main():
    print("="*70)
    print("LES CHANNEL FLOW VERIFICATION: Re_tau=180 vs MKM1999 DNS")
    print("="*70)
    
    # Run verification case
    artifact = run_case(
        half_height=8,
        steps=3000,
        warmup_steps=1000,
        u_tau=0.005,
        re_tau=180.0,
        cs=0.10,
        sample_interval=10,
        verbose=True,
    )
    
    # Write artifact
    out_path = ROOT / "artifacts" / "les" / "case_re180.json"
    write_artifact(artifact, out_path)
    
    print("\n" + "="*70)
    print("VERIFICATION RESULTS")
    print("="*70)
    
    for name, check in artifact["criteria"]["checks"].items():
        status = check["status"]
        metric = check.get("metric", "N/A")
        threshold = check.get("threshold", "N/A")
        print(f"{status:5s} {name}: metric={metric}, threshold={threshold}")
    
    print(f"\nOverall: {'ALL PASS' if artifact['criteria']['all_pass'] else 'SOME FAIL'}")
    print(f"Artifact: {out_path}")
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
