#!/usr/bin/env python3
"""Analyze checkpoint/resume run results and generate convergence report."""

import json
from pathlib import Path
import numpy as np

def main():
    run_dir = Path("artifacts/m459_run")
    
    # Load all segment results
    segments = []
    for i in range(1, 6):
        seg_file = run_dir / f"seg{i}.json"
        if seg_file.exists():
            with open(seg_file) as f:
                seg = json.load(f)
                segments.append(seg)
    
    if not segments:
        print("No segment results found!")
        return
    
    # Extract convergence data
    steps = []
    x1_inst = []
    x1_win = []
    
    for seg in segments:
        for pt in seg["series"]:
            steps.append(pt["step"])
            x1_inst.append(pt["X1_H_inst"])
            x1_win.append(pt["X1_H_win"])
    
    steps = np.array(steps)
    x1_inst = np.array(x1_inst)
    x1_win = np.array(x1_win)
    
    # Reference value
    REF_X1_H = 6.10
    ERR_TOL_PCT = 3.0
    
    # Analysis
    print("=" * 70)
    print("Gartling 1990 Backward-Facing Step Re=800 — Convergence Analysis")
    print("=" * 70)
    print(f"\nTotal steps run: {steps[-1]}")
    print(f"Number of segments: {len(segments)}")
    print(f"Reference X1/H: {REF_X1_H}")
    print(f"Error tolerance: {ERR_TOL_PCT}%\n")
    
    # Segment summary
    print("Segment Summary:")
    print("-" * 70)
    print(f"{'Seg':>4} {'Steps':>8} {'End Step':>10} {'X1/H':>8} {'Error %':>10}")
    print("-" * 70)
    for seg in segments:
        err = (seg["final_X1_H"] - REF_X1_H) / REF_X1_H * 100.0
        print(f"{seg['seg_id']:>4} {seg['steps_run']:>8} {seg['end_step']:>10} "
              f"{seg['final_X1_H']:>8.4f} {err:>+10.2f}%")
    
    # Convergence check
    print("\n" + "=" * 70)
    print("Convergence Assessment")
    print("=" * 70)
    
    # Check if X1/H is stabilizing
    final_x1 = segments[-1]["final_X1_H"]
    final_err = (final_x1 - REF_X1_H) / REF_X1_H * 100.0
    
    print(f"\nFinal X1/H: {final_x1:.4f}")
    print(f"Final error: {final_err:+.2f}%")
    print(f"Within tolerance: {abs(final_err) <= ERR_TOL_PCT}")
    
    # Check drift in last 3 segments
    if len(segments) >= 3:
        last3 = [s["final_X1_H"] for s in segments[-3:]]
        mean_last3 = np.mean(last3)
        drift_pct = (max(last3) - min(last3)) / mean_last3 * 100.0
        converged = drift_pct < 1.0
        
        print(f"\nLast 3 segments X1/H: {[f'{v:.4f}' for v in last3]}")
        print(f"Mean: {mean_last3:.4f}")
        print(f"Drift: {drift_pct:.2f}%")
        print(f"Converged (drift <1%): {converged}")
    else:
        converged = False
        print("\nInsufficient segments for convergence check")
    
    # Physical interpretation
    print("\n" + "=" * 70)
    print("Physical Interpretation")
    print("=" * 70)
    
    if final_x1 < 3.0:
        print("\n⚠️  X1/H is significantly below reference (6.10)")
        print("   Possible causes:")
        print("   1. Grid too coarse (m=32, H=64 cells)")
        print("   2. Insufficient steps (25k vs recommended 150k+)")
        print("   3. Flow not yet reached steady state")
        print("   4. Numerical dissipation delaying recirculation zone development")
    elif abs(final_err) <= ERR_TOL_PCT:
        print("\n✓ X1/H within 3% of reference")
    else:
        print(f"\n⚠️  X1/H error {final_err:+.2f}% exceeds 3% tolerance")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("Recommendations")
    print("=" * 70)
    
    if not converged or abs(final_err) > ERR_TOL_PCT:
        print("\nTo complete the benchmark:")
        print("1. Continue checkpoint/resume to accumulate ≥150k steps")
        print("2. Consider finer grid (m=48 or m=64) for better accuracy")
        print("3. Monitor X1/H drift over consecutive 10k-step windows")
        print("4. Convergence criterion: drift <1% over 3 consecutive windows")
        print("\nEstimated time remaining:")
        avg_rate = np.mean([s["rate_steps_s"] for s in segments])
        remaining_steps = 150000 - steps[-1]
        remaining_time = remaining_steps / avg_rate
        print(f"   Current: {steps[-1]} steps")
        print(f"   Target: 150000 steps")
        print(f"   Remaining: {remaining_steps} steps")
        print(f"   At {avg_rate:.0f} steps/s: {remaining_time/60:.1f} minutes")
    
    # Save analysis
    analysis = {
        "total_steps": int(steps[-1]),
        "num_segments": len(segments),
        "final_X1_H": float(final_x1),
        "final_err_pct": float(final_err),
        "within_tolerance": bool(abs(final_err) <= ERR_TOL_PCT),
        "converged": bool(converged) if len(segments) >= 3 else None,
        "reference_X1_H": REF_X1_H,
        "segments_summary": [
            {
                "seg_id": s["seg_id"],
                "end_step": s["end_step"],
                "final_X1_H": s["final_X1_H"],
                "err_pct": (s["final_X1_H"] - REF_X1_H) / REF_X1_H * 100.0,
            }
            for s in segments
        ],
    }
    
    out_path = Path("artifacts/m459_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✓ Analysis saved to {out_path}")
    
    # Save convergence curve data
    curve_path = Path("artifacts/m459_convergence_curve.json")
    with open(curve_path, "w") as f:
        json.dump({
            "steps": steps.tolist(),
            "X1_H_inst": x1_inst.tolist(),
            "X1_H_win": x1_win.tolist(),
        }, f, indent=2)
    
    print(f"✓ Convergence curve saved to {curve_path}")


if __name__ == "__main__":
    main()
