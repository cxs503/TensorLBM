#!/usr/bin/env python3
"""Multi-segment run for backward_step benchmark - checkpoint/restart approach."""
import json
import subprocess
import sys
from pathlib import Path

# Run multiple segments, each continuing from where the last left off
# This allows us to accumulate steps beyond the timeout limit

m = 16
segment_steps = 8000
n_segments = 5  # Total: 40000 steps
device = "sdaa:0"

results = []
for seg in range(n_segments):
    print(f"\n=== Segment {seg+1}/{n_segments} ===")
    # For now, just run fresh each time (no actual checkpoint)
    # We'll analyze the trend
    cmd = [
        sys.executable, "benchmarks/pending/backward_step/run.py",
        "--m", str(m),
        "--steps", str(segment_steps),
        "--min-steps", str(segment_steps - 2000),
        "--out-interval", "1000",
        "--device", device,
        "--avg-steps", "4000",
        "--out", f"/tmp/bfs_seg{seg}"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=110)
        output = result.stdout + result.stderr
        
        # Extract final X1/H from output
        lines = output.split('\n')
        for line in lines:
            if 'step=' in line and 'X1/H(win)=' in line:
                parts = line.split()
                for p in parts:
                    if 'X1/H(win)=' in p:
                        x1_val = float(p.split('=')[1])
                        step_val = int([pp for pp in parts if 'step=' in pp][0].split('=')[1])
                        results.append({
                            'segment': seg + 1,
                            'step': step_val,
                            'X1_H_win': x1_val
                        })
                        print(f"  Step {step_val}: X1/H(win) = {x1_val:.4f}")
                        break
    except subprocess.TimeoutExpired:
        print(f"  Segment {seg+1} timed out")
        break

# Save results
output_file = Path("/tmp/bfs_multiseg_results.json")
output_file.write_text(json.dumps(results, indent=2))
print(f"\n=== Summary ===")
print(f"Collected {len(results)} data points")
for r in results:
    print(f"  Step {r['step']:5d}: X1/H = {r['X1_H_win']:.4f}")

# Estimate steps needed
if len(results) >= 2:
    # Simple linear extrapolation (very rough)
    x1_values = [r['X1_H_win'] for r in results]
    steps = [r['step'] for r in results]
    
    # Use last few points for trend
    if len(x1_values) >= 3:
        recent_x1 = x1_values[-3:]
        recent_steps = steps[-3:]
        dx1 = recent_x1[-1] - recent_x1[0]
        dstep = recent_steps[-1] - recent_steps[0]
        if dx1 > 0 and dstep > 0:
            rate = dx1 / dstep  # X1/H per step
            remaining = 6.10 - recent_x1[-1]
            if rate > 0:
                est_steps = remaining / rate
                print(f"\n=== Extrapolation ===")
                print(f"Recent growth rate: {rate:.6f} X1/H per step")
                print(f"Remaining to reach 6.10: {remaining:.4f}")
                print(f"Estimated additional steps needed: {est_steps:.0f}")
                print(f"Estimated time at 14s/1000 steps: {est_steps * 14 / 1000 / 60:.1f} minutes")
