#!/usr/bin/env python3
"""Generate convergence plot for m459 backward_step checkpoint/resume run."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Load convergence data
    with open("artifacts/m459_convergence_curve.json") as f:
        data = json.load(f)
    
    steps = np.array(data["steps"])
    x1_inst = np.array(data["X1_H_inst"])
    x1_win = np.array(data["X1_H_win"])
    
    # Filter out nan values for plotting
    valid = ~np.isnan(x1_inst)
    steps_v = steps[valid]
    x1_inst_v = x1_inst[valid]
    
    valid_win = ~np.isnan(x1_win)
    steps_w = steps[valid_win]
    x1_win_v = x1_win[valid_win]
    
    REF_X1_H = 6.10
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Top panel: instantaneous and windowed X1/H
    ax1.plot(steps_v, x1_inst_v, 'b-', alpha=0.3, linewidth=0.5, label='Instantaneous')
    if len(steps_w) > 0:
        ax1.plot(steps_w, x1_win_v, 'r-', linewidth=2, label='Windowed average')
    ax1.axhline(y=REF_X1_H, color='g', linestyle='--', linewidth=2, label=f'Reference (Gartling 1990): {REF_X1_H}')
    ax1.axhline(y=REF_X1_H * 0.97, color='orange', linestyle=':', linewidth=1, label='±3% tolerance')
    ax1.axhline(y=REF_X1_H * 1.03, color='orange', linestyle=':', linewidth=1)
    
    ax1.set_ylabel('X₁/H')
    ax1.set_title('Gartling 1990 Backward-Facing Step Re=800\nCheckpoint/Resume Convergence (m=32, SDAA)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(8, np.nanmax(x1_inst_v) * 1.1))
    
    # Mark segment boundaries
    seg_steps = [5000, 10000, 15000, 20000, 25000]
    for i, s in enumerate(seg_steps):
        ax1.axvline(x=s, color='gray', linestyle=':', alpha=0.5)
        ax1.text(s, ax1.get_ylim()[1] * 0.95, f'Seg {i+1}', 
                ha='center', va='top', fontsize=8, color='gray')
    
    # Bottom panel: error vs reference
    err_pct = (x1_win_v - REF_X1_H) / REF_X1_H * 100.0
    ax2.plot(steps_w, err_pct, 'r-', linewidth=2)
    ax2.axhline(y=0, color='g', linestyle='--', linewidth=2, label='Target (0% error)')
    ax2.axhline(y=-3, color='orange', linestyle=':', linewidth=1, label='±3% tolerance')
    ax2.axhline(y=3, color='orange', linestyle=':', linewidth=1)
    ax2.fill_between(steps_w, -3, 3, alpha=0.1, color='green', label='Acceptance band')
    
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Error vs Reference (%)')
    ax2.set_title('Error Evolution')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    out_path = Path("artifacts/m459_convergence_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    
    # Also save segment summary plot
    fig2, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    
    with open("artifacts/m459_analysis.json") as f:
        analysis = json.load(f)
    
    seg_ids = [s["seg_id"] for s in analysis["segments_summary"]]
    seg_x1 = [s["final_X1_H"] for s in analysis["segments_summary"]]
    seg_err = [s["err_pct"] for s in analysis["segments_summary"]]
    seg_steps_end = [s["end_step"] for s in analysis["segments_summary"]]
    
    bars = ax3.bar(seg_ids, seg_x1, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axhline(y=REF_X1_H, color='red', linestyle='--', linewidth=2, label=f'Reference: {REF_X1_H}')
    ax3.axhline(y=REF_X1_H * 0.97, color='orange', linestyle=':', linewidth=1, label='±3% band')
    ax3.axhline(y=REF_X1_H * 1.03, color='orange', linestyle=':', linewidth=1)
    
    ax3.set_xlabel('Segment Number')
    ax3.set_ylabel('X₁/H (final value)')
    ax3.set_title('Segment-by-Segment X₁/H Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add step count labels
    for i, (bar, step) in enumerate(zip(bars, seg_steps_end)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{step}\nsteps', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    out_path2 = Path("artifacts/m459_segment_summary.png")
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    print(f"Segment summary saved to {out_path2}")


if __name__ == "__main__":
    main()
