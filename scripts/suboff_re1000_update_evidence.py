#!/usr/bin/env python3
"""Generate fresh-run convergence figure and update evidence file."""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Load fresh run result
with open(REPO / "runs/suboff_re1000_ckpt_L48/result.json") as f:
    fresh = json.load(f)

# Load verified result
with open(REPO / "benchmarks/verified/suboff_re1000/result.json") as f:
    verified = json.load(f)

Cf_ref = 0.04199504732703608
figdir = REPO / "runs/suboff_re1000_ckpt_L48/figures"
figdir.mkdir(parents=True, exist_ok=True)

# ── Figure 5: Fresh L=48 run convergence (this session) ──
fig, ax = plt.subplots(figsize=(9, 5.5))
ckpts = fresh["checkpoints"]
steps = [c["step"] for c in ckpts]
cd_tots = [c["Cd_tot"] for c in ckpts]
errs = [c["err_pct"] for c in ckpts]

ax.plot(steps, cd_tots, 'bo-', linewidth=2, markersize=7, label='L=48 MRT mix50 (last-100-sample window)')
ax.axhline(y=Cf_ref, color='r', linestyle='--', linewidth=1.5, label=f'Blasius Cf = {Cf_ref:.5f}')
ax.axhline(y=Cf_ref*1.03, color='orange', linestyle=':', linewidth=1.2, label='±3% threshold')
ax.axhline(y=Cf_ref*0.97, color='orange', linestyle=':', linewidth=1.2)
ax.fill_between([0, 3200], Cf_ref*0.97, Cf_ref*1.03, alpha=0.1, color='green', label='PASS zone (±3%)')

# Annotate final point
ax.annotate(f'{cd_tots[-1]:.6f}\n({errs[-1]:+.2f}%)', 
            (steps[-1], cd_tots[-1]), textcoords="offset points",
            xytext=(-60, 15), ha='center', fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black'))

ax.set_xlabel('LBM Time Step', fontsize=12)
ax.set_ylabel('Cd_tot (wetted-area normalization)', fontsize=12)
ax.set_title('Fresh Run: SUBOFF Re=1000 L=48 Convergence (This Session)\n'
             'SDAA eager, 3000 steps, 224.8 ms/step, 674s total', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, 3200)
ax.set_ylim(0.035, 0.085)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(figdir / "fresh_run_convergence_L48.png", dpi=150)
plt.close()
print("Saved fresh_run_convergence_L48.png")

# ── Figure 6: Combined convergence (L=48 fresh + L=80 verified) ──
fig, ax = plt.subplots(figsize=(9, 5.5))

# L=48 fresh run (normalize steps to fraction of total)
frac48 = [s / 3000 for s in steps]
ax.plot([f*100 for f in frac48], errs, 'bs-', linewidth=2, markersize=7, 
        label=f'L=48 fresh (3000 steps, err={errs[-1]:+.2f}%)')

# L=80 verified convergence
conv80 = verified["convergence"]["windows"]
fracs80 = [0.25, 0.50, 0.75, 1.0]
errs80 = [(conv80["25%"] - Cf_ref)/Cf_ref*100,
          (conv80["50%"] - Cf_ref)/Cf_ref*100,
          (conv80["75%"] - Cf_ref)/Cf_ref*100,
          (conv80["100%"] - Cf_ref)/Cf_ref*100]
ax.plot([f*100 for f in fracs80], errs80, 'r^-', linewidth=2, markersize=8,
        label=f'L=80 verified (12000 steps, err={errs80[-1]:+.2f}%)')

ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axhline(y=3, color='orange', linestyle=':', linewidth=1.5, label='±3% threshold')
ax.axhline(y=-3, color='orange', linestyle=':', linewidth=1.5)
ax.fill_between([0, 105], -3, 3, alpha=0.08, color='green')

ax.set_xlabel('Simulation Progress (%)', fontsize=12)
ax.set_ylabel('Error vs Blasius (%)', fontsize=12)
ax.set_title('SUBOFF Re=1000: Convergence Comparison (L=48 fresh vs L=80 verified)\n'
             'Both converge toward Blasius; L=80 reaches -1.15% (PASS)', fontsize=12)
ax.legend(fontsize=9, loc='upper right')
ax.set_xlim(0, 105)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(figdir / "convergence_comparison.png", dpi=150)
plt.close()
print("Saved convergence_comparison.png")

# ── Update evidence file ──
evidence = {
    "case": "suboff_re1000",
    "mission_id": 212,
    "round": 8,
    "timestamp": "2026-10-29 19:58:00",
    "host": "36cb7828dbd5",
    "device_info": {
        "platform": "Linux-6.6.52-1.7.lns23.loongarch64-loongarch64",
        "python": "3.11.6",
        "torch": "2.7.1",
        "sdaa_available": True,
        "sdaa_device_count": 1,
        "device_used": "sdaa:0",
    },
    "reference": {
        "name": "Blasius laminar flat-plate Cf = 1.328/sqrt(Re)",
        "Re": 1000.0,
        "Cf_ref": Cf_ref,
        "normalization": "wetted area S = pi*D*L",
    },
    "threshold": {
        "criterion": "end-to-end Cd_tot within +/-3% of Blasius 0.041995",
        "max_abs_err_pct": 3.0,
    },
    "verified_result_L80": {
        "source": "benchmarks/verified/suboff_re1000/result.json",
        "config": "L=80, MRT, mix50, 12000 steps, 480x169x169 (13.7M cells)",
        "Cd_p": 0.005556427630348497,
        "Cd_f": 0.03595687486402395,
        "Cd_tot": 0.04151330249437245,
        "err_pct": -1.1471467787904794,
        "pass": True,
        "note": "Primary verified result, 2026-09-29"
    },
    "fresh_run_L48": {
        "source": "runs/suboff_re1000_ckpt_L48/result.json",
        "config": "L=48, MRT, mix50, 3000 steps, 288x102x102 (3.0M cells)",
        "device": "sdaa:0",
        "elapsed_s": 674.3,
        "ms_per_step": 224.8,
        "steps_completed": 3000,
        "finite": True,
        "diverged": False,
        "last_100_window": {
            "Cd_p": 0.00684809,
            "Cd_f": 0.03390974,
            "Cd_tot": 0.04075783,
            "err_pct": -2.9461,
            "pass": True,
            "note": "Last 100 samples (steps 2000-3000), most converged window"
        },
        "full_window": {
            "Cd_tot": 0.04560321094329987,
            "err_pct": 8.5919,
            "pass": False,
            "note": "All 300 samples including transient startup - NOT representative"
        },
        "convergence_trend": [
            {"step": 300, "Cd_tot": 0.07280458, "err_pct": 73.36},
            {"step": 600, "Cd_tot": 0.06149471, "err_pct": 46.43},
            {"step": 900, "Cd_tot": 0.05592045, "err_pct": 33.16},
            {"step": 1200, "Cd_tot": 0.04702794, "err_pct": 11.98},
            {"step": 1500, "Cd_tot": 0.04339711, "err_pct": 3.34},
            {"step": 1800, "Cd_tot": 0.04190961, "err_pct": -0.20},
            {"step": 2100, "Cd_tot": 0.0412667, "err_pct": -1.73},
            {"step": 2400, "Cd_tot": 0.04097594, "err_pct": -2.43},
            {"step": 2700, "Cd_tot": 0.04083264, "err_pct": -2.77},
            {"step": 3000, "Cd_tot": 0.04075783, "err_pct": -2.95}
        ],
        "note": "L=48 is coarse grid (hull radius ~2.8 cells). At 3000 steps the last-100-window gives -2.95% (barely PASS). README documents that L=48 at 20000 steps converges to -3.17% (slightly over threshold). L=80 at 12000 steps gives -1.15% (reliable PASS)."
    },
    "bench1_cumulant_L80": {
        "source": "artifacts/suboff_bench1_cumulant_sdaa20.json",
        "config": "L=80, cumulant, 5000 steps, 320x120x120",
        "Cd_tot": 0.04178593600446055,
        "err_pct": 0.49794281917835226,
        "pass": True,
    },
    "overall_assessment": {
        "primary_result": "L=80 MRT mix50 12000 steps: Cd_tot=0.041513, err=-1.15%, PASS (<3%)",
        "fresh_evidence": "L=48 MRT mix50 3000 steps: Cd_tot=0.040758 (last-100-window), err=-2.95%, PASS (<3%) but coarse grid",
        "grid_convergence": "L=48 (-3.17% at 20000 steps) -> L=80 (-1.15% at 12000 steps): error decreases with resolution",
        "verdict": "PASS - primary L=80 result is -1.15%, well within ±3% threshold"
    }
}

with open(REPO / "artifacts/sdaa_suboff_re1000_evidence.json", 'w') as f:
    json.dump(evidence, f, indent=2)
print("\nEvidence file updated.")
print(json.dumps(evidence["overall_assessment"], indent=2))
