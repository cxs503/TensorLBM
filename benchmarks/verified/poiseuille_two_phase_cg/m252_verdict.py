#!/usr/bin/env python3
"""m252 verdict generator for poiseuille_two_phase_cg.

Reads the formal result.json, computes explicit PASS/FAIL per arm,
records SDAA device evidence, and generates comparison plots.
"""

import json
import hashlib
import base64
import io
import os
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "result.json"
ARTIFACTS_DIR = HERE / "artifacts_m252"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ── Device evidence ──────────────────────────────────────────────────
device_evidence = {}
if hasattr(torch, "sdaa"):
    try:
        device_evidence["sdaa_device_name"] = torch.sdaa.get_device_name(0)
        device_evidence["sdaa_device_count"] = torch.sdaa.device_count()
        device_evidence["device_type"] = "SDAA"
    except Exception as e:
        device_evidence["sdaa_error"] = str(e)
        device_evidence["device_type"] = "SDAA_unavailable"
else:
    device_evidence["device_type"] = "NO_SDAA"

print(f"Device evidence: {device_evidence}")

# ── Load formal result ───────────────────────────────────────────────
result = json.loads(RESULT_PATH.read_text())
gate = result["gate"]
arms = result["arms"]

GATE_RELL2 = gate["relL2_max_per_arm"]  # 0.03 = 3%
GATE_MASS_DRIFT = gate["mass_drift_per_step_per_color"]  # 1e-10

# ── Compute explicit verdict per arm ─────────────────────────────────
verdict_arms = []
all_pass = True

for arm in arms:
    relL2 = arm["relL2"]
    drift_r = arm["mass_drift_r_per_step"]
    drift_b = arm["mass_drift_b_per_step"]
    stable = arm["stable"]
    nan_free = bool(np.isfinite(arm["profile"]).all())

    relL2_pass = relL2 <= GATE_RELL2
    drift_pass = max(drift_r, drift_b) <= GATE_MASS_DRIFT
    arm_pass = relL2_pass and drift_pass and stable and nan_free

    verdict_arms.append({
        "name": arm["name"],
        "ny": arm["ny"],
        "tau_r": arm["tau_r"],
        "tau_b": arm["tau_b"],
        "relL2": relL2,
        "relL2_pct": relL2 * 100,
        "relL2_pass": relL2_pass,
        "u_max_ratio": arm["u_max_ratio"],
        "mass_drift_r_per_step": drift_r,
        "mass_drift_b_per_step": drift_b,
        "mass_drift_max": max(drift_r, drift_b),
        "mass_drift_pass": drift_pass,
        "stable": stable,
        "nan_free": nan_free,
        "pass": arm_pass,
        "j_first_blue": arm.get("j_first_blue"),
        "u_max_frozen_rel_tail20": arm.get("u_max_frozen_rel_tail20"),
    })
    if not arm_pass:
        all_pass = False

    status = "PASS" if arm_pass else "FAIL"
    print(f"  {arm['name']}: relL2={relL2*100:.3f}% {'<=' if relL2_pass else '>'} 3% → {status}")

overall_verdict = "PASS" if all_pass else "FAIL"
print(f"\nOverall VERDICT: {overall_verdict}")

# ── Save verdict JSON ────────────────────────────────────────────────
verdict_doc = {
    "mission_id": 252,
    "case": "poiseuille_two_phase_cg",
    "device_evidence": device_evidence,
    "gate": {
        "relL2_threshold": GATE_RELL2,
        "relL2_threshold_pct": GATE_RELL2 * 100,
        "mass_drift_threshold": GATE_MASS_DRIFT,
    },
    "arms": verdict_arms,
    "overall_verdict": overall_verdict,
    "verdict_headline": f"VERDICT: {overall_verdict} (max relL2={max(a['relL2_pct'] for a in verdict_arms):.3f}%, all arms ≤ 3%)",
    "formal_run_md5": hashlib.md5(RESULT_PATH.read_bytes()).hexdigest(),
}

verdict_path = ARTIFACTS_DIR / "m252_verdict.json"
verdict_path.write_text(json.dumps(verdict_doc, indent=2))
print(f"Saved verdict → {verdict_path}")

# ── Generate comparison plots ────────────────────────────────────────
sys.path.insert(0, str(HERE))
from cg_reference import single_phase_profile_wallplanes, two_layer_profile_wallplanes

figures = []

for arm in arms:
    ny = arm["ny"]
    # Profile includes wall rows; slice to fluid rows only (j=1..ny-2)
    profile = np.array(arm["profile"])[1:-1]
    j_vals = np.arange(1, ny - 1)  # fluid rows

    if arm["tau_r"] == arm["tau_b"]:
        nu = (arm["tau_r"] - 0.5) / 3.0
        ana = single_phase_profile_wallplanes(ny, arm["gx"], nu).numpy()
        label = f"eq (τ={arm['tau_r']})"
    else:
        jb = arm["j_first_blue"]
        nu_lo = (arm["tau_r"] - 0.5) / 3.0
        nu_hi = (arm["tau_b"] - 0.5) / 3.0
        mu_ratio = nu_lo / nu_hi
        ana = two_layer_profile_wallplanes(ny, jb, arm["gx"], nu_lo, nu_hi, mu_ratio).numpy()
        label = f"g10 (τ_r={arm['tau_r']}, τ_b={arm['tau_b']})"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(j_vals, ana, "k-", linewidth=2, label="Analytic (wall-plane convention)")
    ax.plot(j_vals, profile, "r.", markersize=4, label=f"LBM sim (relL2={arm['relL2']*100:.3f}%)")
    ax.set_xlabel("Fluid row index j")
    ax.set_ylabel("Streamwise velocity $u_x$")
    ax.set_title(f"{arm['name']}: ny={ny}, {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode("utf-8")

    fig_path = ARTIFACTS_DIR / f"profile_{arm['name']}.png"
    fig_path.write_bytes(base64.b64decode(png_b64))
    figures.append({
        "name": f"profile_{arm['name']}",
        "caption": f"{arm['name']}: LBM vs analytic velocity profile (relL2={arm['relL2']*100:.3f}%)",
        "png_base64": png_b64,
    })
    print(f"  Saved figure → {fig_path}")

# ── Convergence plot (u_max history) ─────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors = {"eq_ny64": "blue", "eq_ny128": "cyan", "g10_ny64": "red", "g10_ny128": "orange"}
for arm in arms:
    hist = arm.get("u_max_hist", [])
    if hist:
        steps = [h[0] for h in hist]
        umax = [h[1] for h in hist]
        ax.plot(steps, umax, "o-", color=colors.get(arm["name"], "gray"),
                label=f"{arm['name']} (final={umax[-1]:.6e})")
ax.set_xlabel("Step")
ax.set_ylabel("$u_{max}$")
ax.set_title("Convergence: $u_{max}$ vs step (all arms)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
plt.close(fig)
buf.seek(0)
png_b64 = base64.b64encode(buf.read()).decode("utf-8")
conv_path = ARTIFACTS_DIR / "convergence_umax.png"
conv_path.write_bytes(base64.b64decode(png_b64))
figures.append({
    "name": "convergence_umax",
    "caption": "Convergence history of u_max for all 4 arms (log scale)",
    "png_base64": png_b64,
})
print(f"  Saved figure → {conv_path}")

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"VERDICT: {overall_verdict}")
print(f"  Max relL2: {max(a['relL2_pct'] for a in verdict_arms):.3f}% (threshold: 3%)")
print(f"  Max mass drift: {max(a['mass_drift_max'] for a in verdict_arms):.3e} (threshold: 1e-10)")
print(f"  All stable: {all(a['stable'] for a in verdict_arms)}")
print(f"  Device: {device_evidence.get('sdaa_device_name', 'N/A')}")
print(f"{'='*60}")

# Save figures list for report
figures_json_path = ARTIFACTS_DIR / "figures.json"
figures_json_path.write_text(json.dumps([{
    "name": f["name"],
    "caption": f["caption"],
} for f in figures], indent=2))
