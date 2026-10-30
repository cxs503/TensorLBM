#!/usr/bin/env python
"""SDAA sweep: Laplace droplet benchmark (B23) on SDAA device.

Runs the SCMP single-component Shan-Chen static droplet benchmark on SDAA
for 2D (D2Q9) and optionally 3D (D3Q19), verifying the Young-Laplace law.

Usage:
    python scripts/sdaa_sweep_laplace_droplet.py [--dims 2,3] [--out artifacts/]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import numpy as np
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.multiphase import collide_sc_single_component, psi_exp
from tensorlbm.multiphase3d import collide_sc_single_component_3d
from tensorlbm.solver import stream
from tensorlbm.solver3d import stream3d

# ── Physics constants (same as benchmarks/verified/laplace_droplet/run.py) ──
CS2 = 1.0 / 3.0
G_LIB = 5.0       # library argument (sign-flipped convention)
G_EFF = -5.0      # physical standard-convention SC94 coupling
TAU = 1.0
RHO_L, RHO_V = 1.957, 0.1596  # discrete coexistence
W_INT = 4.0


def eos_pressure(rho: torch.Tensor, g: float = G_EFF) -> torch.Tensor:
    psi = 1.0 - torch.exp(-rho)
    return rho / 3.0 + g * psi * psi / 6.0


def init_rho(dim: int, L: int, R: float, device: torch.device) -> torch.Tensor:
    if dim == 2:
        ys = torch.arange(L, dtype=torch.float32, device=device)
        xs = torch.arange(L, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    else:
        zs = torch.arange(L, dtype=torch.float32, device=device)
        ys = torch.arange(L, dtype=torch.float32, device=device)
        xs = torch.arange(L, dtype=torch.float32, device=device)
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
        r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2 + (zz - L / 2.0) ** 2)
    rho = RHO_V + 0.5 * (RHO_L - RHO_V) * (1.0 + torch.tanh((R - r) / W_INT))
    return rho.clamp(min=1e-3)


def _radius_field(dim: int, L: int, device: torch.device) -> torch.Tensor:
    if dim == 2:
        ys = torch.arange(L, dtype=torch.float32, device=device)
        xs = torch.arange(L, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    zs = torch.arange(L, dtype=torch.float32, device=device)
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2 + (zz - L / 2.0) ** 2)


def measure(dim, f, R_guess, rr, device):
    if dim == 2:
        rho, ux, uy = macroscopic(f)
        umag = torch.sqrt(ux**2 + uy**2)
    else:
        rho, ux, uy, uz = macroscopic3d(f)
        umag = torch.sqrt(ux**2 + uy**2 + uz**2)
    p = eos_pressure(rho)
    r_eq = R_guess
    for _ in range(3):
        inside = rr <= r_eq * 0.5
        outside = rr >= r_eq * 1.5
        rho_in = float(rho[inside].mean().item()) if inside.any() else float("nan")
        rho_out = float(rho[outside].mean().item()) if outside.any() else float("nan")
        mid = 0.5 * (rho_in + rho_out)
        n_liq = int((rho > mid).sum().item())
        if dim == 2:
            r_eq_new = math.sqrt(n_liq / math.pi)
        else:
            r_eq_new = (3.0 * n_liq / (4.0 * math.pi)) ** (1.0 / 3.0)
        if abs(r_eq_new - r_eq) < 1e-4:
            r_eq = r_eq_new
            break
        r_eq = r_eq_new
    inside = rr <= r_eq * 0.5
    outside = rr >= r_eq * 1.5
    p_in = float(p[inside].mean().item()) if inside.any() else float("nan")
    p_out = float(p[outside].mean().item()) if outside.any() else float("nan")
    rho_in = float(rho[inside].mean().item()) if inside.any() else float("nan")
    rho_out = float(rho[outside].mean().item()) if outside.any() else float("nan")
    return {
        "p_in": p_in, "p_out": p_out, "dp": p_in - p_out,
        "rho_in": rho_in, "rho_out": rho_out, "R_eq": r_eq,
        "max_u": float(umag.max().item()), "mass": float(rho.sum().item()),
    }


def run_droplet(dim, R, L, device, max_steps=30000, min_steps=6000, sample_interval=1000):
    t0 = time.perf_counter()
    rho0 = init_rho(dim, L, R, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    if dim == 2:
        f = equilibrium(rho0, zero, zero)
    else:
        f = equilibrium3d(rho0, zero, zero, zero)
    rr = _radius_field(dim, L, device)

    def _step(f):
        if dim == 2:
            return stream(collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))
        return stream3d(collide_sc_single_component_3d(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))

    hist = []
    step = 0
    converged = False
    for step in range(1, max_steps + 1):
        f = _step(f)
        if step % sample_interval == 0:
            rho_cur = f.sum(dim=0)
            if float(rho_cur.min().item()) < 0.0 or not torch.isfinite(rho_cur).all():
                raise RuntimeError(f"dim={dim} R={R}: NaN/negative rho at step {step}")
            m = measure(dim, f, R, rr, device)
            m["step"] = step
            hist.append(m)
            if len(hist) >= 2:
                d_dp = abs(hist[-1]["dp"] - hist[-2]["dp"]) / max(abs(hist[-1]["dp"]), 1e-12)
                d_r = abs(hist[-1]["R_eq"] - hist[-2]["R_eq"]) / max(hist[-1]["R_eq"], 1e-12)
                if d_dp < 2e-4 and d_r < 2e-3 and step >= min_steps:
                    converged = True
                    break
    tail = hist[-3:]
    final = {
        "step": int(step),
        "converged": converged,
        "p_in": float(np.mean([s["p_in"] for s in tail])),
        "p_out": float(np.mean([s["p_out"] for s in tail])),
        "dp": float(np.mean([s["dp"] for s in tail])),
        "rho_in": float(np.mean([s["rho_in"] for s in tail])),
        "rho_out": float(np.mean([s["rho_out"] for s in tail])),
        "R_eq": float(np.mean([s["R_eq"] for s in tail])),
        "max_u": float(np.mean([s["max_u"] for s in tail])),
        "mass_drift": abs(float(tail[-1]["mass"]) - mass0) / mass0,
        "dt_s": time.perf_counter() - t0,
    }
    return final, hist


def fit_laplace(rows, dim):
    inv_r = np.array([1.0 / r["R_eq"] for r in rows])
    y = np.array([r["dp"] for r in rows])
    if dim == 3:
        y = y / 2.0
    sigma_origin = float(np.sum(inv_r * y) / np.sum(inv_r * inv_r))
    A = np.vstack([inv_r, np.ones_like(inv_r)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = a * inv_r + b
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    per_r = []
    for r in rows:
        sigma_i = r["dp"] * r["R_eq"] if dim == 2 else r["dp"] * r["R_eq"] / 2.0
        dev = abs(sigma_i - sigma_origin) / sigma_origin
        per_r.append({
            "R_init": r["R_init"], "R_eq": r["R_eq"],
            "dp": r["dp"], "sigma_i": sigma_i,
            "dev_sigma_pct": dev * 100.0,
            "step": r["step"], "converged": r["converged"],
            "mass_drift": r["mass_drift"],
        })
    max_dev = max(p["dev_sigma_pct"] for p in per_r)
    intercept_frac = abs(b) / (abs(a) * max(inv_r)) if abs(a) > 0 else float("nan")
    return {
        "sigma_eff_fit": sigma_origin,
        "slope_free": float(a),
        "intercept_free": float(b),
        "R2": r2,
        "max_dev_sigma_pct": max_dev,
        "intercept_frac_pct": intercept_frac * 100.0,
        "pass_laplace": bool(max_dev <= 3.0 and r2 >= 0.999 and intercept_frac <= 0.03),
        "per_radius": per_r,
    }


def check_sdaa_device():
    """Check SDAA device availability and return evidence dict."""
    evidence = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "sdaa_available": False,
        "sdaa_device_count": 0,
        "sdaa_devices": [],
    }
    if hasattr(torch, "sdaa") and torch.sdaa.is_available():
        evidence["sdaa_available"] = True
        evidence["sdaa_device_count"] = torch.sdaa.device_count()
        for i in range(torch.sdaa.device_count()):
            props = torch.sdaa.get_device_properties(i)
            evidence["sdaa_devices"].append({
                "index": i,
                "name": torch.sdaa.get_device_name(i),
                "total_memory_mb": props.total_memory,
            })
    return evidence


def main():
    ap = argparse.ArgumentParser(description="SDAA sweep: Laplace droplet benchmark")
    ap.add_argument("--dims", default="2", help="Dimensions to run (comma-separated, e.g. '2' or '2,3')")
    ap.add_argument("--radii", default="15,25,40")
    ap.add_argument("--max-steps", type=int, default=30000)
    ap.add_argument("--min-steps", type=int, default=6000)
    ap.add_argument("--sample-interval", type=int, default=1000)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    radii = [float(x) for x in args.radii.split(",")]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: SDAA device evidence ──
    print("=" * 60)
    print("SDAA Laplace Droplet Sweep")
    print("=" * 60)
    device_evidence = check_sdaa_device()
    print(f"\n[Device Evidence]")
    print(f"  SDAA available: {device_evidence['sdaa_available']}")
    print(f"  SDAA device count: {device_evidence['sdaa_device_count']}")
    for d in device_evidence["sdaa_devices"]:
        print(f"  Device {d['index']}: {d['name']} ({d['total_memory_mb']}MiB)")

    if not device_evidence["sdaa_available"]:
        print("\nERROR: SDAA not available. Cannot proceed.")
        sys.exit(1)

    device = torch.device("sdaa")
    print(f"\nUsing device: {device}")

    # ── Step 2: Run benchmark for each dimension ──
    results = {
        "benchmark": "laplace_droplet",
        "device": "sdaa",
        "device_evidence": device_evidence,
        "model": "SCMP SC94 psi_exp, G_eff=-5.0, tau=1.0, G_lib=+5.0",
        "dims": {},
    }

    for dim in dims:
        print(f"\n{'='*60}")
        print(f"Running {dim}D Laplace droplet on SDAA")
        print(f"{'='*60}")

        rows = []
        for R in radii:
            L = int(4 * R)
            print(f"\n  R={R:.0f}, L={L} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            final, hist = run_droplet(dim, R, L, device,
                                       max_steps=args.max_steps,
                                       min_steps=args.min_steps,
                                       sample_interval=args.sample_interval)
            dt = time.perf_counter() - t0
            final["R_init"] = R
            rows.append(final)
            status = "✅ converged" if final["converged"] else "⚠️ not converged"
            print(f"{status} | step={final['step']} | Δp={final['dp']:.6f} | "
                  f"R_eq={final['R_eq']:.2f} | {dt:.1f}s")

        # ── Step 3: Fit Laplace law ──
        fit = fit_laplace(rows, dim)
        fit["dim"] = dim
        results["dims"][f"{dim}d"] = fit

        print(f"\n  [{dim}D Laplace Fit Results]")
        print(f"    σ_eff = {fit['sigma_eff_fit']:.6f}")
        print(f"    R² = {fit['R2']:.6f}")
        print(f"    max |σ_i - σ_fit|/σ_fit = {fit['max_dev_sigma_pct']:.2f}%")
        print(f"    |intercept|/(slope/Rmax) = {fit['intercept_frac_pct']:.2f}%")
        print(f"    PASS: {fit['pass_laplace']}")

    # ── Step 4: Write evidence file ──
    evidence_path = out_dir / "sdaa_laplace_droplet_evidence.json"
    with open(evidence_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Evidence written to: {evidence_path}")
    print(f"{'='*60}")

    # ── Step 5: Summary ──
    all_pass = all(results["dims"][f"{dim}d"]["pass_laplace"] for dim in dims)
    print(f"\n{'='*60}")
    if all_pass:
        print("✅ ALL DIMENSIONS PASSED Laplace criterion on SDAA")
    else:
        print("❌ SOME DIMENSIONS FAILED Laplace criterion on SDAA")
    print(f"{'='*60}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
