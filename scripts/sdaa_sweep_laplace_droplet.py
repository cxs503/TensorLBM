#!/usr/bin/env python3
"""SDAA sweep for B23 Laplace droplet benchmark (TensorLBM verified case).

Runs the SCMP Shan-Chen static-droplet Laplace law verification on SDAA
device, computing all criteria from real simulation output (zero hand-copy).

Device: 'sdaa' (Hygon DCU).  Device evidence line is captured from
torch.sdaa.get_device_name(0) at runtime.

Triton / compile: the verified run.py uses compile_route which on SDAA
falls back to eager (teco_inductor incomplete — see compile_route.py
docstring).  This script runs **eager** directly (compile_mode='eager')
to avoid the one-shot compile-then-fallback overhead; the step chain is
identical (collide_sc_single_component + stream).  No triton kernels are
invoked in the eager path; TRITON_INTERPRET is not needed but is set as
a precaution in case any transitive import touches triton.

Grid / step choices:
  2D: original radii R=15,25,40 (L=60,100,160) — fast on SDAA (~2 min total).
  3D: reduced radii R=10,15,20 (L=40,60,80) — original R=40/L=160 would
      take ~65 min on SDAA, exceeding the 20-min wall-clock budget.
      The Laplace law (Δp=2σ/R) is scale-invariant; smaller droplets
      still test the linear relationship, though lattice-discreteness
      corrections are slightly larger (documented in results).

Criteria (from README, unchanged):
  1. Per-radius σ consistency: max_i |σ_i − σ_fit|/σ_fit ≤ 3%
  2. Linear fit R² ≥ 0.999
  3. |intercept| / (slope·max(1/R)) ≤ 3%

Output: artifacts/sdaa_laplace_droplet.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Precautionary: if any transitive import touches triton, use interpreter mode.
os.environ.setdefault("TRITON_INTERPRET", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
# Also need benchmarks/ for compile_route if we were to use it; we go eager.
_BENCH_DIR = _REPO_ROOT / "benchmarks"
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import numpy as np
import torch

torch.set_num_threads(32)

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.multiphase3d import collide_sc_single_component_3d  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

# ── Physics constants (from verified run.py) ──────────────────────────
G_LIB = 5.0       # library argument (sign-flipped convention, backward gather)
G_EFF = -5.0      # physical standard-convention SC94 coupling for EOS
TAU = 1.0
RHO_L, RHO_V = 1.957, 0.1596
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


def radius_field(dim: int, L: int, device: torch.device) -> torch.Tensor:
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
        umag = torch.sqrt(ux ** 2 + uy ** 2)
    else:
        rho, ux, uy, uz = macroscopic3d(f)
        umag = torch.sqrt(ux ** 2 + uy ** 2 + uz ** 2)
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


def run_droplet(dim, R, L, device, max_steps, min_steps, sample_interval,
                conv_dp=2e-4, conv_r=2e-3):
    t0 = time.perf_counter()
    rho0 = init_rho(dim, L, R, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    if dim == 2:
        f = equilibrium(rho0, zero, zero)
    else:
        f = equilibrium3d(rho0, zero, zero, zero)
    rr = radius_field(dim, L, device)

    def step_fn(f):
        if dim == 2:
            return stream(collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))
        return stream3d(collide_sc_single_component_3d(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))

    hist = []
    step = 0
    converged = False
    for step in range(1, max_steps + 1):
        f = step_fn(f)
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
                if d_dp < conv_dp and d_r < conv_r and step >= min_steps:
                    converged = True
                    break
    tail = hist[-3:]
    final = {
        "step": int(step), "converged": converged,
        "p_in": float(np.mean([s["p_in"] for s in tail])),
        "p_out": float(np.mean([s["p_out"] for s in tail])),
        "dp": float(np.mean([s["dp"] for s in tail])),
        "dp_std": float(np.std([s["dp"] for s in tail])),
        "rho_in": float(np.mean([s["rho_in"] for s in tail])),
        "rho_out": float(np.mean([s["rho_out"] for s in tail])),
        "R_eq": float(np.mean([s["R_eq"] for s in tail])),
        "R_eq_std": float(np.std([s["R_eq"] for s in tail])),
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
            "R_init": r["R_init"], "R_eq": r["R_eq"], "p_in": r["p_in"],
            "p_out": r["p_out"], "dp": r["dp"], "rho_in": r["rho_in"],
            "rho_out": r["rho_out"], "max_u": r["max_u"],
            "sigma_i": sigma_i, "dev_sigma_pct": dev * 100.0,
            "step": r["step"], "mass_drift": r["mass_drift"],
            "converged": r["converged"],
        })
    max_dev = max(p["dev_sigma_pct"] for p in per_r)
    intercept_frac = abs(b) / (abs(a) * max(inv_r)) if abs(a) > 0 else float("nan")
    return {
        "sigma_eff_fit": sigma_origin,
        "slope_free": float(a), "intercept_free": float(b),
        "R2": r2, "max_dev_sigma_pct": max_dev,
        "intercept_frac_pct": intercept_frac * 100.0,
        "pass_laplace": bool(max_dev <= 3.0 and r2 >= 0.999 and intercept_frac <= 0.03),
        "per_radius": per_r,
    }


def main():
    print("=" * 70)
    print("SDAA Sweep: Laplace Droplet (B23)")
    print("=" * 70)

    # ── Device evidence ──────────────────────────────────────────────
    device_name = "UNKNOWN"
    device_ok = False
    device_err = ""
    try:
        device_name = torch.sdaa.get_device_name(0)
        device_ok = True
        print(f"[device] torch.sdaa.get_device_name(0) = {device_name}")
    except Exception as e:
        device_err = f"{type(e).__name__}: {e}"
        print(f"[device] ERROR: {device_err}")

    dev = torch.device("sdaa")
    # Quick sanity: can we allocate?
    try:
        _t = torch.zeros(4, device=dev)
        _t.fill_(1.0)
        assert _t.sum().item() == 4.0
        print(f"[device] sdaa tensor alloc OK: {_t.device}")
        del _t
    except Exception as e:
        device_ok = False
        device_err = f"tensor alloc failed: {type(e).__name__}: {e}"
        print(f"[device] ERROR: {device_err}")

    triton_interpret = os.environ.get("TRITON_INTERPRET", "not set")
    print(f"[env] TRITON_INTERPRET={triton_interpret}")
    print(f"[mode] eager (no torch.compile; SDAA teco_inductor incomplete)")

    t_global = time.perf_counter()
    results = {
        "benchmark": "laplace_droplet",
        "case_dir": "benchmarks/verified/laplace_droplet",
        "device_evidence": {
            "torch_version": torch.__version__,
            "device_name": device_name,
            "device_ok": device_ok,
            "device_error": device_err,
            "get_device_name_call": "torch.sdaa.get_device_name(0)",
        },
        "config": {
            "model": "SCMP SC94 psi_exp (psi=1-exp(-rho)), G_eff=-5.0, tau=1.0",
            "library_G": G_LIB,
            "eos": "p(rho) = rho/3 + G_eff*psi^2/6 (G_eff=-5.0)",
            "compile_mode": "eager",
            "triton_interpret": triton_interpret,
            "triton_note": "Eager path does not invoke triton kernels; TRITON_INTERPRET set as precaution for transitive imports.",
            "sdaa_compile_note": "SDAA teco_inductor (Torch-SDAA) incomplete for LBM step chain; eager mode used (see compile_route.py docstring).",
        },
        "criteria": {
            "laplace": "2D dp=sigma/R, 3D dp=2*sigma/R; per-radius sigma within 3% of fit; R2>=0.999; |intercept|<=3% of slope/Rmax",
            "thresholds": {"max_dev_sigma_pct": 3.0, "R2_min": 0.999, "intercept_frac_pct_max": 3.0},
        },
    }

    if not device_ok:
        results["status"] = "FAIL"
        results["fail_reason"] = f"SDAA device not usable: {device_err}"
        results["wall_clock_s"] = time.perf_counter() - t_global
        _save(results)
        return

    # ── 2D: original radii ───────────────────────────────────────────
    print("\n--- 2D (D2Q9) ---")
    radii_2d = [(15.0, 60), (25.0, 100), (40.0, 160)]
    rows_2d = []
    for R, L in radii_2d:
        print(f"  2D R={R} L={L} ...", flush=True)
        try:
            final, hist = run_droplet(
                dim=2, R=R, L=L, device=dev,
                max_steps=30000, min_steps=6000, sample_interval=1000)
            final["R_init"] = R
            rows_2d.append(final)
            print(f"    converged={final['converged']} step={final['step']} "
                  f"R_eq={final['R_eq']:.2f} dp={final['dp']:.6f} "
                  f"dt={final['dt_s']:.1f}s")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            rows_2d.append({"R_init": R, "error": f"{type(e).__name__}: {e}"})

    fit_2d = None
    if len(rows_2d) >= 3 and all("error" not in r for r in rows_2d):
        fit_2d = fit_laplace(rows_2d, dim=2)
        print(f"  2D fit: sigma={fit_2d['sigma_eff_fit']:.6f} "
              f"R2={fit_2d['R2']:.6f} max_dev={fit_2d['max_dev_sigma_pct']:.3f}% "
              f"intercept_frac={fit_2d['intercept_frac_pct']:.3f}% "
              f"PASS={fit_2d['pass_laplace']}")
    else:
        print(f"  2D: insufficient valid runs ({len(rows_2d)}), skipping fit")

    # ── 3D: reduced radii (wall-clock constraint) ────────────────────
    print("\n--- 3D (D3Q19) ---")
    radii_3d = [(10.0, 40), (15.0, 60), (20.0, 80)]
    rows_3d = []
    for R, L in radii_3d:
        print(f"  3D R={R} L={L} ...", flush=True)
        try:
            final, hist = run_droplet(
                dim=3, R=R, L=L, device=dev,
                max_steps=20000, min_steps=4000, sample_interval=500)
            final["R_init"] = R
            rows_3d.append(final)
            print(f"    converged={final['converged']} step={final['step']} "
                  f"R_eq={final['R_eq']:.2f} dp={final['dp']:.6f} "
                  f"dt={final['dt_s']:.1f}s")
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")
            rows_3d.append({"R_init": R, "error": f"{type(e).__name__}: {e}"})

    fit_3d = None
    if len(rows_3d) >= 3 and all("error" not in r for r in rows_3d):
        fit_3d = fit_laplace(rows_3d, dim=3)
        print(f"  3D fit: sigma={fit_3d['sigma_eff_fit']:.6f} "
              f"R2={fit_3d['R2']:.6f} max_dev={fit_3d['max_dev_sigma_pct']:.3f}% "
              f"intercept_frac={fit_3d['intercept_frac_pct']:.3f}% "
              f"PASS={fit_3d['pass_laplace']}")
    else:
        print(f"  3D: insufficient valid runs ({len(rows_3d)}), skipping fit")

    wall = time.perf_counter() - t_global
    print(f"\n[done] wall clock: {wall:.1f}s ({wall/60:.1f} min)")

    # ── Assemble criteria verdicts ───────────────────────────────────
    criteria_results = []
    overall_pass = True

    for label, fit, dim in [("2D", fit_2d, 2), ("3D", fit_3d, 3)]:
        if fit is None:
            criteria_results.append({
                "dimension": label, "criterion": "laplace_fit",
                "status": "FAIL", "reason": "insufficient valid runs for fit",
            })
            overall_pass = False
            continue
        # Per-criterion breakdown
        c1 = fit["max_dev_sigma_pct"] <= 3.0
        c2 = fit["R2"] >= 0.999
        c3 = fit["intercept_frac_pct"] <= 3.0
        criteria_results.append({
            "dimension": label,
            "criterion": "per_radius_sigma_consistency",
            "value": fit["max_dev_sigma_pct"],
            "threshold": "<= 3.0%",
            "status": "PASS" if c1 else "FAIL",
        })
        criteria_results.append({
            "dimension": label,
            "criterion": "linear_fit_R2",
            "value": fit["R2"],
            "threshold": ">= 0.999",
            "status": "PASS" if c2 else "FAIL",
        })
        criteria_results.append({
            "dimension": label,
            "criterion": "intercept_fraction",
            "value": fit["intercept_frac_pct"],
            "threshold": "<= 3.0%",
            "status": "PASS" if c3 else "FAIL",
        })
        criteria_results.append({
            "dimension": label,
            "criterion": "laplace_overall",
            "value": fit["pass_laplace"],
            "threshold": "all three above PASS",
            "status": "PASS" if fit["pass_laplace"] else "FAIL",
        })
        if not fit["pass_laplace"]:
            overall_pass = False

    results["dims"] = {
        "2d": fit_2d if fit_2d else {"error": "fit not computed", "raw": rows_2d},
        "3d": fit_3d if fit_3d else {"error": "fit not computed", "raw": rows_3d},
    }
    results["2d_config"] = {"radii": [r[0] for r in radii_2d], "L": [r[1] for r in radii_2d],
                            "max_steps": 30000, "min_steps": 6000, "sample_interval": 1000}
    results["3d_config"] = {"radii": [r[0] for r in radii_3d], "L": [r[1] for r in radii_3d],
                            "max_steps": 20000, "min_steps": 4000, "sample_interval": 500,
                            "note": "Reduced radii (R=10,15,20 vs original 15,25,40) due to SDAA wall-clock constraint; original R=40/L=160 would take ~65 min."}
    results["criteria_results"] = criteria_results
    results["overall_pass"] = overall_pass
    results["status"] = "PASS" if overall_pass else "FAIL"
    results["wall_clock_s"] = wall
    results["wall_clock_min"] = wall / 60.0

    _save(results)
    print(f"\n[output] saved to artifacts/sdaa_laplace_droplet.json")
    print(f"[verdict] {'PASS' if overall_pass else 'FAIL'}")


def _save(results):
    out_dir = Path(__file__).resolve().parents[1] / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sdaa_laplace_droplet.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"[output] {out_path}")


if __name__ == "__main__":
    main()
