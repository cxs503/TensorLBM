#!/usr/bin/env python
"""CPU verification wrapper for droplet_oscillation benchmark.

Verifies the code path works on CPU by running a minimal 200-step test,
then validates the existing result.json pass criteria programmatically.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import collide_sc_single_component, psi_exp
from tensorlbm.solver import stream

TAU = 1.0
G_LIB = 5.0
RHO_L_INIT, RHO_V_INIT = 1.957, 0.1596
W_INT = 4.0
EPS = 0.05


def init_ellipse_rho(L, R0, eps, rho_l, rho_v, device):
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    dx, dy = xx - L / 2.0, yy - L / 2.0
    r = torch.sqrt(dx * dx + dy * dy)
    theta = torch.atan2(dy, dx)
    r_surf = R0 * (1.0 + eps * torch.cos(2.0 * theta))
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + torch.tanh((r_surf - r) / W_INT))
    return rho.clamp(min=1e-3)


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"CPU verification of droplet_oscillation benchmark\n")

    # Minimal test: R=16, 200 steps to verify code path
    R0 = 16
    L = int(4 * R0)
    print(f"Running minimal CPU smoke test: R={R0}, L={L}, 200 steps")

    t0 = time.perf_counter()
    rho0 = init_ellipse_rho(L, R0, EPS, RHO_L_INIT, RHO_V_INIT, device)
    mass0 = float(rho0.sum().item())
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)

    # Run 200 steps
    for step in range(1, 201):
        f = stream(collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))
        if step % 50 == 0:
            rho = f.sum(dim=0)
            assert float(rho.min().item()) >= 0.0, f"Negative rho at step {step}"
            assert torch.isfinite(rho).all(), f"NaN/Inf at step {step}"
            rho_in, ux, uy = macroscopic(f)
            mass = float(rho.sum().item())
            print(f"  step={step}: mass={mass:.2f} drift={abs(mass-mass0)/mass0:.2e} "
                  f"max_u={float(torch.sqrt(ux*ux+uy*uy).max().item()):.4f}")

    dt = time.perf_counter() - t0
    print(f"\n  Smoke test passed in {dt:.1f}s")

    # Validate existing result.json
    result_path = Path(__file__).parent / "result.json"
    assert result_path.exists(), f"result.json not found at {result_path}"
    with open(result_path) as f:
        existing = json.load(f)

    print(f"\n  Validating existing result.json:")
    cases = existing.get("cases", {})
    validation = {}
    all_pass = True
    monotone = True
    prev_err = None

    for key in ["R128", "R160", "R224"]:
        if key not in cases:
            print(f"    {key}: MISSING")
            validation[key] = {"status": "MISSING"}
            continue
        c = cases[key]
        err = c.get("err_pct_primary", float("nan"))
        r2 = c.get("fit_r2", 0)
        mass = c.get("mass_drift", 1)
        sigma_rel = c.get("sigma_omega_rel_pct", 100)
        gamma_over_omega = c.get("gamma_over_omega_d", 1)

        err_ok = abs(err) <= 3.0
        r2_ok = r2 >= 0.99
        mass_ok = mass < 2e-3
        sigma_ok = sigma_rel < 1.0

        level_pass = err_ok and r2_ok and mass_ok and sigma_ok
        if not level_pass:
            all_pass = False
        if prev_err is not None and abs(err) >= prev_err:
            monotone = False
        prev_err = abs(err)

        status = "PASS" if level_pass else "FAIL"
        print(f"    {key}: err={err:+.4f}% r2={r2:.6f} mass_drift={mass:.2e} "
              f"sigma_rel={sigma_rel:.2f}% gamma/omega={gamma_over_omega:.3f} -> {status}")
        validation[key] = {
            "err_pct": err, "r2": r2, "mass_drift": mass,
            "sigma_omega_rel_pct": sigma_rel, "gamma_over_omega_d": gamma_over_omega,
            "err_ok": err_ok, "r2_ok": r2_ok, "mass_ok": mass_ok, "sigma_ok": sigma_ok,
            "status": status,
        }

    # Check monotone convergence
    errs = [abs(validation[k]["err_pct"]) for k in ["R128", "R160", "R224"]
            if k in validation and validation[k]["status"] != "MISSING"]
    for i in range(len(errs) - 1):
        if errs[i + 1] >= errs[i]:
            monotone = False

    overall_pass = all_pass and monotone
    print(f"\n  Overall: all_within_3pct={all_pass}, monotone={monotone}")
    print(f"  VERDICT: {'PASS' if overall_pass else 'FAIL'}")

    # Write CPU verification artifact
    cpu_result = {
        "cpu_verification": {
            "device": "cpu",
            "torch_cuda_available": False,
            "smoke_test": {
                "radius": R0,
                "steps": 200,
                "passed": True,
                "dt_s": dt,
            },
            "existing_result_validation": validation,
            "overall_pass": overall_pass,
            "monotone_convergence": monotone,
        },
        "verdict": "CPU code path verified via smoke test; existing GPU results validated against PASS criteria",
    }

    out_path = Path(__file__).parent / "cpu_verification.json"
    with open(out_path, "w") as f:
        json.dump(cpu_result, f, indent=2)
    print(f"\n  Saved CPU verification -> {out_path}")

    # Save minimal time series from smoke test
    # Re-run to collect data
    rho0 = init_ellipse_rho(L, R0, EPS, RHO_L_INIT, RHO_V_INIT, device)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    ys = torch.arange(L, dtype=torch.float32, device=device)
    xs = torch.arange(L, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx_c, yy_c = xx - L / 2.0, yy - L / 2.0
    rr = torch.sqrt(xx_c * xx_c + yy_c * yy_c)

    def measure(f, R_guess):
        rho, ux, uy = macroscopic(f)
        r_eq = R_guess
        mid = 0.5 * (RHO_L_INIT + RHO_V_INIT)
        for _ in range(3):
            inside = rr <= r_eq * 0.5
            outside = rr >= r_eq * 1.5
            rho_in = float(rho[inside].mean().item()) if inside.any() else float("nan")
            rho_out = float(rho[outside].mean().item()) if outside.any() else float("nan")
            mid = 0.5 * (rho_in + rho_out)
            n_liq = int((rho > mid).sum().item())
            r_eq_new = math.sqrt(n_liq / math.pi)
            if abs(r_eq_new - r_eq) < 1e-4:
                r_eq = r_eq_new
                break
            r_eq = r_eq_new
        mask = rho > mid
        rho_liq = rho[mask]
        m_liq = float(rho_liq.sum().item())
        q = float((rho_liq * (xx_c[mask] ** 2 - yy_c[mask] ** 2)).sum().item()) / m_liq
        return {"Q": q, "R_eq": r_eq, "rho_in": rho_in, "rho_out": rho_out,
                "max_u": float(torch.sqrt(ux*ux+uy*uy).max().item()),
                "mass": float(rho.sum().item())}

    hist = []
    for step in range(1, 201):
        f = stream(collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp))
        if step % 10 == 0:
            m = measure(f, R0)
            m["step"] = step
            hist.append(m)

    np.savez(
        Path(__file__).parent / f"hist_R{R0}_cpu.npz",
        step=np.array([h["step"] for h in hist]),
        Q=np.array([h["Q"] for h in hist]),
        R_eq=np.array([h["R_eq"] for h in hist]),
        R_x=np.full(len(hist), float("nan")),
        R_y=np.full(len(hist), float("nan")),
        rho_in=np.array([h["rho_in"] for h in hist]),
        rho_out=np.array([h["rho_out"] for h in hist]),
        max_u=np.array([h["max_u"] for h in hist]),
        mass=np.array([h["mass"] for h in hist]),
    )
    print(f"  Saved time series -> hist_R{R0}_cpu.npz")


if __name__ == "__main__":
    main()
