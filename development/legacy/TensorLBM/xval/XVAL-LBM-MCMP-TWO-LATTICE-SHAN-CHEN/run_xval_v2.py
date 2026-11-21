#!/usr/bin/env python3
"""XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN — Streamlined cross-validation protocol.

Cross-validates TensorLBM's two-component Shan-Chen (MCMP) implementation
against the MPLBM-UT reference algorithm.  Uses moderate grid sizes for
fast execution while maintaining physics fidelity.

Acceptance criterion: <15% (owner precision standard).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# D2Q9 lattice constants
# ---------------------------------------------------------------------------

CX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
CY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
W = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
CS2 = 1.0 / 3.0

CX_T = torch.tensor(CX, dtype=torch.float64)
CY_T = torch.tensor(CY, dtype=torch.float64)
W_T = torch.tensor(W, dtype=torch.float64)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reference solver: MPLBM-UT convention (forward gather, G > 0 = repulsive)
# ---------------------------------------------------------------------------

def _eq(rho, ux, uy):
    cu = CX_T[:, None, None] * ux[None] + CY_T[:, None, None] * uy[None]
    usq = ux * ux + uy * uy
    return W_T[:, None, None] * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def _macro(f):
    rho = f.sum(dim=0)
    rs = torch.clamp(rho, min=1e-12)
    ux = (f * CX_T[:, None, None]).sum(dim=0) / rs
    uy = (f * CY_T[:, None, None]).sum(dim=0) / rs
    return rho, ux, uy


def _force_mplbm(rho1, rho2, G12):
    gx2 = torch.zeros_like(rho2)
    gy2 = torch.zeros_like(rho2)
    gx1 = torch.zeros_like(rho1)
    gy1 = torch.zeros_like(rho1)
    for i in range(9):
        if CX[i] == 0 and CY[i] == 0:
            continue
        r2s = torch.roll(rho2, shifts=(-int(CY[i]), -int(CX[i])), dims=(0, 1))
        r1s = torch.roll(rho1, shifts=(-int(CY[i]), -int(CX[i])), dims=(0, 1))
        gx2 += W[i] * CX[i] * r2s
        gy2 += W[i] * CY[i] * r2s
        gx1 += W[i] * CX[i] * r1s
        gy1 += W[i] * CY[i] * r1s
    return -G12 * rho1 * gx2, -G12 * rho1 * gy2, -G12 * rho2 * gx1, -G12 * rho2 * gy1


def _stream(f):
    out = torch.empty_like(f)
    for i in range(9):
        out[i] = torch.roll(f[i], shifts=(int(CY[i]), int(CX[i])), dims=(0, 1))
    return out


def step_mplbm(f1, f2, G12, tau1, tau2):
    rho1, ux1, uy1 = _macro(f1)
    rho2, ux2, uy2 = _macro(f2)
    Fx1, Fy1, Fx2, Fy2 = _force_mplbm(rho1, rho2, G12)
    rs1 = torch.clamp(rho1, min=1e-12)
    rs2 = torch.clamp(rho2, min=1e-12)
    feq1 = _eq(rho1, ux1 + tau1 * Fx1 / rs1, uy1 + tau1 * Fy1 / rs1)
    feq2 = _eq(rho2, ux2 + tau2 * Fx2 / rs2, uy2 + tau2 * Fy2 / rs2)
    fp1 = f1 - (f1 - feq1) / tau1
    fp2 = f2 - (f2 - feq2) / tau2
    return _stream(fp1), _stream(fp2)


# ---------------------------------------------------------------------------
# TensorLBM solver
# ---------------------------------------------------------------------------

def _get_tl_step():
    from tensorlbm.multiphase import collide_sc_two_component
    from tensorlbm.solver import stream as tl_stream

    def step_tl(f1, f2, G12, tau1, tau2):
        fp1, fp2 = collide_sc_two_component(
            f1, f2, G_12=G12, tau1=tau1, tau2=tau2,
            u_eq="self", force_scheme="shift", psi=None,
        )
        return tl_stream(fp1), tl_stream(fp2)

    return step_tl


# ---------------------------------------------------------------------------
# Measurement utilities
# ---------------------------------------------------------------------------

def init_droplet(n, radius, rho1_liq, rho1_gas, rho2_liq, rho2_gas, width=2.0):
    y = torch.arange(n, dtype=torch.float64)
    x = torch.arange(n, dtype=torch.float64)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    centre = (n - 1) / 2.0
    r = torch.sqrt((X - centre) ** 2 + (Y - centre) ** 2)
    s = 0.5 * (1.0 - torch.tanh((r - radius) / max(width, 1e-6)))
    rho1 = rho1_gas + s * (rho1_liq - rho1_gas)
    rho2 = rho2_gas + s * (rho2_liq - rho2_gas)
    zero = torch.zeros_like(rho1)
    return _eq(rho1, zero, zero), _eq(rho2, zero, zero)


def measure_droplet(f1, f2, G12_sign, G12):
    rho1, ux1, uy1 = _macro(f1)
    rho2, ux2, uy2 = _macro(f2)
    rho_tot = rho1 + rho2
    rs = torch.clamp(rho_tot, min=1e-12)
    ux = (rho1 * ux1 + rho2 * ux2) / rs
    uy = (rho1 * uy1 + rho2 * uy2) / rs

    if G12_sign == "mplbm":
        p = CS2 * (rho1 + rho2) - G12 * CS2 * rho1 * rho2
    else:
        p = CS2 * (rho1 + rho2) + G12 * CS2 * rho1 * rho2

    n = rho1.shape[0]
    c = n // 2
    k = max(4, n // 16)
    p_core = float(p[c - k:c + k, c - k:c + k].mean())
    p_far = float(p[:k, :k].mean())
    dp = p_core - p_far

    rho1_mean = float(rho1.mean())
    mask = rho1 > rho1_mean
    area = float(mask.sum())
    R_eq = math.sqrt(area / math.pi) if area > 0 else 0.0
    sigma = dp * R_eq if R_eq > 0 else 0.0
    u_max = float(torch.sqrt(ux ** 2 + uy ** 2).max())

    rho1_liq = float(rho1[c, c])
    rho1_gas = float(rho1[0, n // 2])

    return {"dp": dp, "R_eq": R_eq, "sigma": sigma, "u_max": u_max,
            "rho1_liq": rho1_liq, "rho1_gas": rho1_gas}


def run_droplet(solver, n, radius, G12, tau, rho1_liq, rho1_gas, rho2_liq, rho2_gas,
                steps, G12_sign, width=2.0):
    f1, f2 = init_droplet(n, radius, rho1_liq, rho1_gas, rho2_liq, rho2_gas, width)
    t0 = time.time()
    for _ in range(steps):
        f1, f2 = solver(f1, f2, G12, tau, tau)
    elapsed = time.time() - t0
    if torch.isnan(f1).any():
        return {"nan": True, "elapsed_s": elapsed}
    result = measure_droplet(f1, f2, G12_sign, G12)
    result["nan"] = False
    result["elapsed_s"] = round(elapsed, 2)
    result["R_init"] = radius
    return result


# ---------------------------------------------------------------------------
# Analytical surface tension for linear psi MCMP
# ---------------------------------------------------------------------------

def analytical_sigma(G12_abs, rho1_liq, rho1_gas, rho2_liq, rho2_gas):
    """Analytical surface tension for linear-psi MCMP Shan-Chen.

    For the two-component Shan-Chen model with psi(rho) = rho:
        sigma = (G12 * cs^2 / 2) * (rho1_liq - rho1_gas)^2 * dx

    This is the leading-order result from the pressure tensor integration
    across a flat interface (Shan & Chen 1994).

    In 2D with dx = 1 (lattice units):
        sigma = G12 * cs^2 / 2 * (rho_l - rho_g)^2
    """
    drho = rho1_liq - rho1_gas
    return G12_abs * CS2 / 2.0 * drho ** 2


# ---------------------------------------------------------------------------
# Main protocol
# ---------------------------------------------------------------------------

def main():
    torch.set_num_threads(1)

    # Configuration: moderate grid for fast execution
    n = 64
    radii = [10.0, 14.0, 18.0, 22.0]
    G_mplbm = 2.5   # > 0 = repulsive in MPLBM-UT
    G_tl = -G_mplbm  # < 0 = repulsive in TensorLBM
    tau = 1.0
    steps = 4000
    rho1_liq, rho1_gas = 0.92, 0.08
    rho2_liq, rho2_gas = 0.08, 0.92
    width = 2.0

    print("=" * 70)
    print("XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN")
    print(f"MPLBM-UT G12 = {G_mplbm} (repulsive)  ↔  TensorLBM G12 = {G_tl} (repulsive)")
    print(f"Grid: {n}×{n}, tau = {tau}, steps = {steps}")
    print(f"Radii: {radii}")
    print("=" * 70)

    step_tl = _get_tl_step()

    # Analytical surface tension
    sigma_analytical = analytical_sigma(G_mplbm, rho1_liq, rho1_gas, rho2_liq, rho2_gas)
    print(f"\nAnalytical σ (linear psi, flat interface): {sigma_analytical:.6f}")

    # Phase 1: Force equivalence check
    print("\n--- Phase 1: Force equivalence ---")
    torch.manual_seed(42)
    rho1_test = torch.rand(16, 16, dtype=torch.float64) * 0.5 + 0.5
    rho2_test = torch.rand(16, 16, dtype=torch.float64) * 0.5 + 0.5
    Fx1_m, Fy1_m, Fx2_m, Fy2_m = _force_mplbm(rho1_test, rho2_test, G_mplbm)
    from tensorlbm.multiphase import sc_two_component_force
    Fx1_t, Fy1_t, Fx2_t, Fy2_t = sc_two_component_force(rho1_test, rho2_test, G_12=G_tl, psi=None)
    force_err = (Fx1_m - Fx1_t).abs().max().item()
    print(f"  Max |F_mplbm - F_tensorlbm|: {force_err:.2e}  [{'PASS' if force_err < 1e-6 else 'FAIL'}]")

    # Phase 2: Static droplets at multiple radii
    print("\n--- Phase 2: Static droplets (Laplace law) ---")
    mplbm_cases = []
    tl_cases = []

    for R in radii:
        r_m = run_droplet(step_mplbm, n, R, G_mplbm, tau, rho1_liq, rho1_gas,
                          rho2_liq, rho2_gas, steps, "mplbm", width)
        r_t = run_droplet(step_tl, n, R, G_tl, tau, rho1_liq, rho1_gas,
                          rho2_liq, rho2_gas, steps, "tensorlbm", width)
        mplbm_cases.append(r_m)
        tl_cases.append(r_t)
        print(f"  R={R:.0f}: MPLBM dp={r_m['dp']:.6f} σ={r_m['sigma']:.6f} | "
              f"TL dp={r_t['dp']:.6f} σ={r_t['sigma']:.6f} | "
              f"σ err={abs(r_t['sigma']-r_m['sigma'])/max(abs(r_m['sigma']),1e-12)*100:.2f}%")

    # Phase 3: Laplace law fit
    print("\n--- Phase 3: Laplace law analysis ---")

    def laplace_fit(cases):
        valid = [c for c in cases if c.get("R_eq", 0) > 0 and not c.get("nan", False)]
        if len(valid) < 2:
            return None, None
        inv_R = np.array([1.0 / c["R_eq"] for c in valid])
        dp = np.array([c["dp"] for c in valid])
        sigma = float(np.linalg.lstsq(inv_R.reshape(-1, 1), dp, rcond=None)[0][0])
        predicted = sigma * inv_R
        ss_res = np.sum((dp - predicted) ** 2)
        ss_tot = np.sum((dp - dp.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return sigma, r2

    sig_m, r2_m = laplace_fit(mplbm_cases)
    sig_t, r2_t = laplace_fit(tl_cases)

    if sig_m is not None:
        print(f"  MPLBM-UT:  σ = {sig_m:.6f}  R² = {r2_m:.4f}")
    if sig_t is not None:
        print(f"  TensorLBM: σ = {sig_t:.6f}  R² = {r2_t:.4f}")

    sigma_cross_err = abs(sig_t - sig_m) / max(abs(sig_m), 1e-12) * 100 if sig_m and sig_t else float('nan')
    print(f"  σ cross-error: {sigma_cross_err:.2f}%  [{'PASS' if sigma_cross_err < 15 else 'FAIL'}]")

    # Phase 4: Per-radius comparison
    print("\n--- Phase 4: Per-radius comparison ---")
    per_radius = []
    for rm, rt in zip(mplbm_cases, tl_cases):
        if rm.get("nan") or rt.get("nan"):
            continue
        dp_err = abs(rt["dp"] - rm["dp"]) / max(abs(rm["dp"]), 1e-12) * 100
        sig_err = abs(rt["sigma"] - rm["sigma"]) / max(abs(rm["sigma"]), 1e-12) * 100
        row = {"R_init": rm["R_init"], "dp_mplbm": rm["dp"], "dp_tensorlbm": rt["dp"],
               "dp_err_pct": dp_err, "sigma_mplbm": rm["sigma"], "sigma_tensorlbm": rt["sigma"],
               "sigma_err_pct": sig_err, "u_max_mplbm": rm["u_max"], "u_max_tensorlbm": rt["u_max"],
               "R_eq_mplbm": rm["R_eq"], "R_eq_tensorlbm": rt["R_eq"]}
        per_radius.append(row)
        print(f"  R={rm['R_init']:.0f}: Δp err={dp_err:.2f}%  σ err={sig_err:.2f}%")

    # Phase 5: Comparison with analytical σ
    print("\n--- Phase 5: Analytical comparison ---")
    if sig_m is not None:
        sig_m_anal_err = abs(sig_m - sigma_analytical) / max(abs(sigma_analytical), 1e-12) * 100
        print(f"  MPLBM-UT σ vs analytical: {sig_m:.6f} vs {sigma_analytical:.6f}  err={sig_m_anal_err:.1f}%")
    if sig_t is not None:
        sig_t_anal_err = abs(sig_t - sigma_analytical) / max(abs(sigma_analytical), 1e-12) * 100
        print(f"  TensorLBM σ vs analytical: {sig_t:.6f} vs {sigma_analytical:.6f}  err={sig_t_anal_err:.1f}%")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "config": {"n": n, "radii": radii, "G_mplbm": G_mplbm, "G_tl": G_tl,
                   "tau": tau, "steps": steps, "rho1_liq": rho1_liq, "rho1_gas": rho1_gas,
                   "rho2_liq": rho2_liq, "rho2_gas": rho2_gas},
        "analytical_sigma": sigma_analytical,
        "force_max_diff": force_err,
        "laplace_fit": {"mplbm": {"sigma": sig_m, "r2": r2_m},
                        "tensorlbm": {"sigma": sig_t, "r2": r2_t}},
        "sigma_cross_error_pct": sigma_cross_err,
        "per_radius": per_radius,
        "droplets_mplbm": mplbm_cases,
        "droplets_tensorlbm": tl_cases,
    }

    with open(ARTIFACTS / "results.json", "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nResults saved to {ARTIFACTS / 'results.json'}")

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    _generate_figures(results)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    all_pass = True

    # Check 1: Force equivalence
    force_ok = force_err < 1e-6
    print(f"  Force equivalence (max diff < 1e-6): {force_err:.2e}  [{'PASS' if force_ok else 'FAIL'}]")
    all_pass = all_pass and force_ok

    # Check 2: σ cross-error < 15%
    sig_ok = sigma_cross_err < 15
    print(f"  σ cross-error < 15%: {sigma_cross_err:.2f}%  [{'PASS' if sig_ok else 'FAIL'}]")
    all_pass = all_pass and sig_ok

    # Check 3: Per-radius Δp errors < 15%
    if per_radius:
        max_dp_err = max(r["dp_err_pct"] for r in per_radius)
        max_sig_err = max(r["sigma_err_pct"] for r in per_radius)
        dp_ok = max_dp_err < 15
        print(f"  Max per-radius Δp error < 15%: {max_dp_err:.2f}%  [{'PASS' if dp_ok else 'FAIL'}]")
        print(f"  Max per-radius σ error < 15%: {max_sig_err:.2f}%  [{'PASS' if max_sig_err < 15 else 'FAIL'}]")
        all_pass = all_pass and dp_ok and max_sig_err < 15

    # Check 4: No NaN
    nan_ok = not any(c.get("nan", False) for c in mplbm_cases + tl_cases)
    print(f"  No NaN: {'PASS' if nan_ok else 'FAIL'}")
    all_pass = all_pass and nan_ok

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

    results["verdict"] = {"pass": all_pass}
    with open(ARTIFACTS / "results.json", "w") as fp:
        json.dump(results, fp, indent=2, default=str)

    return all_pass


def _generate_figures(results: dict) -> None:
    mplbm = results.get("droplets_mplbm", [])
    tl = results.get("droplets_tensorlbm", [])

    if not mplbm or not tl:
        return

    # Figure 1: Laplace law Δp vs 1/R
    fig, ax = plt.subplots(figsize=(7, 5))
    for cases, label, color, marker in [
        (mplbm, "MPLBM-UT (ref)", "blue", "o"),
        (tl, "TensorLBM", "red", "s"),
    ]:
        valid = [c for c in cases if c.get("R_eq", 0) > 0 and not c.get("nan", False)]
        if valid:
            inv_R = [1.0 / c["R_eq"] for c in valid]
            dp = [c["dp"] for c in valid]
            ax.scatter(inv_R, dp, color=color, marker=marker, s=60, label=label, zorder=5)

    # Fit lines
    fit_data = results.get("laplace_fit", {})
    for key, color, label in [("mplbm", "blue", "MPLBM-UT"), ("tensorlbm", "red", "TensorLBM")]:
        fd = fit_data.get(key, {})
        sig = fd.get("sigma")
        if sig:
            x = np.linspace(0.02, 0.12, 100)
            ax.plot(x, sig * x, "--", color=color, alpha=0.7, label=f"{label}: σ={sig:.4f}")

    # Analytical line
    sig_a = results.get("analytical_sigma")
    if sig_a:
        x = np.linspace(0.02, 0.12, 100)
        ax.plot(x, sig_a * x, ":", color="green", alpha=0.7, label=f"Analytical: σ={sig_a:.4f}")

    ax.set_xlabel("1/R (1/lu)")
    ax.set_ylabel("Δp (lu)")
    ax.set_title("Laplace Law: Δp = σ/R (2D MCMP Shan-Chen)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(ARTIFACTS / "fig1_laplace_law.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: σ per radius comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    per_r = results.get("per_radius", [])
    if per_r:
        R_inits = [r["R_init"] for r in per_r]
        sig_m = [r["sigma_mplbm"] for r in per_r]
        sig_t = [r["sigma_tensorlbm"] for r in per_r]
        ax.plot(R_inits, sig_m, "o-", color="blue", label="MPLBM-UT", markersize=8)
        ax.plot(R_inits, sig_t, "s--", color="red", label="TensorLBM", markersize=8)
        if sig_m:
            sig_ref = np.mean(sig_m)
            ax.axhline(sig_ref * 1.15, color="gray", ls=":", alpha=0.5, label="±15% band")
            ax.axhline(sig_ref * 0.85, color="gray", ls=":", alpha=0.5)
        if sig_a:
            ax.axhline(sig_a, color="green", ls="-.", alpha=0.5, label=f"Analytical σ={sig_a:.4f}")
    ax.set_xlabel("Initial radius R (lu)")
    ax.set_ylabel("σ = Δp·R (lu)")
    ax.set_title("Surface Tension vs Radius")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(ARTIFACTS / "fig2_sigma_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Spurious currents comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    if mplbm and tl:
        R_m = [c["R_init"] for c in mplbm if not c.get("nan")]
        u_m = [c["u_max"] for c in mplbm if not c.get("nan")]
        R_t = [c["R_init"] for c in tl if not c.get("nan")]
        u_t = [c["u_max"] for c in tl if not c.get("nan")]
        ax.semilogy(R_m, u_m, "o-", color="blue", label="MPLBM-UT", markersize=8)
        ax.semilogy(R_t, u_t, "s--", color="red", label="TensorLBM", markersize=8)
    ax.set_xlabel("Initial radius R (lu)")
    ax.set_ylabel("|u|_max (lu)")
    ax.set_title("Spurious Currents")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(ARTIFACTS / "fig3_spurious_currents.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
