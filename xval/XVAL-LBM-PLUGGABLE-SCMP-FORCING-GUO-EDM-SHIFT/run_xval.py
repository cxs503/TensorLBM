#!/usr/bin/env python
"""Historical replay using the frozen solver snapshot.

XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT — Round 2 cross-benchmark.

Cross-validate TensorLBM's pluggable SCMP forcing schemes (velocity_shift,
EDM, Guo) against a clean reference implementation derived from the
published literature (zero hand-copy: formulas transcribed directly from
papers).

OSS / literature reference formulas:
  - Velocity shift:  u_eq = u + τ·F/ρ  (Shan & Chen 1993)
  - EDM:  Δf = feq(ρ, u + F/ρ) − feq(ρ, u)  (Kupershtokh 2004)
  - Guo:  S_i = w_i(1−1/(2τ))[(c_i−u*)·F/cs² + (c_i·u*)(c_i·F)/cs⁴]
          with u* = u + F/(2ρ)  (Guo, Zheng & Shi 2002, PRE 65, 046308)

Benchmark: static droplet Laplace law test (2-D D2Q9, periodic domain).
  ΔP = σ / R_eff  →  σ = slope of ΔP vs 1/R_eff

Criterion (owner): benchmark error < 15 %; cross-arm < 3 %.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo import — use the work directory directly
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2] / "development/legacy/TensorLBM"
sys.path.insert(0, str(_REPO / "src"))

from tensorlbm.d2q9 import C, W, equilibrium, macroscopic, stream  # noqa: E402
from tensorlbm.multiphase import (  # noqa: E402
    collide_sc_single_component,
    sc_single_component_force,
    collide_velocity_shift,
    collide_edm,
    collide_guo,
    psi_exp,
    sc_force,
    stream_periodic,
)
from tensorlbm.multiphase.laplace import (  # noqa: E402
    sc_pressure,
    interface_radius,
    laplace_pressure_calibration,
    fit_laplace,
)

# Verify we're using the right repo
import tensorlbm
assert tensorlbm.__file__.startswith(str(_REPO / "src")), (
    f"Wrong repo: {tensorlbm.__file__} (expected {_REPO / 'src'})"
)

# ---------------------------------------------------------------------------
# Lattice constants
# ---------------------------------------------------------------------------
CS2 = 1.0 / 3.0
C_ARR = C.numpy()  # (9, 2)
W_ARR = W.numpy()  # (9,)

# ---------------------------------------------------------------------------
# Reference (OSS-style) implementations — formulas from papers, zero hand-copy
# ---------------------------------------------------------------------------

def _feq_2d(rho, ux, uy):
    """D2Q9 equilibrium, numpy, shape (9, ny, nx)."""
    cu = C_ARR[:, 0][:, np.newaxis, np.newaxis] * ux[np.newaxis] + C_ARR[:, 1][:, np.newaxis, np.newaxis] * uy[np.newaxis]
    u2 = ux**2 + uy**2
    return W_ARR[:, np.newaxis, np.newaxis] * rho[np.newaxis] * (
        1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2[np.newaxis]
    )


def _macroscopic_2d(f):
    """Return (rho, ux, uy) from f of shape (9, ny, nx)."""
    rho = f.sum(axis=0)
    rho_s = np.clip(rho, 1e-12, None)
    ux = (C_ARR[:, 0] @ f) / rho_s
    uy = (C_ARR[:, 1] @ f) / rho_s
    return rho, ux, uy


def _sc_force_2d(rho, G, psi_fn, gx=0.0, gy=0.0):
    """Standard SC self-interaction force (backward-gather convention).

    F = G·cs²·ψ(x)·Σ_i w_i c_i ψ(x−c_i) + ρ·g

    With G > 0 and backward gather this gives attraction (library convention).
    """
    ny, nx = rho.shape
    psi = psi_fn(rho)
    Fx = np.zeros_like(rho)
    Fy = np.zeros_like(rho)
    for i in range(9):
        cx, cy = int(C_ARR[i, 0]), int(C_ARR[i, 1])
        psi_nb = np.roll(np.roll(psi, -cy, axis=0), -cx, axis=1)
        Fx += W_ARR[i] * cx * psi_nb
        Fy += W_ARR[i] * cy * psi_nb
    Fx = G * CS2 * psi * Fx + rho * gx
    Fy = G * CS2 * psi * Fy + rho * gy
    return Fx, Fy


def collide_ref_velocity_shift(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """Reference velocity-shift SCMP collision (Shan & Chen 1993)."""
    rho, ux, uy = _macroscopic_2d(f)
    Fx, Fy = _sc_force_2d(rho, G, psi_fn, gx, gy)
    rho_s = np.clip(rho, 1e-12, None)
    feq = _feq_2d(rho, ux + tau * Fx / rho_s, uy + tau * Fy / rho_s)
    return f - (f - feq) / tau


def collide_ref_edm(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """Reference EDM collision (Kupershtokh 2004)."""
    rho, ux, uy = _macroscopic_2d(f)
    Fx, Fy = _sc_force_2d(rho, G, psi_fn, gx, gy)
    rho_s = np.clip(rho, 1e-12, None)
    feq_u = _feq_2d(rho, ux, uy)
    feq_d = _feq_2d(rho, ux + Fx / rho_s, uy + Fy / rho_s)
    return f - (f - feq_u) / tau + (feq_d - feq_u)


def collide_ref_guo(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """Reference Guo (2002) second-order forcing collision.

    S_i = w_i (1 − 1/(2τ)) [ (c_i − u*)·F / cs² + (c_i·u*)(c_i·F) / cs⁴ ]
    with u* = u + F/(2ρ).
    """
    rho, ux, uy = _macroscopic_2d(f)
    Fx, Fy = _sc_force_2d(rho, G, psi_fn, gx, gy)
    rho_s = np.clip(rho, 1e-12, None)

    # Physical velocity with half-force correction
    uxp = ux + 0.5 * Fx / rho_s
    uyp = uy + 0.5 * Fy / rho_s

    feq = _feq_2d(rho, uxp, uyp)

    # Guo source term
    cu = C_ARR[:, 0][:, np.newaxis, np.newaxis] * uxp[np.newaxis] + C_ARR[:, 1][:, np.newaxis, np.newaxis] * uyp[np.newaxis]  # (9, ny, nx)
    cf = C_ARR[:, 0][:, np.newaxis, np.newaxis] * Fx[np.newaxis] + C_ARR[:, 1][:, np.newaxis, np.newaxis] * Fy[np.newaxis]    # (9, ny, nx)
    uf = uxp * Fx + uyp * Fy  # (ny, nx)

    pref = 1.0 - 0.5 / tau
    S = W_ARR[:, np.newaxis, np.newaxis] * pref * (
        (cf - uf[np.newaxis]) / CS2 + cu * cf / (CS2 * CS2)
    )
    return f - (f - feq) / tau + S


# ---------------------------------------------------------------------------
# TensorLBM wrappers (torch)
# ---------------------------------------------------------------------------

def collide_tl_velocity_shift(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """TensorLBM velocity-shift SCMP collision."""
    return collide_sc_single_component(
        f, G=G, tau=tau, psi_fn=psi_fn, gx=gx, gy=gy,
        forcing="velocity_shift", scheme="standard"
    )


def collide_tl_edm(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """TensorLBM EDM SCMP collision."""
    return collide_sc_single_component(
        f, G=G, tau=tau, psi_fn=psi_fn, gx=gx, gy=gy,
        forcing="edm", scheme="standard"
    )


def collide_tl_guo(f, G, tau, psi_fn, gx=0.0, gy=0.0):
    """TensorLBM Guo SCMP collision."""
    return collide_sc_single_component(
        f, G=G, tau=tau, psi_fn=psi_fn, gx=gx, gy=gy,
        forcing="guo", scheme="standard"
    )


# ---------------------------------------------------------------------------
# Droplet initialization
# ---------------------------------------------------------------------------

def init_droplet(L, R, rho_l, rho_v, width=4.0, dtype=np.float64):
    """Initialize a circular droplet with tanh interface."""
    ys = np.arange(L, dtype=dtype)
    xs = np.arange(L, dtype=dtype)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    r = np.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + np.tanh((R - r) / width))
    zero = np.zeros_like(rho)
    return _feq_2d(rho, zero, zero)


def init_droplet_torch(L, R, rho_l, rho_v, device, width=4.0):
    """Initialize a circular droplet with tanh interface (torch)."""
    ys = torch.arange(L, dtype=torch.float64, device=device)
    xs = torch.arange(L, dtype=torch.float64, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    rho = rho_v + 0.5 * (rho_l - rho_v) * (1.0 + torch.tanh((R - r) / width))
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def stream_numpy(f):
    """AA-pattern streaming for numpy (9, ny, nx)."""
    f_new = np.empty_like(f)
    for i in range(9):
        cx, cy = int(C_ARR[i, 0]), int(C_ARR[i, 1])
        # For velocity (cx, cy), shift by (-cy, -cx) in (row, col)
        f_new[i] = np.roll(np.roll(f[i], -cy, axis=0), -cx, axis=1)
    return f_new


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_droplet(collide_fn, L, R, rho_l, rho_v, G, tau, psi_fn,
                max_steps=8000, sample=200, min_steps=2000, conv_tol=1e-5,
                device=None, is_torch=False):
    """Run a static droplet simulation to steady state."""
    if is_torch:
        f = init_droplet_torch(L, R, rho_l, rho_v, device)
        mass0 = float(f.sum().item())
        ys = torch.arange(L, dtype=torch.float64, device=device)
        yy, xx = torch.meshgrid(ys, ys, indexing="ij")
        r_grid = torch.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)
    else:
        f = init_droplet(L, R, rho_l, rho_v)
        mass0 = float(f.sum())
        ys = np.arange(L)
        yy, xx = np.meshgrid(ys, ys, indexing="ij")
        r_grid = np.sqrt((xx - L / 2.0) ** 2 + (yy - L / 2.0) ** 2)

    core_mask = r_grid <= 0.25 * L / 2
    far_mask = r_grid >= 0.80 * L / 2

    hist = []
    stable = True
    err_msg = None
    step_i = 0

    for step_i in range(1, max_steps + 1):
        if is_torch:
            f = collide_fn(f)
            f = stream(f)
        else:
            f = stream_numpy(collide_fn(f))

        if step_i % sample != 0:
            continue

        if is_torch:
            rho = f.sum(dim=0).cpu().numpy()
        else:
            rho = f.sum(axis=0)

        if not np.isfinite(rho).all() or rho.min() < 0:
            stable = False
            err_msg = f"nonfinite/negative rho at step {step_i}"
            break

        rho_l_cur = float(rho[core_mask].mean())
        rho_v_cur = float(rho[far_mask].mean())

        # Iterative R_eff refinement
        r_eff = 0.0
        for _ in range(2):
            r_eff = float(np.sqrt(
                max((rho - rho_v_cur).sum(), 0.0) /
                (math.pi * max(rho_l_cur - rho_v_cur, 1e-12))
            ))
            if r_eff <= 0:
                break
            core_mask = r_grid <= 0.5 * r_eff
            rho_l_cur = float(rho[core_mask].mean())
            rho_v_cur = float(rho[far_mask].mean())

        if r_eff <= 0:
            stable = False
            err_msg = f"degenerate droplet at step {step_i}"
            break

        # Mechanical pressure: p_mech = ρ·cs² − (G/2)·cs²·ψ²
        psi_val = psi_fn(torch.tensor(rho) if is_torch else rho)
        if is_torch:
            psi_val = psi_val.cpu().numpy()
        p_mech = rho * CS2 - 0.5 * G * CS2 * psi_val**2
        dp = float(p_mech[core_mask].mean() - p_mech[far_mask].mean())

        # Spurious velocity
        if is_torch:
            mom_x = (f * C.to(device)[:, 0].view(9, 1, 1)).sum(dim=0).cpu().numpy()
            mom_y = (f * C.to(device)[:, 1].view(9, 1, 1)).sum(dim=0).cpu().numpy()
        else:
            mom_x = C_ARR[0] @ f
            mom_y = C_ARR[1] @ f
        rho_s = np.clip(rho, 1e-12, None)
        u_max = float(np.sqrt((mom_x / rho_s)**2 + (mom_y / rho_s)**2).max())

        mass = float(rho.sum())
        mass_drift = abs(mass - mass0) / mass0

        hist.append(dict(
            step=step_i, rho_l=rho_l_cur, rho_v=rho_v_cur,
            dp=dp, r_eff=r_eff, u_max=u_max, mass=mass, mass_drift=mass_drift
        ))

        # Convergence check
        if len(hist) >= 4 and step_i >= min_steps:
            rec = hist[-3:]
            if all(
                abs(h["rho_l"] - hp["rho_l"]) / max(abs(h["rho_l"]), 1e-12) <= conv_tol
                and abs(h["rho_v"] - hp["rho_v"]) / max(abs(h["rho_v"]), 1e-12) <= conv_tol
                for h, hp in zip(rec[1:], rec[:-1])
            ):
                break

    out = dict(
        stable=stable, error=err_msg, steps_run=step_i,
        converged=len(hist) >= 4 and step_i < max_steps and stable,
        hist=hist,
    )
    if hist:
        tail = hist[-5:]
        for k in ("rho_l", "rho_v", "dp", "r_eff"):
            m = sum(t[k] for t in tail) / len(tail)
            out[k] = m
            out[k + "_std"] = (sum((t[k] - m)**2 for t in tail) / len(tail))**0.5
        out["ratio"] = out["rho_l"] / out["rho_v"]
        out["u_max"] = max(t["u_max"] for t in tail)
        out["mass_drift"] = max(t["mass_drift"] for t in tail)
        out["mass_rate_per_step"] = out["mass_drift"] / out["steps_run"]
    return out


# ---------------------------------------------------------------------------
# Sigma extraction from Laplace fit
# ---------------------------------------------------------------------------

def extract_sigma(results_list):
    """Extract surface tension σ from a list of droplet results via Laplace fit."""
    dps = [r["dp"] for r in results_list if r["stable"] and r.get("r_eff", 0) > 0]
    radii = [r["r_eff"] for r in results_list if r["stable"] and r.get("r_eff", 0) > 0]
    if len(dps) < 2:
        return None, None, None
    fit = fit_laplace(dps, radii)
    return fit["sigma"], fit["r2"], fit["max_rel_err"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cpu")
    L = 128
    G = 5.0
    tau = 1.0
    rho_l_init = 4.5
    rho_v_init = 0.37
    radii = [12, 16, 20, 24]
    psi_fn = psi_exp

    # Reference collide functions (numpy)
    ref_collide = {
        "velocity_shift": lambda f: collide_ref_velocity_shift(f, G, tau, psi_fn),
        "edm": lambda f: collide_ref_edm(f, G, tau, psi_fn),
        "guo": lambda f: collide_ref_guo(f, G, tau, psi_fn),
    }

    # TensorLBM collide functions (torch)
    tl_collide = {
        "velocity_shift": lambda f: collide_tl_velocity_shift(f, G, tau, psi_fn),
        "edm": lambda f: collide_tl_edm(f, G, tau, psi_fn),
        "guo": lambda f: collide_tl_guo(f, G, tau, psi_fn),
    }

    results = {"ref": {}, "tl": {}}
    all_results = []

    print("=" * 70)
    print("XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT — Round 2")
    print(f"L={L}, G={G}, tau={tau}, radii={radii}")
    print("=" * 70)

    for arm_name, collide_dict, is_torch in [
        ("ref", ref_collide, False),
        ("tl", tl_collide, True),
    ]:
        for scheme_name, collide_fn in collide_dict.items():
            print(f"\n--- {arm_name.upper()} / {scheme_name} ---")
            scheme_results = []
            for R in radii:
                t0 = time.time()
                r = run_droplet(
                    collide_fn, L, R, rho_l_init, rho_v_init, G, tau, psi_fn,
                    device=device, is_torch=is_torch
                )
                elapsed = time.time() - t0
                status = "OK" if r["stable"] else "FAIL"
                rl = f"{r.get('rho_l', float('nan')):.4f}" if r["stable"] else "nan"
                rv = f"{r.get('rho_v', float('nan')):.4f}" if r["stable"] else "nan"
                dp = f"{r.get('dp', float('nan')):.6f}" if r["stable"] else "nan"
                reff = f"{r.get('r_eff', float('nan')):.2f}" if r["stable"] else "nan"
                umax = f"{r.get('u_max', float('nan')):.2e}" if r["stable"] else "nan"
                print(f"  R={R}: {status} rl={rl} rv={rv} dp={dp} R_eff={reff} "
                      f"u_max={umax} steps={r['steps_run']} t={elapsed:.1f}s")
                scheme_results.append(r)

            sigma, r2, max_err = extract_sigma(scheme_results)
            sigma_str = f"{sigma:.6f}" if sigma is not None else "N/A"
            r2_str = f"{r2:.4f}" if r2 is not None else "N/A"
            print(f"  σ={sigma_str}  R²={r2_str}")

            results[arm_name][scheme_name] = {
                "sigma": sigma,
                "r2": r2,
                "max_rel_err": max_err,
                "n": len([r for r in scheme_results if r["stable"]]),
                "runs": scheme_results,
            }

    # -----------------------------------------------------------------------
    # Comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPARISON: TensorLBM vs Reference")
    print("=" * 70)

    comparison = {}
    all_pass_15 = True
    all_pass_3 = True

    for scheme in ["velocity_shift", "edm", "guo"]:
        ref_sigma = results["ref"][scheme]["sigma"]
        tl_sigma = results["tl"][scheme]["sigma"]
        if ref_sigma is not None and tl_sigma is not None and ref_sigma != 0:
            err_pct = abs(tl_sigma - ref_sigma) / abs(ref_sigma) * 100
            pass_15 = err_pct < 15
            pass_3 = err_pct < 3
            if not pass_15:
                all_pass_15 = False
            if not pass_3:
                all_pass_3 = False
            status = "PASS" if pass_15 else "FAIL"
            print(f"  {scheme}: σ_ref={ref_sigma:.6f} σ_tl={tl_sigma:.6f} "
                  f"err={err_pct:.2f}% [{status}]")
            comparison[scheme] = {
                "sigma_ref": ref_sigma,
                "sigma_tl": tl_sigma,
                "error_pct": err_pct,
                "pass_15": pass_15,
                "pass_3": pass_3,
            }
        else:
            print(f"  {scheme}: insufficient data for comparison")
            comparison[scheme] = {
                "sigma_ref": ref_sigma,
                "sigma_tl": tl_sigma,
                "error_pct": None,
                "pass_15": False,
                "pass_3": False,
            }

    # -----------------------------------------------------------------------
    # Single-step verification
    # -----------------------------------------------------------------------
    print("\n--- Single-step verification (L=64, R=12) ---")
    L_ss = 64
    R_ss = 12
    f_ref = init_droplet(L_ss, R_ss, rho_l_init, rho_v_init)
    f_tl = init_droplet_torch(L_ss, R_ss, rho_l_init, rho_v_init, device)

    single_step = {}
    for scheme in ["velocity_shift", "edm", "guo"]:
        f_ref_1 = stream_numpy(ref_collide[scheme](f_ref))
        f_tl_1 = stream(tl_collide[scheme](f_tl))

        rho_ref = f_ref_1.sum(axis=0)
        rho_tl = f_tl_1.sum(dim=0).cpu().numpy()

        single_step[scheme] = {
            "ref_rho_min": float(rho_ref.min()),
            "ref_rho_max": float(rho_ref.max()),
            "ref_mass": float(rho_ref.sum()),
            "tl_rho_min": float(rho_tl.min()),
            "tl_rho_max": float(rho_tl.max()),
            "tl_mass": float(rho_tl.sum()),
            "rho_max_diff": float(abs(rho_ref.max() - rho_tl.max())),
            "mass_diff": float(abs(rho_ref.sum() - rho_tl.sum())),
        }
        print(f"  {scheme}: ref ρ=[{single_step[scheme]['ref_rho_min']:.4f}, "
              f"{single_step[scheme]['ref_rho_max']:.4f}] mass={single_step[scheme]['ref_mass']:.2f}  "
              f"tl ρ=[{single_step[scheme]['tl_rho_min']:.4f}, "
              f"{single_step[scheme]['tl_rho_max']:.4f}] mass={single_step[scheme]['tl_mass']:.2f}  "
              f"Δmass={single_step[scheme]['mass_diff']:.6f}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    ad = Path(__file__).parent / "artifacts"
    ad.mkdir(parents=True, exist_ok=True)

    # Compact results for JSON
    compact_results = {
        "anchor": "XVAL-LBM-PLUGGABLE-SCMP-FORCING-GUO-EDM-SHIFT",
        "mission_id": 21238,
        "parameters": {
            "G": G, "tau": tau, "L": L, "radii": radii,
            "rho_l": rho_l_init, "rho_v": rho_v_init, "psi_fn": "psi_exp"
        },
        "sigma": {
            "ref": {
                s: {"sigma": results["ref"][s]["sigma"], "r2": results["ref"][s]["r2"],
                     "n": results["ref"][s]["n"]}
                for s in ["velocity_shift", "edm", "guo"]
            },
            "tl": {
                s: {"sigma": results["tl"][s]["sigma"], "r2": results["tl"][s]["r2"],
                     "n": results["tl"][s]["n"]}
                for s in ["velocity_shift", "edm", "guo"]
            },
        },
        "comparison": comparison,
        "verdict": {
            "all_pass_15": all_pass_15,
            "all_pass_3": all_pass_3,
        },
        "single_step_verification": single_step,
    }

    with open(ad / "xval_results.json", "w") as fp:
        json.dump(compact_results, fp, indent=2)

    # Also update the results.json in the xval directory
    with open(Path(__file__).parent / "results.json", "w") as fp:
        json.dump(compact_results, fp, indent=2)

    print(f"\nResults saved to {ad}/xval_results.json")

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  All <15%: {'PASS' if all_pass_15 else 'FAIL'}")
    print(f"  All <3%:  {'PASS' if all_pass_3 else 'FAIL'}")

    if all_pass_15:
        print("\n  TensorLBM pluggable SCMP forcing schemes PASS the 15% criterion.")
    else:
        print("\n  TensorLBM pluggable SCMP forcing schemes FAIL the 15% criterion.")

    return 0 if all_pass_15 else 1


if __name__ == "__main__":
    sys.exit(main())
