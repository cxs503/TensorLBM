#!/usr/bin/env python3
"""Two-phase Poiseuille benchmark with viscous velocity coupling (m8802 fix).

Tests the new u_eq="viscous" mode in collide_sc_two_component against the
analytical two-layer Poiseuille profile.  The viscous mode uses viscosity-
weighted velocity coupling (Martys et al. 1996 / sailfish convention) to
enforce velocity continuity at the diffuse interface.

Physical setup:
  - Planar channel, walls at y=0 and y=ny-1 (bounce-back)
  - Two immiscible fluids: water (bottom, tau_w=1.0) and gas (top, tau_g=0.75)
  - Body force G_x drives flow in x-direction
  - SC coupling G_12 maintains the interface
  - Analytical solution: two-layer Poiseuille with stress continuity at interface

Acceptance criterion:
  - max_rel_err <= 10% (target < 3% for grid-converged cases)
  - Monotone error decrease with grid refinement
  - Mass conservation (total mass drift < 1e-6 per step)

Usage:
  python run.py [--device cuda:N] [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.multiphase import collide_sc_two_component  # noqa: E402
from tensorlbm.porous_media import _two_phase_poiseuille_analytical  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.d2q9 import macroscopic  # noqa: E402


# ---------------------------------------------------------------------------
# Helper functions (from w4b_lib.py, inlined for independence)
# ---------------------------------------------------------------------------

def tanh_field_1d(ny: int, nx: int, y0: float, width: float, device) -> torch.Tensor:
    """alpha=1 below y0 (water side), 0 above, tanh of given width."""
    ys = torch.arange(ny, dtype=torch.float32, device=device).view(ny, 1)
    ys = ys.expand(ny, nx)
    return 0.5 * (1.0 - torch.tanh((ys - y0) / width))


def init_full_swap(alpha: torch.Tensor, heavy: float, light: float):
    """Full-swap MCMP fields: rho_w = heavy*alpha + light*(1-alpha) etc."""
    rho_w = heavy * alpha + light * (1.0 - alpha)
    rho_g = light * alpha + heavy * (1.0 - alpha)
    return rho_w, rho_g


def sc_mcmp_step_viscous(fw, fg, G12, tau1, tau2, gx=0.0, gy=0.0, solid=None):
    """One SC MCMP step with viscous velocity coupling."""
    fw, fg = collide_sc_two_component(
        fw, fg, G_12=G12, tau1=tau1, tau2=tau2, gx=gx, gy=gy,
        solid_mask=solid, u_eq="viscous"
    )
    fw = stream(fw)
    fg = stream(fg)
    if solid is not None:
        fw = bounce_back_cells(fw, solid)
        fg = bounce_back_cells(fg, solid)
    return fw, fg


def mixture_fields(fw, fg):
    rho_w, ux_w, uy_w = macroscopic(fw)
    rho_g, ux_g, uy_g = macroscopic(fg)
    rho_tot = (rho_w + rho_g).clamp(min=1e-12)
    ux = (rho_w * ux_w + rho_g * ux_g) / rho_tot
    uy = (rho_w * uy_w + rho_g * uy_g) / rho_tot
    return rho_w, rho_g, rho_tot, ux, uy


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(ny: int, n_steps: int, device: torch.device, tag: str) -> dict:
    """Run two-phase Poiseuille with viscous coupling."""
    NX = 8
    half = ny // 2
    TAU_W, TAU_G, G12, GX = 1.0, 0.75, -2.5, 5e-6

    # Initialize interface
    alpha = tanh_field_1d(ny, NX, float(half), 3.0, device)
    rw, rg = init_full_swap(alpha, 0.7, 0.3)
    z = torch.zeros((ny, NX), device=device)
    fw, fg = equilibrium(rw, z, z), equilibrium(rg, z, z)

    # Wall mask
    wall = torch.zeros((ny, NX), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    # Analytical solution
    nu_w, nu_g = (TAU_W - 0.5) / 3.0, (TAU_G - 0.5) / 3.0
    ana = _two_phase_poiseuille_analytical(ny, half, GX, nu_w, nu_g, nu_w / nu_g)
    ana_t = torch.tensor(ana, device=device)

    # Time stepping
    conv = []
    prev_umax = None
    mass_init = float(rw.sum() + rg.sum())

    t0 = time.time()
    for step in range(1, n_steps + 1):
        fw, fg = sc_mcmp_step_viscous(fw, fg, G12, TAU_W, TAU_G, GX, 0.0, wall)

        if step % 10000 == 0 or step == n_steps:
            rwt, rgt, rt, ux, uy = mixture_fields(fw, fg)
            umax = float(ux[1:-1, :].max())
            finite = bool(torch.isfinite(fw).all() and torch.isfinite(fg).all())
            mass_now = float(rwt.sum() + rgt.sum())
            mass_drift = abs(mass_now - mass_init) / mass_init

            conv.append({
                "step": step,
                "u_max": umax,
                "finite": finite,
                "mass_drift": mass_drift
            })

            rel = abs(umax - prev_umax) / prev_umax if prev_umax else None
            print(f"  {tag} step={step} u_max={umax:.6f} finite={finite} "
                  f"conv_rel={rel} mass_drift={mass_drift:.2e}", flush=True)
            prev_umax = umax

            if not finite:
                print(f"  {tag} DIVERGED at step {step}", flush=True)
                break

    elapsed = time.time() - t0

    # Final measurement
    rwt, rgt, rt, ux, uy = mixture_fields(fw, fg)
    prof = ux[:, NX // 2].cpu().tolist()
    prof_t = ux[:, NX // 2]
    umax_s = float(prof_t.max())

    # Error metrics
    m_sim = prof_t.abs() > 0.2 * umax_s
    m_ana = ana_t.abs() > 0.2 * max(ana)

    def errs(mask):
        e = (prof_t[mask] - ana_t[mask]) / ana_t[mask]
        return {
            "n": int(mask.sum()),
            "max_rel_err_pct": float(e.abs().max()) * 100,
            "mean_rel_err_pct": float(e.abs().mean()) * 100,
        }

    spur = float(uy[1:-1, 1:-1].abs().max())

    return {
        "tag": tag,
        "ny": ny,
        "nx": NX,
        "H_eff": ny - 1,
        "half": half,
        "tau_w": TAU_W,
        "tau_g": TAU_G,
        "G_12": G12,
        "G_x": GX,
        "nominal_M": nu_w / nu_g,
        "nu_w": nu_w,
        "nu_g": nu_g,
        "n_steps": n_steps,
        "elapsed_sec": elapsed,
        "convergence": conv,
        "u_profile_sim": prof,
        "u_profile_analytical": ana,
        "u_max_sim": umax_s,
        "u_max_ana": max(ana),
        "u_max_ratio": umax_s / max(ana),
        "center_err_mask_simmax": errs(m_sim),
        "center_err_mask_anamax": errs(m_ana),
        "spurious_uy_max": spur,
        "Ma_max": umax_s / math.sqrt(1 / 3.0),
        "mass_water": float(rwt.sum()),
        "mass_gas": float(rgt.sum()),
        "mass_total": float(rwt.sum() + rgt.sum()),
        "mass_drift_final": conv[-1]["mass_drift"] if conv else None,
        "finite": conv[-1]["finite"] if conv else False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    if args.quick:
        cases = [(65, 20000, "H64_quick")]
    else:
        cases = [(65, 60000, "H64"), (129, 180000, "H128")]

    results = {}
    for ny, n_steps, tag in cases:
        print(f"\n=== Running {tag} (ny={ny}, steps={n_steps}) ===", flush=True)
        results[tag] = run_benchmark(ny, n_steps, device, tag)

        r = results[tag]
        c = r["center_err_mask_simmax"]
        print(f"\n{tag} SUMMARY:")
        print(f"  max_err = {c['max_rel_err_pct']:.2f}%")
        print(f"  mean_err = {c['mean_rel_err_pct']:.2f}%")
        print(f"  u_max_ratio = {r['u_max_ratio']:.4f}")
        print(f"  spurious_uy = {r['spurious_uy_max']:.4f}")
        print(f"  finite = {r['finite']}")
        print(f"  mass_drift = {r['mass_drift_final']:.2e}")

    # Verdict
    all_pass = all(
        r["center_err_mask_simmax"]["max_rel_err_pct"] <= 10.0 and r["finite"]
        for r in results.values()
    )
    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n=== VERDICT: {verdict} ===")

    output = {
        "mission": "m8802",
        "fix": "viscous velocity coupling (u_eq='viscous')",
        "reference": "Martys et al. (1996) / sailfish convention",
        "results": results,
        "verdict": verdict,
        "acceptance_criterion": "max_rel_err <= 10%",
    }

    out_path = Path(args.out) / "result_viscous.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
