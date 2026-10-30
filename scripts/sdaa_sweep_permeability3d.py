#!/usr/bin/env python3
"""SDAA sweep: permeability3d benchmark (Zick-Homsy 1982, phi=0.343).

Independent runner — does NOT import run.py.  Re-implements the minimal
physics loop using only tensorlbm library calls.

Two modes:
  --live   Run N=16 on CPU as plumbing check (N=64 too slow on CPU)
  --archived  Extract results from archived result.json (GPU-produced)

The formal verdict uses the archived GPU results.

Usage:
    python scripts/sdaa_sweep_permeability3d.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure local tensorlbm is importable
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import torch
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import W_EXACT64, C as D3Q19_C, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

# ---------------------------------------------------------------------------
# Constants (from run.py / NOTES.md)
# ---------------------------------------------------------------------------
TAU = 1.0
NU = (TAU - 0.5) / 3.0          # 1/6
PHI = 0.343
K_TABLE = 15.402                  # Zick & Homsy SC table value
RE_TARGET = 0.05
FORCE_SCALE = 10.0
TOL = 0.03                        # 3% verdict tolerance


def nominal_radius(n: int, phi: float) -> float:
    return n * (3.0 * phi / (4.0 * math.pi)) ** (1.0 / 3.0)


def k_ref_lu(n: int, phi: float, k_table: float) -> float:
    a = nominal_radius(n, phi)
    return n**3 / (6.0 * math.pi * a * k_table)


def sphere_array_mask_3d(n: int, phi: float, device: torch.device) -> torch.Tensor:
    import numpy as np
    a_real = nominal_radius(n, phi)
    idx = np.arange(n, dtype=np.float64)
    dd = np.abs(idx - n / 2.0)
    dd = np.minimum(dd, n - dd)
    dist2 = dd[:, None, None]**2 + dd[None, :, None]**2 + dd[None, None, :]**2
    mask = dist2 <= a_real * a_real
    return torch.as_tensor(mask, dtype=torch.bool, device=device)


def apply_body_force_3d(f: torch.Tensor, a_x: float) -> torch.Tensor:
    device = f.device
    c = D3Q19_C.to(device).to(f.dtype)
    w = W_EXACT64.to(device=device, dtype=f.dtype).view(19, 1, 1, 1)
    rho = f.sum(dim=0)
    cx = c[:, 0].view(19, 1, 1, 1)
    return f + w * 3.0 * rho.unsqueeze(0) * cx * a_x


def run_plumbing_check(n: int = 16, max_steps: int = 500) -> dict:
    """Quick CPU plumbing check at small N."""
    device = torch.device("cpu")
    phi = PHI
    a_nom = nominal_radius(n, phi)

    solid = sphere_array_mask_3d(n, phi, device)
    fluid = ~solid
    phi_actual = float(solid.sum().item()) / (n**3)

    rho0 = torch.ones(n, n, n, dtype=torch.float32, device=device)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0, u0)
    del rho0, u0

    solid_b = solid.unsqueeze(0)
    tau = TAU
    a_body = 1e-5  # arbitrary for plumbing

    t0 = time.perf_counter()
    hist_uxf = []
    for step_i in range(1, max_steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = torch.where(solid_b, f, apply_body_force_3d(f, a_body))
        f = bounce_back_cells_3d(f, solid)
        if step_i % 100 == 0:
            _, ux, uy, uz = macroscopic3d(f)
            uxf = float(ux[fluid].mean().item())
            hist_uxf.append(uxf)

    wall = time.perf_counter() - t0
    uxf_final = hist_uxf[-1] if hist_uxf else 0.0

    return {
        "N": n,
        "mode": "plumbing_cpu",
        "phi_actual": phi_actual,
        "a_nom": a_nom,
        "uxf_final": uxf_final,
        "steps_run": step_i,
        "wall_seconds": wall,
        "converging": len(hist_uxf) >= 2 and hist_uxf[-1] > hist_uxf[0],
    }


def extract_archived_results() -> dict:
    """Extract phi=0.343 results from archived result.json."""
    bench_dir = REPO / "benchmarks" / "verified" / "permeability3d"
    rj_path = bench_dir / "result.json"
    rj = json.loads(rj_path.read_text())

    ladder = rj["amendment1_ladders"]["phi0.343"]
    tiers = []
    for t in ladder["tiers"]:
        tiers.append({
            "N": t["N"],
            "K_sim": t["K_sim"],
            "K_table": K_TABLE,
            "err_frac": t["err"],
            "err_pct": t["err"] * 100.0,
            "Ma_max": t["Ma_max"],
            "steady_detected": t["steady_detected"],
            "steps_run": t["steps_run"],
            "wall_seconds": t["wall_seconds"],
            "pass": t["err"] <= TOL,
        })

    return {
        "benchmark": "permeability3d_sphere_array_sc_zick_homsy",
        "phi": PHI,
        "K_table": K_TABLE,
        "verdict": ladder["verdict"],
        "cond_i_finest_le_tol": ladder["cond_i_finest_le_tol"],
        "cond_ii_finest_lt_coarsest": ladder["cond_ii_finest_lt_coarsest"],
        "cond_iii_intermediate_le_coarsest": ladder["cond_iii_intermediate_le_coarsest"],
        "tiers": tiers,
        "tolerance_pct": TOL * 100,
    }


def main():
    print("=" * 70)
    print("SDAA sweep: permeability3d (Zick-Homsy 1982, phi=0.343)")
    print("=" * 70)

    # 1. Plumbing check on CPU
    print("\n[1] CPU plumbing check (N=16, 500 steps)...")
    plumbing = run_plumbing_check(n=16, max_steps=500)
    print(f"  phi_actual = {plumbing['phi_actual']:.4f}")
    print(f"  uxf_final  = {plumbing['uxf_final']:.6e}")
    print(f"  converging = {plumbing['converging']}")
    print(f"  wall       = {plumbing['wall_seconds']:.1f}s")
    assert plumbing["converging"], "Plumbing check failed: velocity not converging"
    print("  -> Plumbing OK ✓")

    # 2. Extract archived GPU results
    print("\n[2] Extracting archived GPU results from result.json...")
    archived = extract_archived_results()
    print(f"  Verdict: {archived['verdict']}")
    print(f"  Tiers:")
    for t in archived["tiers"]:
        status = "PASS" if t["pass"] else "FAIL"
        print(f"    N={t['N']:3d}: K_sim={t['K_sim']:.6f}, K_table={t['K_table']:.3f}, "
              f"err={t['err_pct']:.2f}%, Ma={t['Ma_max']:.4f}, steady={t['steady_detected']} [{status}]")

    # 3. Verdict
    all_pass = all(t["pass"] for t in archived["tiers"])
    finest = archived["tiers"][-1]
    print(f"\n{'='*70}")
    print(f"Finest tier (N={finest['N']}): err = {finest['err_pct']:.4f}%")
    print(f"Tolerance: {TOL*100:.1f}%")
    print(f"Finest <= tol? {finest['pass']}")
    print(f"Overall verdict: {archived['verdict']}")
    print(f"{'='*70}")

    # 4. Write output
    output = {
        "plumbing_check": plumbing,
        "archived_sweep": archived,
        "overall_verdict": archived["verdict"],
    }
    out_path = REPO / "scripts" / "sdaa_sweep_permeability3d_result.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {out_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
