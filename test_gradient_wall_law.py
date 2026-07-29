#!/usr/bin/env python3
"""Test gradient wall law on SUBOFF and KVLCC2 at 200³ resolution.

Runs 4 concurrent simulations on SDAA cards 0-3:
  SDAA:0 → SUBOFF 200³ log-law (base)
  SDAA:1 → SUBOFF 200³ gradient
  SDAA:2 → KVLCC2 200³ log-law (base)
  SDAA:3 → KVLCC2 200³ gradient

Each: D3Q27 Cumulant, 2000 steps, Cs=0.05, running-average drag.
Outputs comparison to /tmp/gradient_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

# Set up path for local dev
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tensorlbm.boundaries_d3q27 import bounce_back_cells_27, far_field_bc_27
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.d3q27 import correct_mass27, equilibrium27, macroscopic27
from tensorlbm.ship_cad import ShipHullType, build_hull_mask
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.wall_function_common import (
    _near_wall_mask,
    compute_u_tau,
    wall_function,
)

# ---------------------------------------------------------------------------
# Streaming (torch.roll, memory-efficient)
# ---------------------------------------------------------------------------

_D3Q27_SHIFTS: list[tuple[int, int, int]] = [
    (0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1), (1, 1, 0), (-1, 1, 0), (1, -1, 0),
    (-1, -1, 0), (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
    (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1),
]


def stream27_roll(f: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(f)
    for q in range(27):
        sx, sy, sz = _D3Q27_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


# ---------------------------------------------------------------------------
# Drag computation
# ---------------------------------------------------------------------------

def compute_drags(f, solid, u_tau):
    rho, ux, uy, uz = macroscopic27(f)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    near = _near_wall_mask(solid)
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    drag_fric = float((tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item())
    p = (rho - 1.0) / 3.0
    fluid = ~solid
    sm = torch.roll(solid, -1, dims=2)
    sp = torch.roll(solid, 1, dims=2)
    drag_pres = float((p * (sm.to(f.dtype) - sp.to(f.dtype)) * fluid.to(f.dtype)).sum().item())
    return drag_fric, drag_pres


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def run_simulation(
    device: str,
    solid: torch.Tensor,
    hull_name: str,
    n_steps: int,
    u_in: float,
    nu: float,
    wall_law: str,
    y_val: float = 0.5,
    cs_smag: float = 0.05,
) -> dict[str, Any]:
    """Run a single wall-function simulation and return results."""
    device_t = torch.device(device)
    solid = solid.to(device_t)
    nz, ny, nx = solid.shape

    # Wetted area for Ct normalization
    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in ** 2 * S

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device_t)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium27(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      device=device_t)
    initial_mass = float(f.sum().item())
    tau = 3.0 * nu + 0.5

    # Smagorinsky
    from tensorlbm.cumulant_smag import collide_cumulant_smag_d3q27 as _collide_fn

    ct_fric_series: list[float] = []
    ct_pres_series: list[float] = []
    ct_total_series: list[float] = []

    for step in range(1, n_steps + 1):
        # Collision (Cumulant with Smagorinsky)
        f = _collide_fn(f, tau, C_s=cs_smag)

        # Stream
        f = stream27_roll(f)

        # Macroscopic
        rho, ux, uy, uz = macroscopic27(f)
        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)

        # Compute u_tau
        u_tau = compute_u_tau(u_mag, nu, y_val=y_val, wall_law=wall_law)

        # Drag (before wall_function modifies f)
        drag_fric, drag_pres = compute_drags(f, solid, u_tau)

        # Ct
        ct_fric = drag_fric / dyn_p_S
        ct_pres = drag_pres / dyn_p_S
        ct_total = ct_fric + ct_pres

        ct_fric_series.append(ct_fric)
        ct_pres_series.append(ct_pres)
        ct_total_series.append(ct_total)

        # Wall function (Guo body force)
        y_plus = u_tau * y_val / nu  # compute y_plus from pre-computed u_tau
        f = wall_function(
            f, solid, u_tau, y_plus,
            lattice="D3Q27", nu=nu, y_val=y_val,
        )

        # Far-field BC
        f = far_field_bc_27(f, u_in)

        # Bounce-back
        f = bounce_back_cells_27(f, solid)

        # Mass correction
        f = correct_mass27(f, initial_mass)

        # Finite check every 100 steps
        if step % 100 == 0:
            if not torch.isfinite(f).all():
                return {
                    "device": device, "hull": hull_name, "wall_law": wall_law,
                    "status": "DIVERGED", "step_diverged": step,
                    "Ct_total": float("nan"), "Ct_fric": float("nan"),
                    "Ct_pres": float("nan"),
                }

    # Running average (last 50%)
    n_avg = n_steps // 2
    avg_fric = sum(ct_fric_series[-n_avg:]) / n_avg
    avg_pres = sum(ct_pres_series[-n_avg:]) / n_avg
    avg_total = avg_fric + avg_pres

    # Also compute full average
    full_fric = sum(ct_fric_series) / n_steps
    full_pres = sum(ct_pres_series) / n_steps
    full_total = full_fric + full_pres

    return {
        "device": device,
        "hull": hull_name,
        "wall_law": wall_law,
        "grid": (nx, ny, nz),
        "steps": n_steps,
        "u_in": u_in,
        "nu": nu,
        "y_val": y_val,
        "wetted_area": S,
        "status": "OK",
        "Ct_fric": avg_fric,
        "Ct_pres": avg_pres,
        "Ct_total": avg_total,
        "Ct_fric_full": full_fric,
        "Ct_pres_full": full_pres,
        "Ct_total_full": full_total,
        "ct_series_last_20": ct_total_series[-20:],
    }


# ---------------------------------------------------------------------------
# Job configuration
# ---------------------------------------------------------------------------

JOBS = [
    # SUBOFF BARE_HULL 200³
    {"device": "sdaa:0", "hull": "SUBOFF", "wall_law": "log",
     "hull_type": SuboffHullType.BARE_HULL, "nx": 200, "ny": 200, "nz": 200,
     "hull_length": 160.0},
    {"device": "sdaa:1", "hull": "SUBOFF", "wall_law": "gradient",
     "hull_type": SuboffHullType.BARE_HULL, "nx": 200, "ny": 200, "nz": 200,
     "hull_length": 160.0},
    # KVLCC2 200³
    {"device": "sdaa:2", "hull": "KVLCC2", "wall_law": "log",
     "hull_type": ShipHullType.KVLCC2, "nx": 200, "ny": 200, "nz": 200,
     "hull_length": 160.0},
    {"device": "sdaa:3", "hull": "KVLCC2", "wall_law": "gradient",
     "hull_type": ShipHullType.KVLCC2, "nx": 200, "ny": 200, "nz": 200,
     "hull_length": 160.0},
]

# Shared params
U_IN = 0.06
N_STEPS = 2000
CS_SMAG = 0.05
Y_VAL = 0.5


def build_geometry(job: dict) -> torch.Tensor:
    """Build solid mask on CPU."""
    if job["hull"] == "SUBOFF":
        cx = job["nx"] * 0.35
        cy = job["ny"] / 2.0
        cz = job["nz"] / 2.0
        solid, _stats = build_suboff_mask(
            hull_type=job["hull_type"],
            nx=job["nx"], ny=job["ny"], nz=job["nz"],
            cx=cx, cy=cy, cz=cz, length=job["hull_length"],
            device="cpu",
        )
    else:
        cx = job["nx"] / 2.0
        cy = job["ny"] / 2.0
        cz_keel = job["nz"] / 4.0
        length = job["hull_length"]
        beam = job["ny"] * 0.25
        draft = job["nz"] * 0.3
        solid, _stats = build_hull_mask(
            hull_type=job["hull_type"],
            nx=job["nx"], ny=job["ny"], nz=job["nz"],
            cx=cx, cy=cy, cz_keel=cz_keel,
            length=length, beam=beam, draft=draft,
            device="cpu",
        )
    return solid


def main():
    # Check SDAA availability
    try:
        for dev in ["sdaa:0", "sdaa:1", "sdaa:2", "sdaa:3"]:
            test_t = torch.zeros(1, device=dev)
    except Exception as e:
        print(f"WARNING: SDAA devices not all available: {e}")
        print("Falling back to CPU (results will be slower).")
        for j in JOBS:
            j["device"] = "cpu"

    results: list[dict[str, Any]] = []
    hull_length = JOBS[0]["hull_length"]
    nu = U_IN * hull_length / 2_000_000.0  # Re = 2M

    print(f"nu = {nu:.8e}, n_steps = {N_STEPS}, Cs = {CS_SMAG}")
    print(f"Grid: {JOBS[0]['nx']}×{JOBS[0]['ny']}×{JOBS[0]['nz']}")
    print()

    for job in JOBS:
        print(f"Building geometry: {job['hull']} for {job['device']}...")
        solid = build_geometry(job)
        n_solid = int(solid.sum().item())
        print(f"  Solid cells: {n_solid}, "
              f"Grid: {solid.shape[2]}×{solid.shape[1]}×{solid.shape[0]}")
        print(f"Running: {job['hull']} wall_law={job['wall_law']} on {job['device']}...")
        sys.stdout.flush()

        try:
            result = run_simulation(
                device=job["device"],
                solid=solid,
                hull_name=job["hull"],
                n_steps=N_STEPS,
                u_in=U_IN,
                nu=nu,
                wall_law=job["wall_law"],
                y_val=Y_VAL,
                cs_smag=CS_SMAG,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            result = {
                "device": job["device"],
                "hull": job["hull"],
                "wall_law": job["wall_law"],
                "status": "ERROR",
                "error": str(e),
            }

        results.append(result)
        print(f"  Status: {result.get('status')}, "
              f"Ct_total: {result.get('Ct_total', 'N/A')}, "
              f"Ct_fric: {result.get('Ct_fric', 'N/A')}, "
              f"Ct_pres: {result.get('Ct_pres', 'N/A')}")
        print()

    # Build comparison summary
    comparison = {}
    for hull in ["SUBOFF", "KVLCC2"]:
        base = [r for r in results if r["hull"] == hull and r["wall_law"] == "log"]
        grad = [r for r in results if r["hull"] == hull and r["wall_law"] == "gradient"]
        if base and grad and base[0].get("status") == "OK" and grad[0].get("status") == "OK":
            b, g = base[0], grad[0]
            comparison[hull] = {
                "Ct_total_base": b["Ct_total"],
                "Ct_total_gradient": g["Ct_total"],
                "delta_Ct": g["Ct_total"] - b["Ct_total"],
                "delta_Ct_pct": (g["Ct_total"] - b["Ct_total"]) / abs(b["Ct_total"]) * 100
                if abs(b["Ct_total"]) > 1e-10 else 0.0,
                "Ct_fric_base": b["Ct_fric"],
                "Ct_fric_gradient": g["Ct_fric"],
                "Ct_pres_base": b["Ct_pres"],
                "Ct_pres_gradient": g["Ct_pres"],
            }

    output = {
        "description": "Gradient wall law test: SUBOFF and KVLCC2 at 200³",
        "parameters": {
            "n_steps": N_STEPS,
            "u_in": U_IN,
            "nu": nu,
            "cs_smag": CS_SMAG,
            "y_val": Y_VAL,
            "Re": 2_000_000,
        },
        "results": results,
        "comparison": comparison,
    }

    out_path = Path("/tmp/gradient_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Results written to {out_path}")
    return output


if __name__ == "__main__":
    main()
