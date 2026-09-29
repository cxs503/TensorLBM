"""Verify wall_function double-treatment bug.

Hypothesis: wallfn_fullgrid_runner applies BOTH wall_function (Guo body force)
AND bounce_back on solid, causing double wall treatment and ~2x drag overestimate.

Test: Compare two step orders:
  - Original (buggy): collide → stream → wall_function → far_field_bc → bounce_back
  - Fixed: collide → stream → wall_function → far_field_bc (no bounce_back)

Expected: Fixed version should give Ct ≈ 0.004 (ITTC-1957), ~1% error.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.boundaries_d3q27 import bounce_back_cells_27, far_field_bc_27
from tensorlbm.cumulant import collide_cumulant_d3q27
from tensorlbm.d3q27 import (
    collide_bgk27,
    collide_mrt27,
    collide_rlbm27,
    correct_mass27,
    equilibrium27,
    macroscopic27,
)
from tensorlbm.solver3d import collide_bgk3d, collide_mrt3d, collide_rlbm3d, stream3d
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.wall_function_common import (
    compute_u_tau,
    compute_y_plus,
    wall_function,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.suboff_wallfn_fullgrid_runner import stream27_roll


@dataclass
class TestConfig:
    lattice: str = "D3Q27"
    collision: str = "CUMULANT"
    nx: int = 192
    ny: int = 72
    nz: int = 72
    u_in: float = 0.06
    re: float = 2.0e6
    hull_length: float = 72.0
    n_steps: int = 600
    device_id: int = 0
    use_bounce_back: bool = True  # True = original (buggy), False = fixed
    y_val: float = 0.5
    wall_law: str = "log"

    @property
    def tau(self) -> float:
        nu = self.u_in * self.hull_length / self.re
        return 3.0 * nu + 0.5

    @property
    def nu(self) -> float:
        return self.u_in * self.hull_length / self.re


def _stream(lattice: str, f: torch.Tensor) -> torch.Tensor:
    if lattice == "D3Q19":
        return stream3d(f)
    return stream27_roll(f)


def _far_field_bc(lattice: str, f, u_in):
    if lattice == "D3Q19":
        return far_field_bc_3d(f, u_in)
    return far_field_bc_27(f, u_in)


def _bounce_back(lattice: str, f, mask):
    if lattice == "D3Q19":
        return bounce_back_cells_3d(f, mask)
    return bounce_back_cells_27(f, mask)


def _macroscopic(lattice: str, f: torch.Tensor):
    if lattice == "D3Q19":
        return macroscopic3d(f)
    return macroscopic27(f)


def _equilibrium(lattice: str, rho, ux, uy, uz):
    if lattice == "D3Q19":
        return equilibrium3d(rho, ux, uy, uz)
    return equilibrium27(rho, ux, uy, uz)


def _correct_mass(lattice: str, f, target_mass):
    if lattice == "D3Q19":
        from tensorlbm.solver3d import correct_mass3d
        return correct_mass3d(f, target_mass)
    return correct_mass27(f, target_mass)


def _collide(lattice: str, collision: str, f: torch.Tensor, tau: float) -> torch.Tensor:
    if lattice == "D3Q27":
        if collision == "CUMULANT":
            return collide_cumulant_d3q27(f, tau)
        if collision == "BGK":
            return collide_bgk27(f, tau)
        if collision == "MRT":
            return collide_mrt27(f, tau)
        if collision == "RLBM":
            return collide_rlbm27(f, tau)
    else:  # D3Q19
        if collision == "BGK":
            return collide_bgk3d(f, tau)
        if collision == "MRT":
            return collide_mrt3d(f, tau)
        if collision == "RLBM":
            return collide_rlbm3d(f, tau)
    raise ValueError(f"Unknown {lattice} {collision}")


def _compute_drags(f, solid, u_tau, lattice):
    """Compute friction and pressure drag."""
    rho, ux, uy, uz = _macroscopic(lattice, f)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)

    # Near-wall mask
    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid

    # Friction drag
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    drag_fric = float((tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item())

    # Pressure drag
    p = (rho - 1.0) / 3.0
    sm = torch.roll(solid, -1, dims=2)
    sp = torch.roll(solid, 1, dims=2)
    drag_pres = float((p * (sm.to(f.dtype) - sp.to(f.dtype)) * fluid.to(f.dtype)).sum().item())

    return drag_fric, drag_pres


def run_test(cfg: TestConfig) -> dict:
    """Run one test configuration."""
    device = torch.device(f"sdaa:{cfg.device_id}")
    torch.sdaa.set_device(device)

    print(f"[SDAA:{cfg.device_id}] {cfg.lattice} {cfg.collision} "
          f"bounce_back={cfg.use_bounce_back} tau={cfg.tau:.6f}")

    # Build geometry
    cx = cfg.nx * 0.35
    cy = cfg.ny / 2.0
    cz = cfg.nz / 2.0
    solid, _ = build_suboff_mask(
        hull_type=SuboffHullType.BARE_HULL,
        nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
        cx=cx, cy=cy, cz=cz,
        length=cfg.hull_length, device="cpu",
    )
    solid = solid.to(device)

    wetted_area = _voxel_wetted_area(solid, 1.0)
    dynamic_pressure = 0.5 * 1.0 * cfg.u_in ** 2 * wetted_area

    # Initialize
    rho0 = torch.ones((cfg.nz, cfg.ny, cfg.nx))
    ux0 = torch.full_like(rho0, cfg.u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0[solid.cpu()] = 0.0
    f = _equilibrium(cfg.lattice, rho0, ux0, uy0, uz0)
    f = f.to(device)
    initial_mass = float(f.sum().item())

    tau = cfg.tau
    nu = cfg.nu

    # Run
    ct_series = []
    for step in range(1, cfg.n_steps + 1):
        # 1. Collision
        f = _collide(cfg.lattice, cfg.collision, f, tau)

        # 2. Streaming
        f = _stream(cfg.lattice, f)

        # 3. Wall function
        rho, ux, uy, uz = _macroscopic(cfg.lattice, f)
        u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
        u_tau = compute_u_tau(u_mag, nu, y_val=cfg.y_val, wall_law=cfg.wall_law)
        y_plus = compute_y_plus(u_tau, nu, y_val=cfg.y_val)

        # Compute drag BEFORE wall_function modifies f
        drag_fric, drag_pres = _compute_drags(f, solid, u_tau, cfg.lattice)

        # Apply wall function
        f = wall_function(f, solid, u_tau, y_plus, lattice=cfg.lattice, nu=nu, y_val=cfg.y_val)

        # 4. Far-field BC
        f = _far_field_bc(cfg.lattice, f, cfg.u_in)

        # 5. Bounce-back (ONLY if use_bounce_back=True)
        if cfg.use_bounce_back:
            f = _bounce_back(cfg.lattice, f, solid)

        # 6. Record
        ct_fric = drag_fric / dynamic_pressure if dynamic_pressure > 0 else 0.0
        ct_pres = drag_pres / dynamic_pressure if dynamic_pressure > 0 else 0.0
        ct_total = ct_fric + ct_pres
        ct_series.append({"step": step, "ct_fric": ct_fric, "ct_pres": ct_pres, "ct_total": ct_total})

        # Mass correction every 100 steps
        if step % 100 == 0:
            f = _correct_mass(cfg.lattice, f, initial_mass)

        # Finiteness check
        if not torch.isfinite(f).all():
            print(f"[SDAA:{cfg.device_id}] DIVERGED at step {step}")
            break

        if step % 100 == 0:
            print(f"[SDAA:{cfg.device_id}] step={step} Ct={ct_total:.6f}")

    # Average over last 50%
    warmup = max(1, len(ct_series) // 2)
    ct_fric_avg = sum(e["ct_fric"] for e in ct_series[warmup:]) / max(len(ct_series[warmup:]), 1)
    ct_pres_avg = sum(e["ct_pres"] for e in ct_series[warmup:]) / max(len(ct_series[warmup:]), 1)
    ct_total_avg = ct_fric_avg + ct_pres_avg

    ref_ct = 0.00405  # ITTC-1957
    err_pct = abs(ct_total_avg - ref_ct) / ref_ct * 100

    result = {
        "lattice": cfg.lattice,
        "collision": cfg.collision,
        "bounce_back": cfg.use_bounce_back,
        "Ct_fric": ct_fric_avg,
        "Ct_pres": ct_pres_avg,
        "Ct_total": ct_total_avg,
        "error_pct": err_pct,
        "steps": len(ct_series),
        "finite": torch.isfinite(f).all().item(),
    }

    print(f"[SDAA:{cfg.device_id}] DONE: Ct={ct_total_avg:.6f} err={err_pct:.1f}%")
    return result


def main():
    """Run verification tests on multiple SDAA cards."""
    # Test configurations
    configs = [
        # Original (buggy): with bounce_back
        TestConfig(lattice="D3Q27", collision="CUMULANT", device_id=0, use_bounce_back=True),
        TestConfig(lattice="D3Q27", collision="MRT", device_id=1, use_bounce_back=True),
        TestConfig(lattice="D3Q27", collision="RLBM", device_id=2, use_bounce_back=True),
        # Fixed: without bounce_back
        TestConfig(lattice="D3Q27", collision="CUMULANT", device_id=3, use_bounce_back=False),
        TestConfig(lattice="D3Q27", collision="MRT", device_id=4, use_bounce_back=False),
        TestConfig(lattice="D3Q27", collision="RLBM", device_id=5, use_bounce_back=False),
    ]

    print("=" * 80)
    print("Wall Function Double-Treatment Verification")
    print("=" * 80)
    print(f"Grid: 192×72×72, Re=2×10⁶, u_in=0.06, 600 steps")
    print(f"Reference: ITTC-1957 Ct=0.00405")
    print()

    # Run sequentially (SDAA doesn't support true parallel in single process)
    results = []
    for cfg in configs:
        result = run_test(cfg)
        results.append(result)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Lattice':<8} {'Collision':<10} {'BounceBack':<12} {'Ct_fric':<10} {'Ct_pres':<10} {'Ct_total':<10} {'Error%':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['lattice']:<8} {r['collision']:<10} {str(r['bounce_back']):<12} "
              f"{r['Ct_fric']:<10.6f} {r['Ct_pres']:<10.6f} {r['Ct_total']:<10.6f} {r['error_pct']:<8.1f}")

    # Save results
    out_path = Path("wallfn_verification_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
