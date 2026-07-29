"""Verify wall_function double-treatment bug — LARGE GRID version.

Uses all 32 SDAA cards via multiprocessing.

Test matrix (16 configs × 2 grid sizes = 32 total):
  - Grids: 320×120×120, 480×180×180
  - Lattice: D3Q27
  - Collisions: CUMULANT, MRT, RLBM, BGK
  - Bounce-back: True (original/buggy) vs False (fixed)

Reference: ITTC-1957 Ct = 0.00405, Re = 2×10⁶
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from multiprocessing import Process, Queue
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent / "src"))

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
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.wall_function_common import (
    compute_u_tau,
    compute_y_plus,
    wall_function,
)
from tensorlbm.suboff_wallfn_fullgrid_runner import stream27_roll


@dataclass
class TestConfig:
    lattice: str
    collision: str
    nx: int
    ny: int
    nz: int
    u_in: float
    re: float
    hull_length: float
    n_steps: int
    device_id: int
    use_bounce_back: bool
    y_val: float = 0.5
    wall_law: str = "log"

    @property
    def tau(self) -> float:
        nu = self.u_in * self.hull_length / self.re
        return 3.0 * nu + 0.5

    @property
    def nu(self) -> float:
        return self.u_in * self.hull_length / self.re


def _compute_drags(f, solid, u_tau):
    rho, ux, uy, uz = macroscopic27(f)
    u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid
    tau_w = u_tau * u_tau
    inv_umag = 1.0 / u_mag
    drag_fric = float((tau_w * (ux * inv_umag) * near.to(f.dtype)).sum().item())
    p = (rho - 1.0) / 3.0
    sm = torch.roll(solid, -1, dims=2)
    sp = torch.roll(solid, 1, dims=2)
    drag_pres = float((p * (sm.to(f.dtype) - sp.to(f.dtype)) * fluid.to(f.dtype)).sum().item())
    return drag_fric, drag_pres


def _collide(collision: str, f: torch.Tensor, tau: float) -> torch.Tensor:
    if collision == "CUMULANT":
        return collide_cumulant_d3q27(f, tau)
    if collision == "BGK":
        return collide_bgk27(f, tau)
    if collision == "MRT":
        return collide_mrt27(f, tau)
    if collision == "RLBM":
        return collide_rlbm27(f, tau)
    raise ValueError(f"Unknown collision {collision}")


def run_one(cfg: TestConfig, result_queue: Queue):
    """Run one test on a specific SDAA card."""
    device = torch.device(f"sdaa:{cfg.device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{cfg.device_id} {cfg.nx}³ {cfg.collision} bb={cfg.use_bounce_back}]"
    print(f"{tag} START tau={cfg.tau:.6f} nu={cfg.nu:.8f}")

    try:
        # Build geometry on CPU
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
        f = equilibrium27(rho0, ux0, uy0, uz0)
        f = f.to(device)
        initial_mass = float(f.sum().item())

        tau = cfg.tau
        nu = cfg.nu

        ct_series = []
        t0 = time.time()

        for step in range(1, cfg.n_steps + 1):
            f = _collide(cfg.collision, f, tau)
            f = stream27_roll(f)

            rho, ux, uy, uz = macroscopic27(f)
            u_mag = torch.sqrt(ux * ux + uy * uy + uz * uz).clamp(min=1e-12)
            u_tau = compute_u_tau(u_mag, nu, y_val=cfg.y_val, wall_law=cfg.wall_law)
            y_plus = compute_y_plus(u_tau, nu, y_val=cfg.y_val)

            drag_fric, drag_pres = _compute_drags(f, solid, u_tau)

            f = wall_function(f, solid, u_tau, y_plus, lattice="D3Q27", nu=nu, y_val=cfg.y_val)
            f = far_field_bc_27(f, cfg.u_in)

            if cfg.use_bounce_back:
                f = bounce_back_cells_27(f, solid)

            ct_fric = drag_fric / dynamic_pressure if dynamic_pressure > 0 else 0.0
            ct_pres = drag_pres / dynamic_pressure if dynamic_pressure > 0 else 0.0
            ct_total = ct_fric + ct_pres
            ct_series.append({"step": step, "ct_fric": ct_fric, "ct_pres": ct_pres, "ct_total": ct_total})

            if step % 100 == 0:
                f = correct_mass27(f, initial_mass)

            if not torch.isfinite(f).all():
                print(f"{tag} DIVERGED at step {step}")
                break

            if step % 200 == 0:
                elapsed = time.time() - t0
                print(f"{tag} step={step}/{cfg.n_steps} Ct={ct_total:.6f} ({elapsed:.1f}s)")

        elapsed = time.time() - t0
        warmup = max(1, len(ct_series) // 2)
        ct_fric_avg = sum(e["ct_fric"] for e in ct_series[warmup:]) / max(len(ct_series[warmup:]), 1)
        ct_pres_avg = sum(e["ct_pres"] for e in ct_series[warmup:]) / max(len(ct_series[warmup:]), 1)
        ct_total_avg = ct_fric_avg + ct_pres_avg

        ref_ct = 0.00405
        err_pct = abs(ct_total_avg - ref_ct) / ref_ct * 100

        result = {
            "grid": f"{cfg.nx}x{cfg.ny}x{cfg.nz}",
            "collision": cfg.collision,
            "bounce_back": cfg.use_bounce_back,
            "Ct_fric": ct_fric_avg,
            "Ct_pres": ct_pres_avg,
            "Ct_total": ct_total_avg,
            "error_pct": err_pct,
            "steps": len(ct_series),
            "finite": bool(torch.isfinite(f).all().item()),
            "elapsed_s": elapsed,
            "tau": tau,
            "nu": nu,
        }

        print(f"{tag} DONE: Ct={ct_total_avg:.6f} err={err_pct:.1f}% ({elapsed:.1f}s)")
        result_queue.put(result)

    except Exception as e:
        print(f"{tag} ERROR: {e}")
        result_queue.put({
            "grid": f"{cfg.nx}x{cfg.ny}x{cfg.nz}",
            "collision": cfg.collision,
            "bounce_back": cfg.use_bounce_back,
            "error": str(e),
        })


def main():
    collisions = ["CUMULANT", "MRT", "RLBM", "BGK"]
    grids = [
        (320, 120, 120, 120.0),   # medium
        (480, 180, 180, 180.0),   # large
    ]
    n_steps_map = {320: 1500, 480: 2000}

    configs = []
    device_id = 0
    for nx, ny, nz, hull_len in grids:
        for col in collisions:
            for bb in [True, False]:
                configs.append(TestConfig(
                    lattice="D3Q27",
                    collision=col,
                    nx=nx, ny=ny, nz=nz,
                    u_in=0.06,
                    re=2.0e6,
                    hull_length=hull_len,
                    n_steps=n_steps_map[nx],
                    device_id=device_id,
                    use_bounce_back=bb,
                ))
                device_id += 1

    print("=" * 90)
    print("Wall Function Double-Treatment Verification — LARGE GRID")
    print("=" * 90)
    print(f"Configs: {len(configs)}")
    print(f"Grids: 320×120×120 (1500 steps), 480×180×180 (2000 steps)")
    print(f"Re=2×10⁶, u_in=0.06, D3Q27")
    print(f"Reference: ITTC-1957 Ct=0.00405")
    print()

    result_queue = Queue()
    procs = []
    for cfg in configs:
        p = Process(target=run_one, args=(cfg, result_queue))
        p.start()
        procs.append(p)

    # Collect results
    results = []
    for _ in configs:
        r = result_queue.get()
        results.append(r)

    for p in procs:
        p.join()

    # Sort results
    results.sort(key=lambda r: (r.get("grid", ""), r.get("collision", ""), r.get("bounce_back", True)))

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Grid':<18} {'Collision':<10} {'BB':<6} {'Ct_fric':<10} {'Ct_pres':<10} {'Ct_total':<10} {'Err%':<8} {'OK':<5}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r.get('grid','?'):<18} {r.get('collision','?'):<10} {str(r.get('bounce_back','?')):<6} ERROR: {r['error']}")
        else:
            print(f"{r['grid']:<18} {r['collision']:<10} {str(r['bounce_back']):<6} "
                  f"{r['Ct_fric']:<10.6f} {r['Ct_pres']:<10.6f} {r['Ct_total']:<10.6f} {r['error_pct']:<8.1f} {'✓' if r['finite'] else '✗'}")

    out_path = Path("wallfn_large_grid_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
