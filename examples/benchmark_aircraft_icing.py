#!/usr/bin/env python
"""Aircraft icing benchmark — NACA 0012 rime ice (NASA Glenn IRT).

Phase 1 (this file): NACA 0012 geometry + air flow (D3Q19 BGK).
Phase 2 (TODO): supercooled droplets (PF multiphase) + impact freezing.

Benchmark (NASA NACA 0012 rime, Ruff & Wright):
  chord c=0.5334 m, V=67 m/s, Re=2.5e6, LWC=0.5 g/m³, MVD=20 μm,
  T=−10 °C (rime, full freezing), t=360 s, AoA=4°.
  Validate: ice shape profile (leading-edge horn) vs NASA IRT / LEWICE.
"""
from __future__ import annotations
import argparse, math, os, sys
from pathlib import Path
import numpy as np
import torch

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from hull_fs_pf_mrt import _init_g_equilibrium, _laplacian_3d


def naca0012_mask(nx, ny, nz, chord_frac=0.4, aoa_deg=4.0,
                  cx_frac=0.3, cy_frac=0.5, dev=None):
    """NACA 0012 airfoil solid mask (2-D, nz=1).

    Thickness distribution (NACA 4-digit, t=0.12):
      yt = 5·0.12·(0.2969√x − 0.126x − 0.3516x² + 0.2843x³ − 0.1015x⁴)
    Airfoil rotated by AoA, placed at (cx0, cy0) with chord = nx·chord_frac.
    """
    if dev is None:
        dev = torch.device("cpu")
    t = 0.12
    chord = nx * chord_frac
    cx0 = nx * cx_frac
    cy0 = ny * cy_frac
    aoa = math.radians(aoa_deg)
    cos_a, sin_a = math.cos(aoa), math.sin(aoa)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev), torch.arange(ny, device=dev),
        torch.arange(nx, device=dev), indexing="ij")
    dx = xx.float() - cx0
    dy = yy.float() - cy0
    # rotate to airfoil-aligned coords
    xr = (dx * cos_a + dy * sin_a) / chord        # normalized chord [0,1]
    yr = (-dx * sin_a + dy * cos_a) / chord       # normalized thickness
    xclamp = xr.clamp(0.0, 1.0)
    yt = 5.0 * t * (0.2969 * torch.sqrt(xclamp) - 0.126 * xclamp
                    - 0.3516 * xclamp ** 2 + 0.2843 * xclamp ** 3
                    - 0.1015 * xclamp ** 4)
    inside = (xr >= 0.0) & (xr <= 1.0) & (yr.abs() <= yt)
    return inside


def run_aircraft_icing(nx=200, ny=100, nz=1, u_in=0.06, tau=0.55,
                       chord_frac=0.4, aoa_deg=4.0, steps=2000,
                       device="cpu", log_every=500):
    """Phase 1: NACA 0012 + air flow (no icing yet)."""
    dev = torch.device(device)
    solid = naca0012_mask(nx, ny, nz, chord_frac, aoa_deg, dev=dev)
    n_solid = int(solid.sum().item())
    print(f"  NACA 0012: chord={nx*chord_frac:.0f} cells, AoA={aoa_deg}°, "
          f"solid cells={n_solid}")

    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.full((nz, ny, nx), u_in, device=dev)
    zero = torch.zeros_like(u0)
    f = equilibrium3d(rho0, u0, zero.clone(), zero.clone(), device=dev)
    opp = OPPOSITE.to(dev)

    # Inlet equilibrium (fixed)
    feq_in = equilibrium3d(rho0, u0, zero.clone(), zero.clone(), device=dev)

    # Lagrangian droplet particles (rime ice: droplets hit airfoil → freeze)
    # Each particle: (x, y) position, advected by flow field.
    n_droplets_per_step = 12  # droplets seeded per step (LWC-matched: 0.5g/m³, MVD=20μm)
    droplets = []  # list of (x, y) tensors
    ice_mask = solid.clone()
    original_solid = solid.clone()  # airfoil only (no ice)
    original_solid = solid.clone()  # airfoil only (no ice) for ice shape extraction
    collision_count = torch.zeros((nz, ny, nx), device=dev, dtype=torch.float32)
    ice_threshold = 1.1  # droplets per cell to freeze (ρ_water/ρ_ice=1000/917)

    print(f"  Flow: u_in={u_in}, tau={tau}, Re≈{u_in*nx*chord_frac/((tau-0.5)/3):.0f}, "
          f"steps={steps}")
    print(f"  {'step':>6s} {'u_max':>8s} {'rho_min':>8s} {'rho_max':>8s} {'cd':>8s} {'cl':>8s}")

    history = []
    for step in range(1, steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        # Bounce-back on airfoil (save pre-bounce for force measurement)
        f_pre = f.clone()
        f = torch.where(solid.unsqueeze(0), f[opp], f)
        # Inlet (Dirichlet)
        f[:, :, :, 0] = feq_in[:, :, :, 0]
        # Outlet (convective)
        f[:, :, :, -1] = f[:, :, :, -2]
        # Top/bottom (freestream) — dim2 = ny
        f[:, :, 0, :] = feq_in[:, :, 0, :]
        f[:, :, -1, :] = feq_in[:, :, -1, :]
        # z-direction (wingtip, freestream) — dim1 = nz
        if nz > 1:
            f[:, 0, :, :] = feq_in[:, 0, :, :]
            f[:, -1, :, :] = feq_in[:, -1, :, :]

        # === Lagrangian droplet particles (3D) ===
        rho, ux, uy, uz = macroscopic3d(f)
        # Seed new droplets at inlet (random y, z)
        new_y = torch.randint(0, ny, (n_droplets_per_step,), device=dev).float()  # full y (physical)
        new_z = torch.randint(0, nz, (n_droplets_per_step,), device=dev).float()
        new_x = torch.zeros(n_droplets_per_step, device=dev)
        droplets.append(torch.stack([new_x, new_y, new_z], dim=1))
        all_d = torch.cat(droplets, dim=0)
        # Advect (3D, droplet inertia)
        xi = all_d[:, 0].long().clamp(0, nx - 1)
        yi = all_d[:, 1].long().clamp(0, ny - 1)
        zi = all_d[:, 2].long().clamp(0, nz - 1)
        # Droplet velocity: partial flow follow (Stokes St=0.155)
        # u_d = St_eff·u_in + (1-St_eff)·u_flow, St_eff = St/(1+St) = 0.134
        st_eff = 0.134
        all_d[:, 0] += st_eff * u_in + (1.0 - st_eff) * ux[zi, yi, xi]
        all_d[:, 1] += (1.0 - st_eff) * uy[zi, yi, xi]
        # Collision (3D): droplet in fluid adjacent to ice → freeze
        xi2 = all_d[:, 0].long().clamp(0, nx - 1)
        yi2 = all_d[:, 1].long().clamp(0, ny - 1)
        zi2 = all_d[:, 2].long().clamp(0, nz - 1)
        nbr_ice = torch.zeros_like(ice_mask)
        for _ax, _sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
            nbr_ice |= (torch.roll(ice_mask, _sgn, dims=_ax) & ~ice_mask)
        # Droplet hits ice (entered solid) → freeze at upstream fluid cell
        in_ice = ice_mask[zi2, yi2, xi2]
        hit = in_ice
        if hit.any():
            hit_xi = xi2[hit]
            hit_yi = yi2[hit]
            hit_zi = zi2[hit]
            # Freeze at upstream cell (xi-1, likely fluid)
            fx_x = (hit_xi - 1).clamp(0, nx - 1)
            fx_y = hit_yi
            fx_z = hit_zi
            fluid = ~ice_mask[fx_z, fx_y, fx_x]
            fx_x = fx_x[fluid]
            fx_y = fx_y[fluid]
            fx_z = fx_z[fluid]
            if len(fx_x) > 0:
                collision_count[fx_z, fx_y, fx_x] += 1
                freeze_now = collision_count >= ice_threshold
                new_ice = freeze_now & ~ice_mask
                if new_ice.any():
                    ice_mask = ice_mask | new_ice
                    collision_count[new_ice] = 0
        in_bounds = ((all_d[:, 0] >= 0) & (all_d[:, 0] < nx) &
                     (all_d[:, 1] >= 0) & (all_d[:, 1] < ny) &
                     (all_d[:, 2] >= 0) & (all_d[:, 2] < nz))
        keep = ~hit & in_bounds
        all_d = all_d[keep]
        droplets = [all_d] if len(all_d) > 0 else []
        solid = ice_mask  # update solid for bounce-back

        if step % log_every == 0 or step == steps:
            rho, ux, uy, uz = macroscopic3d(f)
            # Force on airfoil (momentum exchange: f_pre - f_post on solid)
            c = C.to(dev).float()
            # Only surface solid cells (adjacent to fluid) contribute to force
            fluid = ~solid
            surf = torch.zeros_like(solid)
            for _ax, _sgn in [(2, 1), (2, -1), (1, 1), (1, -1), (0, 1), (0, -1)]:
                surf |= (solid & torch.roll(fluid, _sgn, dims=_ax))
            df = (f_pre - f) * surf.unsqueeze(0).float()
            fx = float((df * c[:, 0].view(19, 1, 1, 1)).sum().item())
            fy = -float((df * c[:, 1].view(19, 1, 1, 1)).sum().item())  # y-up: flip sign
            rho_ref = 1.0
            q = 0.5 * rho_ref * u_in ** 2
            chord = nx * chord_frac
            cd = fx / (q * chord * nz)
            cl = fy / (q * chord * nz)
            u_max = float(ux.abs().max().item())
            n_ice = int(ice_mask.sum().item()) - n_solid
            n_droplets = len(all_d)
            print(f"  {step:6d} {u_max:8.4f} {float(rho.min().item()):8.3f} "
                  f"{float(rho.max().item()):8.3f} {cd:8.4f} {cl:8.4f} "
                  f"ice={n_ice} drops={n_droplets}", flush=True)
            history.append({"step": step, "cd": cd, "cl": cl, "u_max": u_max})

    return {"f": f, "solid": solid, "ice_mask": ice_mask,
            "original_solid": original_solid, "history": history,
            "nx": nx, "ny": ny, "nz": nz, "u_in": u_in, "chord": nx * chord_frac}


def save_plot(result, out_path="outputs/aircraft_icing_flow.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available)")
        return
    f = result["f"]
    solid = result["solid"]
    rho, ux, uy, _ = macroscopic3d(f)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(ux[0].detach().cpu().numpy(), origin="lower", cmap="RdBu_r",
                   extent=[0, result["nx"], 0, result["ny"]], vmin=-0.1, vmax=0.15)
    ax.contour(solid[0].detach().cpu().numpy().T, levels=[0.5], colors="k", linewidths=2,
               extent=[0, result["nx"], 0, result["ny"]])
    ax.set_title(f"u_x  (NACA 0012, AoA={4}°)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)
    ax = axes[1]
    spd = torch.sqrt(ux ** 2 + uy ** 2)[0].detach().cpu().numpy()
    im = ax.imshow(spd, origin="lower", cmap="viridis",
                   extent=[0, result["nx"], 0, result["ny"]])
    ax.contour(solid[0].detach().cpu().numpy().T, levels=[0.5], colors="w", linewidths=2,
               extent=[0, result["nx"], 0, result["ny"]])
    ax.set_title("speed")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.colorbar(im, ax=ax)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"  Plot saved: {p}")


def main():
    p = argparse.ArgumentParser(description="Aircraft icing — NACA 0012 (Phase 1: flow)")
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--ny", type=int, default=100)
    p.add_argument("--nz", type=int, default=1)
    p.add_argument("--u-in", type=float, default=0.06)
    p.add_argument("--tau", type=float, default=0.55)
    p.add_argument("--chord-frac", type=float, default=0.4)
    p.add_argument("--aoa", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--output", default="outputs/aircraft_icing_flow.png")
    args = p.parse_args()
    print("=" * 64)
    print("  AIRCRAFT ICING — NACA 0012 (Phase 1: flow field)")
    print("=" * 64)
    r = run_aircraft_icing(nx=args.nx, ny=args.ny, nz=args.nz, u_in=args.u_in,
                           tau=args.tau, chord_frac=args.chord_frac, aoa_deg=args.aoa,
                           steps=args.steps, device=args.device, log_every=args.log_every)
    save_plot(r, args.output)
    h = r["history"]
    if h:
        cd_final = h[-1]["cd"]
        cl_final = h[-1]["cl"]
        print(f"\n  Final: cd={cd_final:.4f}  cl={cl_final:.4f}")
        # NACA 0012 AoA=4° reference: cl ≈ 0.4 (thin airfoil: cl=2π·sin(4°)=0.44)
        cl_ref = 2 * math.pi * math.sin(math.radians(args.aoa))
        print(f"  Reference (thin airfoil): cl={cl_ref:.4f}")
        err = abs(cl_final - cl_ref) / cl_ref * 100
        print(f"  cl error: {err:.1f}%")


if __name__ == "__main__":
    main()
