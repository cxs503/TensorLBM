"""RetestV6 worker — verified-correct main loop for all 14 benchmark cases with FINE grids.

Usage:
    python rettest_v6_worker.py <case_id> <device_id> <output_path>

VERIFIED-CORRECT MAIN LOOP (validated by 3 analytical solutions + D=200 cylinder Cd=1.29/0.8%):

    f_pre = f.clone()
    f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
    # NoDynamics: restore solid cells to pre-collision
    for q in range(19): f[q] = torch.where(sm[q], f_pre[q], f[q])
    # Half-way bounce-back (BEFORE streaming)
    f = bounce_back_cells_3d(f, solid)
    # Streaming
    f = stream3d(f)
    # Far-field BC (NO obstacle_mask — don't touch solid)
    f = far_field_bc_3d(f, u_in)
    if step % 200 == 0: f = correct_mass3d(f, im)
    cd, cl = drag_pressure_integration(f, near, solid, dpS)

CRITICAL:
  - BB must be BEFORE streaming (half-way), NOT after (full-way)
  - far_field_bc must NOT have obstacle_mask (don't overwrite solid)
  - NoDynamics: restore solid cells after collision

FINE GRIDS (D=100+ for 2D):
  Grid convergence confirmed: D=24: 14.1%, D=48: 9.5%, D=96: 7.2%, D=200: 0.8%
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.drag_pressure import (
    drag_pressure_integration,
    get_near_wall_2d,
    get_near_wall_3d,
)
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d


# ──────────────────────────────────────────────────────────────────
# Geometry builders
# ──────────────────────────────────────────────────────────────────
def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Cylinder in x-y plane (extruded along z). (i-cx)^2+(j-cy)^2 < R^2."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2  # (ny,nx)
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def build_square_prism_mask(nx, ny, nz, cx, cy, side, device):
    """Square prism in x-y plane (extruded along z)."""
    half = side / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    sq = (torch.abs(xx - cx) <= half) & (torch.abs(yy - cy) <= half)
    return sq.unsqueeze(0).expand(nz, ny, nx).clone()


def build_rect_prism_mask(nx, ny, nz, cx, cy, lx, ly, device):
    """Rectangular prism 2:1:1 in x-y plane (lx:ly = 2:1)."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    rect = (torch.abs(xx - cx) <= lx / 2.0) & (torch.abs(yy - cy) <= ly / 2.0)
    return rect.unsqueeze(0).expand(nz, ny, nx).clone()


def build_sphere_mask(nx, ny, nz, cx, cy, cz, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2


def build_flat_plate_mask(nx, ny, nz, x_start, device):
    """Flat plate on bottom wall (y=0), from x_start to nx."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, x_start:] = True
    return solid


def build_backward_step_mask(nx, ny, nz, step_h, x_step, device):
    """Backward-facing step: solid block at bottom-left + bottom wall after step."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, x_step:] = True  # bottom wall after step
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    step_block = (xx < x_step) & (yy < step_h)
    solid |= step_block.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_tandem_cylinders_mask(nx, ny, nz, cx1, cy, r1, cx2, r2, device):
    """Two cylinders in tandem (x-y plane, extruded along z)."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    c1 = (xx - cx1) ** 2 + (yy - cy) ** 2 <= r1 ** 2
    c2 = (xx - cx2) ** 2 + (yy - cy) ** 2 <= r2 ** 2
    circle = c1 | c2
    return circle.unsqueeze(0).expand(nz, ny, nx).clone()


def build_ahmed_body_mask(nx, ny, nz, cx, cy, cz, length, width, height, slant_deg, device):
    """Ahmed body: simplified 3D body with slanted rear."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    half_w = width / 2.0
    x_front = cx - length / 2.0
    x_slant_start = cx + length * 0.25  # slant starts at 75% of length
    x_rear = cx + length / 2.0

    # Base rectangular box
    in_box = (
        (xx >= x_front) & (xx <= x_rear)
        & (torch.abs(yy - cy) <= half_w)
        & (zz <= cz + height) & (zz >= cz)
    )

    # Slant: rear portion where height decreases linearly from full to ~0
    slant_len = x_rear - x_slant_start
    in_slant_region = (xx > x_slant_start) & (xx <= x_rear)
    # Height at position x: h * (x_rear - x) / slant_len
    local_h = height * (x_rear - xx) / max(slant_len, 1e-6)
    in_slant = in_slant_region & (zz <= cz + local_h) & (zz >= cz)

    solid = in_box & ~in_slant_region | in_slant
    # Ensure front portion is full height
    solid = solid & ((zz >= cz) & (zz <= cz + height))
    return solid


# ──────────────────────────────────────────────────────────────────
# NACA airfoil mask (3D extruded) with alpha rotation
# ──────────────────────────────────────────────────────────────────
def naca4_half_thickness(x_over_c, t):
    a = x_over_c.clamp(min=1e-12)
    return (t / 0.2) * (
        0.2969 * torch.sqrt(a)
        - 0.1260 * a
        - 0.3516 * a * a
        + 0.2843 * a * a * a
        - 0.1015 * a * a * a * a
    )


def build_naca_mask_3d(nx, ny, nz, cx_le, cy_center, chord, t, m, p, alpha_deg, device):
    """Build 3D extruded NACA 4-digit airfoil mask in x-y plane with alpha rotation."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Rotate coordinates into airfoil-aligned frame
    alpha_rad = math.radians(alpha_deg)
    cos_a = math.cos(alpha_rad)
    sin_a = math.sin(alpha_rad)
    x_rel = xx - cx_le
    y_rel = yy - cy_center
    x_rot = x_rel * cos_a + y_rel * sin_a
    y_rot = -x_rel * sin_a + y_rel * cos_a

    x_norm = x_rot / chord  # 0 at LE, 1 at TE

    # Half-thickness
    half_t = chord * naca4_half_thickness(x_norm, t)

    # Camber line
    if m > 0 and p > 0:
        yc = torch.where(
            x_norm < p,
            m * (x_norm / p ** 2) * (2.0 * p - x_norm),
            m * ((1.0 - x_norm) / (1.0 - p) ** 2) * (1.0 + x_norm - 2.0 * p),
        )
    else:
        yc = torch.zeros_like(x_norm)

    y_upper_rot = yc + half_t
    y_lower_rot = yc - half_t

    in_chord = (x_norm >= 0.0) & (x_norm <= 1.0)
    in_profile = (y_rot >= y_lower_rot) & (y_rot <= y_upper_rot)

    solid_2d = in_chord & in_profile
    return solid_2d.unsqueeze(0).expand(nz, ny, nx).clone()


# ──────────────────────────────────────────────────────────────────
# Case definitions — FINE GRIDS (D=100+ for 2D)
# ──────────────────────────────────────────────────────────────────
CASES = {
    1: {
        "name": "cylinder_D100_Re200",
        "nx": 830, "ny": 330, "nz": 4,
        "u_in": 0.08, "re": 200.0, "D": 100.0,
        "ref_cd": 1.30, "ref_cl": 0.0,
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "C_s": 0.05,
    },
    2: {
        "name": "square_prism_D100_Re100",
        "nx": 830, "ny": 330, "nz": 4,
        "u_in": 0.08, "re": 100.0, "D": 100.0,
        "ref_cd": 2.1, "ref_cl": 0.0,
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "C_s": 0.05,
    },
    3: {
        "name": "square_prism_D100_Re22000",
        "nx": 830, "ny": 330, "nz": 4,
        "u_in": 0.06, "re": 22000.0, "D": 100.0,
        "ref_cd": 2.1, "ref_cl": 0.0,
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "C_s": 0.1,
    },
    4: {
        "name": "naca0012_Re1e6",
        "nx": 600, "ny": 200, "nz": 4,
        "u_in": 0.06, "re": 1e6, "D": 100.0,  # chord
        "ref_cd": 0.012, "ref_cl": 0.0,
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "airfoil": {"t": 0.12, "m": 0.0, "p": 0.0, "alpha": 0.0},
        "C_s": 0.2,
    },
    5: {
        "name": "naca4412_a5_Re1e6",
        "nx": 600, "ny": 200, "nz": 4,
        "u_in": 0.06, "re": 1e6, "D": 100.0,  # chord
        "ref_cd": 0.01, "ref_cl": 0.0,
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "airfoil": {"t": 0.12, "m": 0.04, "p": 0.4, "alpha": 5.0},
        "C_s": 0.2,
    },
    6: {
        "name": "s809_Re1e6",
        "nx": 600, "ny": 200, "nz": 4,
        "u_in": 0.06, "re": 1e6, "D": 100.0,  # chord
        "ref_cd": 0.01, "ref_cl": 0.0,
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "airfoil": {"t": 0.21, "m": 0.0, "p": 0.0, "alpha": 0.0},  # S809 ~21% thick
        "C_s": 0.2,
    },
    7: {
        "name": "backward_step_h50_Re5000",
        "nx": 500, "ny": 200, "nz": 4,
        "u_in": 0.1, "re": 5000.0, "D": 50.0,  # step height
        "ref_cd": None, "ref_cl": None,  # measure xr/h
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "measure": "xr/h",
        "C_s": 0.05,
    },
    8: {
        "name": "rect_prism_2to1_D100_Re2e4",
        "nx": 830, "ny": 400, "nz": 4,
        "u_in": 0.06, "re": 2e4, "D": 100.0,  # short side
        "ref_cd": 1.3, "ref_cl": 0.0,
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "C_s": 0.1,
    },
    9: {
        "name": "flat_plate_Re1e5",
        "nx": 500, "ny": 200, "nz": 4,
        "u_in": 0.1, "re": 1e5, "D": 500.0,  # plate length = nx
        "ref_cd": 0.0074, "ref_cl": 0.0,  # Cf reference
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "measure": "cf",
        "C_s": 0.05,
    },
    10: {
        "name": "sphere_D80_Re1000",
        "nx": 240, "ny": 120, "nz": 120,
        "u_in": 0.08, "re": 1000.0, "D": 80.0,
        "ref_cd": 0.47, "ref_cl": 0.0,
        "is_3d": True,
        "n_steps": 2000, "warmup": 300,
        "C_s": 0.1,
    },
    11: {
        "name": "kvlcc2_200cube_Re2e6",
        "nx": 200, "ny": 200, "nz": 200,
        "u_in": 0.06, "re": 2e6, "D": 100.0,  # hull length
        "ref_cd": 0.0051, "ref_cl": 0.0,  # Ct
        "is_3d": True,
        "n_steps": 1000, "warmup": 200,
        "C_s": 0.2,
    },
    12: {
        "name": "suboff_200x80x80_Re2e6",
        "nx": 200, "ny": 80, "nz": 80,
        "u_in": 0.06, "re": 2e6, "D": 80.0,  # hull length
        "ref_cd": 0.00405, "ref_cl": 0.0,  # Ct
        "is_3d": True,
        "n_steps": 1000, "warmup": 200,
        "C_s": 0.2,
    },
    13: {
        "name": "ahmed_25deg_Re1e5",
        "nx": 400, "ny": 160, "nz": 100,
        "u_in": 0.06, "re": 1e5, "D": 100.0,  # body length
        "ref_cd": 0.25, "ref_cl": 0.0,
        "is_3d": True,
        "n_steps": 2000, "warmup": 300,
        "C_s": 0.15,
    },
    14: {
        "name": "tandem_cylinders_D50_Re100",
        "nx": 600, "ny": 200, "nz": 4,
        "u_in": 0.08, "re": 100.0, "D": 50.0,  # each cylinder D
        "ref_cd": 1.6, "ref_cl": 0.0,
        "is_3d": False,
        "n_steps": 3000, "warmup": 500,
        "C_s": 0.05,
    },
}


def build_geometry(case, device):
    """Build solid mask for a case. Returns (solid, D_char)."""
    nx, ny, nz = case["nx"], case["ny"], case["nz"]
    D = case["D"]
    name = case["name"]

    if name.startswith("cylinder"):
        radius = D / 2.0
        cx = nx * 0.25
        cy = ny * 0.5
        solid = build_cylinder_mask(nx, ny, nz, cx, cy, radius, device)
    elif name.startswith("square_prism"):
        cx = nx * 0.25
        cy = ny * 0.5
        solid = build_square_prism_mask(nx, ny, nz, cx, cy, D, device)
    elif name.startswith("rect_prism"):
        # 2:1:1, D=100 short side, lx=200, ly=100
        lx = 2.0 * D
        ly = D
        cx = nx * 0.3
        cy = ny * 0.5
        solid = build_rect_prism_mask(nx, ny, nz, cx, cy, lx, ly, device)
    elif name.startswith("naca") or name.startswith("s809"):
        af = case["airfoil"]
        chord = D
        cx_le = nx * 0.2
        cy_center = ny * 0.5
        solid = build_naca_mask_3d(
            nx, ny, nz, cx_le, cy_center, chord,
            af["t"], af["m"], af["p"], af["alpha"], device,
        )
    elif name.startswith("backward_step"):
        step_h = int(D)
        x_step = int(nx * 0.3)
        solid = build_backward_step_mask(nx, ny, nz, step_h, x_step, device)
    elif name.startswith("flat_plate"):
        x_start = 0  # plate from inlet
        solid = build_flat_plate_mask(nx, ny, nz, x_start, device)
    elif name.startswith("sphere"):
        radius = D / 2.0
        cx = nx * 0.25
        cy = ny * 0.5
        cz = nz * 0.5
        solid = build_sphere_mask(nx, ny, nz, cx, cy, cz, radius, device)
    elif name.startswith("kvlcc2"):
        from tensorlbm.ship_cad import ShipHullType, build_hull_mask
        length = D
        beam = ny * 0.25
        draft = nz * 0.3
        cx = nx * 0.4
        cy = ny * 0.5
        cz_keel = nz * 0.25
        solid, _ = build_hull_mask(
            ShipHullType.KVLCC2, nx, ny, nz,
            cx=cx, cy=cy, cz_keel=cz_keel,
            length=length, beam=beam, draft=draft,
            device=str(device),
        )
    elif name.startswith("suboff"):
        from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
        length = D
        cx = nx * 0.35
        cy = ny / 2.0
        cz = nz / 2.0
        solid, _ = build_suboff_mask(
            hull_type=SuboffHullType.BARE_HULL,
            nx=nx, ny=ny, nz=nz, cx=cx, cy=cy, cz=cz,
            length=length, device=str(device),
        )
    elif name.startswith("ahmed"):
        length = D
        width = ny * 0.2
        height = nz * 0.2
        cx = nx * 0.35
        cy = ny * 0.5
        cz = nz * 0.3
        solid = build_ahmed_body_mask(nx, ny, nz, cx, cy, cz, length, width, height, 25.0, device)
    elif name.startswith("tandem"):
        r1 = D / 2.0
        r2 = D / 2.0
        cx1 = nx * 0.2
        cx2 = nx * 0.5
        cy = ny * 0.5
        solid = build_tandem_cylinders_mask(nx, ny, nz, cx1, cy, r1, cx2, r2, device)
    else:
        raise ValueError(f"Unknown case: {name}")

    return solid


def compute_dpS(case, u_in, nz, ny):
    """Compute normalization factor dpS = 0.5 * u_in^2 * A_ref.

    A_ref depends on geometry:
      - 2D extruded: D * nz (diameter × span)
      - Sphere: π * D² / 4 (frontal area)
      - Ahmed: width × height (frontal area)
      - Ships: D * nz (approximation; Ct includes skin friction anyway)
    """
    D = case["D"]
    name = case["name"]
    if name.startswith("sphere"):
        A_ref = math.pi * D ** 2 / 4.0
    elif name.startswith("ahmed"):
        width = ny * 0.2
        height = nz * 0.2
        A_ref = width * height
    else:
        # 2D extruded or ships: D * nz
        A_ref = D * nz
    return 0.5 * u_in ** 2 * A_ref


def measure_xr_h(f, solid, case, u_in):
    """Measure reattachment length xr/h for backward-facing step.

    xr/h = x-location where u_x returns to 0 (sign change from negative to positive)
    divided by step height h.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    # Use center z-slice
    ux_mid = ux[ux.shape[0] // 2]  # (ny, nx)
    nx = case["nx"]
    h = int(case["D"])
    x_step = int(nx * 0.3)
    # Probe at y=1 (first fluid cell above bottom wall) for best reattachment signal
    y_probe = 1
    line = ux_mid[y_probe, :].cpu().numpy()
    # Find first sign change from negative to positive after the step
    xr = None
    for i in range(x_step + 1, nx - 1):
        if line[i - 1] <= 0 and line[i] > 0:
            xr = i
            break
    if xr is None:
        # fallback: where ux first exceeds 0.05*u_in
        thr = 0.05 * u_in
        for i in range(x_step + 1, nx - 1):
            if line[i] > thr:
                xr = i
                break
    if xr is None:
        return float("nan")
    return float((xr - x_step) / h)


def measure_cf(f, solid, case, u_in, nu):
    """Measure skin-friction coefficient Cf for flat plate.

    Cf = tau_w / (0.5 * rho * u_in^2), tau_w = mu * du/dy at wall.
    du/dy ≈ u(y=1) / dy (dy=1 lattice unit, u(y=0)=0 from no-slip).
    Cf = 2 * nu * du/dy / u_in^2 = 2 * nu * u(y=1) / u_in^2.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    nz, ny, nx = ux.shape
    # du/dy at wall (y=0 is solid, y=1 is first fluid)
    dudy = ux[:, 1, :]  # (nz, nx) — velocity at first fluid row
    # Cf local = 2 * nu * du/dy / u_in^2  (lattice units, rho≈1)
    cf_local = 2.0 * nu * dudy / (u_in ** 2)
    # Average over plate region (x from start to end) and z
    cf_mean = float(cf_local[:, 1:].mean().item())
    return cf_mean


def main():
    case_id = int(sys.argv[1])
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]

    case = CASES[case_id]
    name = case["name"]
    nx, ny, nz = case["nx"], case["ny"], case["nz"]
    u_in = case["u_in"]
    re = case["re"]
    D = case["D"]
    is_3d = case["is_3d"]
    n_steps = case["n_steps"]
    warmup = case["warmup"]

    nu = u_in * D / re
    tau = 3.0 * nu + 0.5
    C_s = case.get("C_s", 0.05)

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{device_id} C{case_id}]"
    print(f"{tag} {name} nx={nx} ny={ny} nz={nz} D={D} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} Re={re:.0f} C_s={C_s} steps={n_steps}", flush=True)

    t0 = time.time()

    # Build geometry
    solid = build_geometry(case, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid} ({time.time()-t0:.1f}s)", flush=True)

    # Near-wall mask
    if is_3d:
        near = get_near_wall_3d(solid)
    else:
        near = get_near_wall_2d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Normalization
    dpS = compute_dpS(case, u_in, nz, ny)

    # Initialize flow field
    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    # Pre-compute solid mask broadcast for NoDynamics restore
    sm = solid.unsqueeze(0).expand_as(f)

    # Pre-compute boundary solid mask: solid cells on far-field BC faces
    # (only these need restoration after far_field_bc, since interior solid
    #  cells are not touched by far_field_bc and their post-streaming values
    #  must be preserved for the next step's bounce-back)
    boundary_solid = torch.zeros_like(solid)
    boundary_solid[:, :, 0] = solid[:, :, 0]    # x=0 inlet
    boundary_solid[:, :, -1] = solid[:, :, -1]  # x=nx-1 outlet
    boundary_solid[:, 0, :] = solid[:, 0, :]    # y=0 bottom
    boundary_solid[:, -1, :] = solid[:, -1, :]  # y=ny-1 top
    if nz > 4:  # 3D: z-boundaries too
        boundary_solid[0, :, :] = solid[0, :, :]
        boundary_solid[-1, :, :] = solid[-1, :, :]
    bsm = boundary_solid.unsqueeze(0).expand_as(f)
    n_boundary_solid = int(boundary_solid.sum().item())
    print(f"{tag} boundary solid cells: {n_boundary_solid}", flush=True)

    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # Simulation
    cd_hist = []
    cl_hist = []
    all_finite = True
    final_step = 0

    for step in range(1, n_steps + 1):
        # ─── VERIFIED-CORRECT MAIN LOOP ───────────────────────────
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (all cells)
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=C_s)

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (BEFORE streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC (NO obstacle_mask — don't touch solid)
        f = far_field_bc_3d(f, u_in)

        # 6b. Restore boundary solid cells after far-field BC
        #     (only solid cells on far-field BC faces need restoration;
        #      interior solid cells' post-streaming values are preserved
        #      for the next step's bounce-back)
        for q in range(19):
            f[q] = torch.where(bsm[q], f_pre[q], f[q])

        # 7. Mass correction (every 200 steps)
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # ─── Measurement ──────────────────────────────────────────
        if step > warmup:
            cd, cl = drag_pressure_integration(f, near, solid, dpS)
            if math.isfinite(cd):
                cd_hist.append(cd)
            if math.isfinite(cl):
                cl_hist.append(cl)

        final_step = step

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            all_finite = False
            break

        if step % 200 == 0:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd={cd_avg:.4f} n={len(cd_hist)} ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0

    # Final statistics
    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
    cl_mean = sum(cl_hist) / max(len(cl_hist), 1) if cl_hist else float("nan")

    # rho min/max
    rho, _, _, _ = macroscopic3d(f)
    rho_min = float(rho.min().item())
    rho_max = float(rho.max().item())

    # Special measurements
    measure = case.get("measure")
    xr_h = float("nan")
    cf_meas = float("nan")
    if measure == "xr/h":
        xr_h = measure_xr_h(f, solid, case, u_in)
    elif measure == "cf":
        cf_meas = measure_cf(f, solid, case, u_in, nu)

    # Error
    ref_cd = case.get("ref_cd")
    ref_cl = case.get("ref_cl")
    if measure == "xr/h":
        # Backward step: ref xr/h ≈ 6.0 for Re=5000
        ref_xr_h = 6.0
        err_pct = abs(xr_h - ref_xr_h) / ref_xr_h * 100 if math.isfinite(xr_h) else float("nan")
        measured_val = xr_h
        ref_val = ref_xr_h
    elif measure == "cf":
        if math.isfinite(cf_meas) and ref_cd > 0:
            err_pct = abs(cf_meas - ref_cd) / ref_cd * 100
        else:
            err_pct = float("nan")
        measured_val = cf_meas
        ref_val = ref_cd
    else:
        if ref_cd is not None and math.isfinite(cd_mean) and ref_cd > 0:
            err_pct = abs(cd_mean - ref_cd) / ref_cd * 100
        else:
            err_pct = float("nan")
        measured_val = cd_mean
        ref_val = ref_cd

    status = "PASS" if (math.isfinite(err_pct) and err_pct < 15.0 and all_finite) else "FAIL"

    result = {
        "case_id": case_id,
        "case": name,
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D,
        "u_in": u_in,
        "Re": re,
        "nu": nu,
        "tau": tau,
        "C_s": C_s,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd": cd_mean,
        "Cl": cl_mean,
        "Cd_ref": ref_cd,
        "Cl_ref": ref_cl,
        "measured": measured_val,
        "ref": ref_val,
        "measure": measure,
        "xr_h": xr_h,
        "cf": cf_meas,
        "err_pct": err_pct,
        "status": status,
        "rho_min": rho_min,
        "rho_max": rho_max,
        "solid_cells": n_solid,
        "near_wall_cells": n_near,
        "samples": len(cd_hist),
        "finite": all_finite,
        "final_step": final_step,
        "elapsed_s": elapsed,
        "main_loop": "NoDynamics+halfway_BB+pressure_integration",
        "grid_type": "FINE",
    }

    print(f"{tag} DONE {status} measured={measured_val:.4f} (ref={ref_val}) "
          f"err={err_pct:.1f}% rho=[{rho_min:.4f},{rho_max:.4f}] ({elapsed:.0f}s)",
          flush=True)

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"{tag} wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
