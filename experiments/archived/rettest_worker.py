#!/usr/bin/env python3
"""Bounce-back + momentum exchange (Ladd 1994) benchmark runner.

Runs all 14 benchmark cases with D3Q19 MRT+Smag Cs=0.05, bounce-back
(NO wall_function), far_field_bc_3d, correct_mass3d every 200 steps,
2000 steps, sliding window 300.

Usage:
    PYTHONPATH=src python rettest_worker.py <case_name> <device_id> <output_path>
"""
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import far_field_bc_3d, sphere_mask, bounce_back_cells_3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d

# ─── D3Q19 opposite-pair list (skip rest=0) ───
# Pairs: (1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18)
_OPP_PAIRS = []
for _i in range(1, 19):
    _j = int(OPPOSITE[_i].item())
    if _j > _i:
        _OPP_PAIRS.append((_i, _j))
assert len(_OPP_PAIRS) == 9


# ════════════════════════════════════════════════════════════════════
# NEAR-WALL DETECTION
# ════════════════════════════════════════════════════════════════════

def get_near_2d(solid):
    """2D per-layer near-wall detection (no periodic wrap in any direction).

    For each z-slice, find fluid cells adjacent to solid in x and y only.
    Correct for extruded 2D geometries (cylinder, airfoil, prism, step, plate).
    """
    nz, ny, nx = solid.shape
    fluid = ~solid
    near = torch.zeros_like(solid)
    for z in range(nz):
        s = solid[z]
        f = fluid[z]
        n = torch.zeros_like(s)
        n[:, 1:-1] |= (s[:, 2:] | s[:, :-2]) & f[:, 1:-1]
        n[1:-1, :] |= (s[2:, :] | s[:-2, :]) & f[1:-1, :]
        near[z] = n
    return near


def get_near_3d(solid):
    """3D near-wall detection (no periodic wrap in any direction).

    Finds fluid cells adjacent to solid in all 6 face directions.
    Correct for fully 3D geometries (sphere, ships, Ahmed body).
    """
    fluid = ~solid
    near = torch.zeros_like(solid)
    # x-direction
    near[:, :, 1:-1] |= (solid[:, :, 2:] | solid[:, :, :-2]) & fluid[:, :, 1:-1]
    # y-direction
    near[:, 1:-1, :] |= (solid[:, 2:, :] | solid[:, :-2, :]) & fluid[:, 1:-1, :]
    # z-direction
    near[1:-1] |= (solid[2:] | solid[:-2]) & fluid[1:-1]
    return near


# ════════════════════════════════════════════════════════════════════
# BOUNCE-BACK + MOMENTUM EXCHANGE (Ladd 1994)
# ════════════════════════════════════════════════════════════════════

def bb(f, near):
    """Half-way bounce-back at near-wall fluid cells.

    Swaps each opposite pair (i, OPPOSITE[i]) at near-wall cells.
    Uses the correct D3Q19 OPPOSITE mapping for this codebase's lattice ordering.
    """
    mask = near.float()
    for i, j in _OPP_PAIRS:
        sv = f[i].clone()
        f[i] = f[i] * (1 - mask) + f[j] * mask
        f[j] = f[j] * (1 - mask) + sv * mask
    return f


def _shift_3d(tensor, dx, dy, dz):
    """Shift a 3D tensor by (dz, dy, dx). Zeros at exposed boundaries."""
    nz, ny, nx = tensor.shape
    result = torch.zeros_like(tensor)
    # z slices
    if dz > 0:
        sz_src, sz_dst = slice(dz, None), slice(None, -dz)
    elif dz < 0:
        sz_src, sz_dst = slice(None, dz), slice(-dz, None)
    else:
        sz_src = sz_dst = slice(None)
    # y slices
    if dy > 0:
        sy_src, sy_dst = slice(dy, None), slice(None, -dy)
    elif dy < 0:
        sy_src, sy_dst = slice(None, dy), slice(-dy, None)
    else:
        sy_src = sy_dst = slice(None)
    # x slices
    if dx > 0:
        sx_src, sx_dst = slice(dx, None), slice(None, -dx)
    elif dx < 0:
        sx_src, sx_dst = slice(None, dx), slice(-dx, None)
    else:
        sx_src = sx_dst = slice(None)
    result[sz_dst, sy_dst, sx_dst] = tensor[sz_src, sy_src, sx_src]
    return result


def drag_neq(f, near, dpS, device, solid):
    """Non-equilibrium momentum exchange (Ladd 1994).

    F_x = sum over opposite pairs (i, OPPOSITE[i]) at near-wall cells of
    [(f[i] + f[opp]) - (feq[i] + feq[opp])] * cx[i]

    Uses the correct D3Q19 OPPOSITE mapping (not i+9).
    Subtracts equilibrium to isolate the non-equilibrium (friction+pressure)
    contribution.

    Returns Cd = F_x / dpS where dpS = 0.5 * rho * U^2 * S_ref.
    """
    from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d, macroscopic3d
    c = C.to(f.device).float()
    opp = OPPOSITE.to(f.device)
    cx_k = c[:, 0].view(19, 1, 1, 1)
    mask = near.float()
    rho, ux, uy, uz = macroscopic3d(f)
    feq = equilibrium3d(rho, ux, uy, uz, device=f.device)
    dfric = torch.zeros(1, device=f.device)
    for i in range(1, 19):
        opp_i = int(opp[i].item())
        if opp_i > i:
            f_total = (f[i] + f[opp_i]) * cx_k[i] * mask
            feq_total = (feq[i] + feq[opp_i]) * cx_k[i] * mask
            dfric += (f_total - feq_total).sum()
    return float(dfric.item() / dpS)


# ════════════════════════════════════════════════════════════════════
# GEOMETRY BUILDERS
# ════════════════════════════════════════════════════════════════════

def build_cylinder(nx, ny, nz, device, diameter=24.0):
    """Cylinder extruded along z-axis."""
    radius = diameter / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = nx * 0.25, ny * 0.5
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_tandem_cylinders(nx, ny, nz, device, diameter=24.0, spacing_d=4.0):
    """Two cylinders in tandem (along x-axis), extruded along z.

    spacing_d: center-to-center distance in diameters.
    """
    radius = diameter / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cx1 = nx * 0.2
    cx2 = cx1 + spacing_d * diameter
    cy = ny * 0.5
    circle1 = (xx - cx1) ** 2 + (yy - cy) ** 2 <= radius ** 2
    circle2 = (xx - cx2) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = (circle1 | circle2).unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_square_prism(nx, ny, nz, device, side=24.0):
    """Square prism extruded along z-axis."""
    cx, cy = nx * 0.25, ny * 0.5
    half = side / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    rect = (xx >= cx - half) & (xx <= cx + half) & (yy >= cy - half) & (yy <= cy + half)
    solid = rect.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_rect_prism_211(nx, ny, nz, device, D=30.0):
    """Rectangular prism 2:1:1 (length:width:height = 2:1:1), D=width."""
    length = 2.0 * D
    height = D
    cx, cy = nx * 0.3, ny * 0.5
    x_le = cx - length / 2.0
    x_te = cx + length / 2.0
    y_bot = cy - height / 2.0
    y_top = cy + height / 2.0
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    rect = (xx >= x_le) & (xx <= x_te) & (yy >= y_bot) & (yy <= y_top)
    solid = rect.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def naca_half_thickness(x_over_c, t=0.12):
    """NACA 4-digit half-thickness distribution."""
    a = x_over_c.clamp(min=1e-12)
    return (t / 0.2) * (
        0.2969 * torch.sqrt(a) - 0.1260 * a - 0.3516 * a * a
        + 0.2843 * a ** 3 - 0.1015 * a ** 4
    )


def naca_camber(x_over_c, m=0.0, p=0.4):
    """NACA 4-digit camber line."""
    a = x_over_c.clamp(min=1e-12, max=1.0)
    yc = torch.zeros_like(a)
    if m > 0:
        front = a <= p
        back = a > p
        yc[front] = m / (p ** 2) * (2 * p * a[front] - a[front] ** 2)
        yc[back] = m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * a[back] - a[back] ** 2)
    return yc


def build_naca_airfoil(nx, ny, nz, device, chord=200.0, t=0.12, m=0.0, p=0.4):
    """Build NACA 4-digit airfoil mask in x-y plane, extruded along z.

    NACA 0012: t=0.12, m=0
    NACA 4412: t=0.12, m=0.04, p=0.40
    S809 approx: t=0.21, m=0.02, p=0.30
    """
    cx_le = nx * 0.2  # leading edge at 20% from inlet
    cy_center = ny * 0.5

    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Broadcast to (nz, ny, nx)
    yy = yy.unsqueeze(0).expand(nz, ny, nx)
    xx = xx.unsqueeze(0).expand(nz, ny, nx)

    x_norm = (xx - cx_le) / chord
    half_t = chord * naca_half_thickness(x_norm, t)
    camber = chord * naca_camber(x_norm, m, p)

    y_upper = cy_center + camber + half_t
    y_lower = cy_center + camber - half_t

    in_chord = (x_norm >= 0.0) & (x_norm <= 1.0)
    in_profile = (yy >= y_lower) & (yy <= y_upper)
    solid = in_chord & in_profile

    # Close LE and TE
    le_col = int(cx_le)
    te_col = int(cx_le + chord)
    cy_int = int(cy_center)
    solid[:, :, le_col] |= (yy[:, :, le_col] == cy_int)
    if te_col < nx:
        solid[:, :, te_col] |= (yy[:, :, te_col] == cy_int)
    return solid


def build_backward_step(nx, ny, nz, device, step_h=20):
    """Backward-facing step: solid block before the step."""
    x_step = nx // 4  # step at 25% of domain
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, :step_h, :x_step] = True
    # Bottom wall after step
    solid[:, 0, x_step:] = True
    # Top wall (channel)
    solid[:, -1, :] = True
    return solid


def build_flat_plate(nx, ny, nz, device, plate_pct=0.8):
    """Flat plate on bottom (y=0), starting at plate_pct from inlet."""
    x_start = int((1.0 - plate_pct) * nx)
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, x_start:] = True
    return solid


def build_ahmed_body(nx, ny, nz, device, slant_deg=25.0):
    """Simplified Ahmed body with slanted rear.

    Body dimensions (lattice units):
      Length L=200, Width W=80, Height H=60
      Front rounded section: first 30 cells
      Rear slant: last 40 cells, top slopes down at slant_deg
    """
    L = 200.0
    W = 80.0
    H = 60.0
    x_start = 40  # start of body
    x_end = x_start + int(L)
    y_bot = int(ny * 0.25)
    z_front = int((nz - W) / 2)
    z_back = z_front + int(W)

    slant_len = 40
    x_slant_start = x_end - slant_len
    slant_rad = math.radians(slant_deg)

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )

    in_body_x = (xx >= x_start) & (xx < x_end)
    in_body_y = (yy >= y_bot) & (yy < y_bot + H)
    in_body_z = (zz >= z_front) & (zz < z_back)

    # Main rectangular body (before slant)
    main_body = in_body_x & in_body_y & in_body_z & (xx < x_slant_start)

    # Slanted rear: top surface slopes down
    slant_x = (xx - x_slant_start).clamp(min=0)
    slant_drop = slant_x * math.tan(slant_rad)
    y_top_slant = y_bot + H - slant_drop
    slant_body = (xx >= x_slant_start) & (xx < x_end) & in_body_z & \
                 (yy >= y_bot) & (yy < y_top_slant)

    # Rounded front (quarter circle)
    front_r = 30.0
    front_center_x = x_start + front_r
    front_center_y = y_bot + front_r
    front_dist = torch.sqrt((xx - front_center_x) ** 2 + (yy - front_center_y) ** 2)
    front_mask = (xx < x_start + front_r) & (yy >= y_bot) & (yy < y_bot + front_r) & \
                 (front_dist <= front_r) & in_body_z
    # Also fill below the quarter circle
    front_fill = (xx < x_start + front_r) & (yy >= y_bot + front_r) & (yy < y_bot + H) & in_body_z
    front_mask = front_mask | front_fill

    solid = main_body | slant_body | front_mask
    return solid


def build_suboff(nx, ny, nz, device, length=100.0):
    """SUBOFF bare hull (axisymmetric body of revolution)."""
    from tensorlbm.suboff_cad import suboff_hull_mask
    cx = nx * 0.5
    cy = ny * 0.5
    cz = nz * 0.5
    radius = length * 0.05834  # r_over_l
    solid = suboff_hull_mask(nx, ny, nz, cx, cy, cz, length, radius, device)
    return solid


def build_kvlcc2(nx, ny, nz, device, length=100.0):
    """KVLCC2 hull mask."""
    from tensorlbm.ship_cad import build_hull_mask, ShipHullType
    cx = nx * 0.5
    cy = ny * 0.5
    cz_keel = nz * 0.75
    beam = ny * 0.25
    draft = nz * 0.3
    mask, stats = build_hull_mask(
        ShipHullType.KVLCC2, nx, ny, nz,
        cx=cx, cy=cy, cz_keel=cz_keel,
        length=length, beam=beam, draft=draft,
        device=str(device),
    )
    return mask


# ════════════════════════════════════════════════════════════════════
# FAR-FIELD BC WITH ANGLED INFLOW
# ════════════════════════════════════════════════════════════════════

def far_field_angled(f, ux_fs, uy_fs=0.0, uz_fs=0.0):
    """Far-field BC with angled inflow."""
    rho1 = torch.ones((f.shape[1], f.shape[2], f.shape[3]),
                      dtype=f.dtype, device=f.device)
    feq = equilibrium3d(
        rho1,
        torch.full_like(rho1, ux_fs),
        torch.full_like(rho1, uy_fs),
        torch.full_like(rho1, uz_fs),
        device=f.device,
    )
    f[:, :, :, 0] = feq[:, :, :, 0]       # inlet
    f[:, :, :, -1] = f[:, :, :, -2]       # outlet (zero gradient)
    f[:, 0, :, :] = feq[:, 0, :, :]       # y- lateral
    f[:, -1, :, :] = feq[:, -1, :, :]     # y+ lateral
    f[:, :, 0, :] = feq[:, :, 0, :]        # z- lateral
    f[:, :, -1, :] = feq[:, :, -1, :]      # z+ lateral
    return f


def far_field_bc_2d_extruded(f, u_in):
    """Far-field BC for 2D extruded geometries.

    Sets inlet/outlet and y± boundaries to free-stream equilibrium.
    Does NOT set z± boundaries (leaves them periodic from streaming),
    which is correct for 2D extruded cases (cylinder, airfoil, etc.).

    f shape: (19, nz, ny, nx) — dims are (q, z, y, x)
    """
    rho1 = torch.ones((f.shape[1], f.shape[2], f.shape[3]),
                      dtype=f.dtype, device=f.device)
    feq = equilibrium3d(
        rho1, torch.full_like(rho1, u_in),
        torch.zeros_like(rho1), torch.zeros_like(rho1),
        device=f.device,
    )
    f[:, :, :, 0] = feq[:, :, :, 0]       # inlet x=0 (free stream)
    f[:, :, :, -1] = f[:, :, :, -2]       # outlet x=nx-1 (zero gradient)
    f[:, :, 0, :] = feq[:, :, 0, :]       # y- lateral (y=0)
    f[:, :, -1, :] = feq[:, :, -1, :]     # y+ lateral (y=ny-1)
    # z±: NOT set — periodic from streaming (torch.roll)
    return f


def far_field_angled_2d(f, ux_fs, uy_fs=0.0, uz_fs=0.0):
    """Far-field BC with angled inflow for 2D extruded geometries.

    f shape: (19, nz, ny, nx) — dims are (q, z, y, x)
    """
    rho1 = torch.ones((f.shape[1], f.shape[2], f.shape[3]),
                      dtype=f.dtype, device=f.device)
    feq = equilibrium3d(
        rho1,
        torch.full_like(rho1, ux_fs),
        torch.full_like(rho1, uy_fs),
        torch.full_like(rho1, uz_fs),
        device=f.device,
    )
    f[:, :, :, 0] = feq[:, :, :, 0]       # inlet x=0
    f[:, :, :, -1] = f[:, :, :, -2]       # outlet x=nx-1 (zero gradient)
    f[:, :, 0, :] = feq[:, :, 0, :]       # y- lateral (y=0)
    f[:, :, -1, :] = feq[:, :, -1, :]     # y+ lateral (y=ny-1)
    # z±: NOT set — periodic from streaming
    return f


# ════════════════════════════════════════════════════════════════════
# CASE CONFIGURATIONS
# ════════════════════════════════════════════════════════════════════

CASES = {
    "cylinder_Re200": {
        "nx": 200, "ny": 80, "nz": 4,
        "u_in": 0.08, "Re": 200, "ref": 1.30, "ref_name": "Cd",
        "geom": "cylinder", "geom_params": {"diameter": 24.0},
        "near_mode": "2d", "alpha_deg": 0,
        "ref_area_type": "frontal_2d", "ref_dim": 24.0,
    },
    "square_prism_Re100": {
        "nx": 200, "ny": 80, "nz": 4,
        "u_in": 0.08, "Re": 100, "ref": 2.1, "ref_name": "Cd",
        "geom": "square_prism", "geom_params": {"side": 30.0},
        "near_mode": "2d", "alpha_deg": 0,
        "ref_area_type": "frontal_2d", "ref_dim": 30.0,
    },
    "square_prism_Re22000": {
        "nx": 200, "ny": 80, "nz": 4,
        "u_in": 0.08, "Re": 22000, "ref": 2.1, "ref_name": "Cd",
        "geom": "square_prism", "geom_params": {"side": 30.0},
        "near_mode": "2d", "alpha_deg": 0,
        "ref_area_type": "frontal_2d", "ref_dim": 30.0,
    },
    "naca0012_Re6e6": {
        "nx": 300, "ny": 100, "nz": 4,
        "u_in": 0.06, "Re": 6e6, "ref": 0.008, "ref_name": "Cd",
        "geom": "naca", "geom_params": {"chord": 200.0, "t": 0.12, "m": 0.0, "p": 0.4},
        "near_mode": "2d", "alpha_deg": 0,
        "ref_area_type": "chord_span", "ref_dim": 200.0,
    },
    "naca4412_Re3e6": {
        "nx": 300, "ny": 100, "nz": 4,
        "u_in": 0.06, "Re": 3e6, "ref": 0.007, "ref_name": "Cd",
        "geom": "naca", "geom_params": {"chord": 200.0, "t": 0.12, "m": 0.04, "p": 0.40},
        "near_mode": "2d", "alpha_deg": 5.0,
        "ref_area_type": "chord_span", "ref_dim": 200.0,
    },
    "s809_Re2e6": {
        "nx": 300, "ny": 100, "nz": 4,
        "u_in": 0.06, "Re": 2e6, "ref": 0.007, "ref_name": "Cd",
        "geom": "naca", "geom_params": {"chord": 200.0, "t": 0.21, "m": 0.02, "p": 0.30},
        "near_mode": "2d", "alpha_deg": 0,
        "ref_area_type": "chord_span", "ref_dim": 200.0,
    },
    "backward_step_Re5000": {
        "nx": 200, "ny": 80, "nz": 4,
        "u_in": 0.08, "Re": 5000, "ref": 6.0, "ref_name": "xr_h",
        "geom": "backward_step", "geom_params": {"step_h": 20},
        "near_mode": "2d", "alpha_deg": 0,
        "ref_area_type": "none", "ref_dim": 0,
    },
    "rect_prism_Re2e4": {
        "nx": 200, "ny": 120, "nz": 4,
        "u_in": 0.08, "Re": 2e4, "ref": 1.3, "ref_name": "Cd",
        "geom": "rect_prism_211", "geom_params": {"D": 30.0},
        "near_mode": "2d", "alpha_deg": 0,
        "ref_area_type": "frontal_2d", "ref_dim": 30.0,
    },
    "flat_plate_Re2e6": {
        "nx": 200, "ny": 80, "nz": 4,
        "u_in": 0.06, "Re": 2e6, "ref": 0.00405, "ref_name": "Cf",
        "geom": "flat_plate", "geom_params": {"plate_pct": 0.8},
        "near_mode": "2d", "alpha_deg": 0,
        "ref_area_type": "plate_area", "ref_dim": 0,
    },
    "sphere_Re1000": {
        "nx": 120, "ny": 60, "nz": 60,
        "u_in": 0.08, "Re": 1000, "ref": 0.47, "ref_name": "Cd",
        "geom": "sphere", "geom_params": {"diameter": 40.0},
        "near_mode": "3d", "alpha_deg": 0,
        "ref_area_type": "sphere_frontal", "ref_dim": 40.0,
    },
    "kvlcc2_Re2e6": {
        "nx": 200, "ny": 200, "nz": 200,
        "u_in": 0.05, "Re": 2e6, "ref": 0.0051, "ref_name": "Ct",
        "geom": "kvlcc2", "geom_params": {"length": 100.0},
        "near_mode": "3d", "alpha_deg": 0,
        "ref_area_type": "wetted_surface", "ref_dim": 0,
    },
    "suboff_Re2e6": {
        "nx": 200, "ny": 80, "nz": 80,
        "u_in": 0.05, "Re": 2e6, "ref": 0.00405, "ref_name": "Ct",
        "geom": "suboff", "geom_params": {"length": 100.0},
        "near_mode": "3d", "alpha_deg": 0,
        "ref_area_type": "wetted_surface", "ref_dim": 0,
    },
    "ahmed25_Re2e6": {
        "nx": 300, "ny": 120, "nz": 100,
        "u_in": 0.05, "Re": 2e6, "ref": 0.25, "ref_name": "Cd",
        "geom": "ahmed", "geom_params": {"slant_deg": 25.0},
        "near_mode": "3d", "alpha_deg": 0,
        "ref_area_type": "ahmed_frontal", "ref_dim": 0,
    },
    "tandem_cyl_Re100": {
        "nx": 300, "ny": 100, "nz": 4,
        "u_in": 0.08, "Re": 100, "ref": 1.6, "ref_name": "Cd",
        "geom": "tandem_cyl", "geom_params": {"diameter": 24.0, "spacing_d": 4.0},
        "near_mode": "2d", "alpha_deg": 0,
        "ref_area_type": "frontal_2d", "ref_dim": 24.0,
    },
}


# ════════════════════════════════════════════════════════════════════
# GEOMETRY DISPATCH
# ════════════════════════════════════════════════════════════════════

def build_geometry(geom_type, geom_params, nx, ny, nz, device):
    if geom_type == "cylinder":
        return build_cylinder(nx, ny, nz, device, **geom_params)
    elif geom_type == "tandem_cyl":
        return build_tandem_cylinders(nx, ny, nz, device, **geom_params)
    elif geom_type == "square_prism":
        return build_square_prism(nx, ny, nz, device, **geom_params)
    elif geom_type == "naca":
        return build_naca_airfoil(nx, ny, nz, device, **geom_params)
    elif geom_type == "backward_step":
        return build_backward_step(nx, ny, nz, device, **geom_params)
    elif geom_type == "rect_prism_211":
        return build_rect_prism_211(nx, ny, nz, device, **geom_params)
    elif geom_type == "flat_plate":
        return build_flat_plate(nx, ny, nz, device, **geom_params)
    elif geom_type == "sphere":
        diameter = geom_params["diameter"]
        radius = diameter / 2.0
        return sphere_mask(nx, ny, nz, nx * 0.25, ny * 0.5, nz * 0.5, radius, device)
    elif geom_type == "kvlcc2":
        return build_kvlcc2(nx, ny, nz, device, **geom_params)
    elif geom_type == "suboff":
        return build_suboff(nx, ny, nz, device, **geom_params)
    elif geom_type == "ahmed":
        return build_ahmed_body(nx, ny, nz, device, **geom_params)
    else:
        raise ValueError(f"Unknown geometry: {geom_type}")


# ════════════════════════════════════════════════════════════════════
# REFERENCE AREA COMPUTATION
# ════════════════════════════════════════════════════════════════════

def compute_ref_area(ref_type, ref_dim, nx, ny, nz, u_in, solid, near):
    """Compute dpS = 0.5 * rho * U^2 * S_ref for drag normalization."""
    rho = 1.0
    if ref_type == "frontal_2d":
        # D * span (2D extruded)
        S = ref_dim * nz
    elif ref_type == "chord_span":
        S = ref_dim * nz
    elif ref_type == "sphere_frontal":
        R = ref_dim / 2.0
        S = math.pi * R ** 2
    elif ref_type == "plate_area":
        x_start = int((1.0 - 0.8) * nx)
        S = float((nx - x_start) * nz)
    elif ref_type == "wetted_surface":
        # Approximate wetted surface area = number of near-wall cells
        S = float(near.sum().item())
    elif ref_type == "ahmed_frontal":
        # Ahmed body frontal area: W * H
        S = 80.0 * 60.0  # W=80, H=60
    elif ref_type == "none":
        S = 1.0
    else:
        S = 1.0
    return 0.5 * rho * u_in ** 2 * S, S


# ════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════════════════

def run_case(case_name, device_id, output_path):
    cfg = CASES[case_name]
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    u_in = cfg["u_in"]
    re = cfg["Re"]
    cs_smag = 0.05
    n_steps = 2000
    warmup = 500
    window = 300

    # Compute viscosity from Re and characteristic length
    if cfg["geom"] == "cylinder":
        char_len = cfg["geom_params"]["diameter"]
    elif cfg["geom"] == "tandem_cyl":
        char_len = cfg["geom_params"]["diameter"]
    elif cfg["geom"] == "square_prism":
        char_len = cfg["geom_params"]["side"]
    elif cfg["geom"] == "naca":
        char_len = cfg["geom_params"]["chord"]
    elif cfg["geom"] == "rect_prism_211":
        char_len = cfg["geom_params"]["D"]
    elif cfg["geom"] == "sphere":
        char_len = cfg["geom_params"]["diameter"]
    elif cfg["geom"] == "backward_step":
        char_len = cfg["geom_params"]["step_h"]
    elif cfg["geom"] == "flat_plate":
        char_len = float(nx)  # plate length
    elif cfg["geom"] in ("kvlcc2", "suboff"):
        char_len = cfg["geom_params"]["length"]
    elif cfg["geom"] == "ahmed":
        char_len = 200.0  # body length
    else:
        char_len = float(nx)

    nu = u_in * char_len / re
    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{device_id} {case_name}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} u_in={u_in} nu={nu:.6e} tau={tau:.8f} "
          f"Re={re:.0e} Cs={cs_smag} char_len={char_len}", flush=True)

    t0 = time.time()

    # Build geometry
    solid = build_geometry(cfg["geom"], cfg["geom_params"], nx, ny, nz, device)
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}", flush=True)

    # Near-wall detection
    if cfg["near_mode"] == "2d":
        near = get_near_2d(solid)
    else:
        near = get_near_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Reference area
    dpS, S_ref = compute_ref_area(
        cfg["ref_area_type"], cfg["ref_dim"], nx, ny, nz, u_in, solid, near
    )
    print(f"{tag} S_ref={S_ref:.1f} dpS={dpS:.6f}", flush=True)

    # Angle of attack
    alpha_rad = math.radians(cfg["alpha_deg"])
    ux_in = u_in * math.cos(alpha_rad)
    uy_in = u_in * math.sin(alpha_rad)
    uz_in = 0.0

    # Initialize flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), ux_in, device=device)
    uy0 = torch.full((nz, ny, nx), uy_in, device=device)
    uz0 = torch.zeros((nz, ny, nx), device=device)
    ux0[solid] = 0.0
    uy0[solid] = 0.0
    uz0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(rho0.sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s)", flush=True)

    # Sliding window for Cd averaging
    cd_window = deque(maxlen=window)
    diverged = False

    for step in range(1, n_steps + 1):
        # 1. Collision: MRT + Smagorinsky LES
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 2. Bounce-back at solid cells (AFTER collision, BEFORE stream)
        f = bounce_back_cells_3d(f, solid)

        # 3. Stream
        f = stream3d(f)

        # 4. Far-field BC (after stream; no obstacle_mask since BB already done)
        if cfg["alpha_deg"] != 0:
            f = far_field_bc_3d(f, u_in=ux_in, uy=uy_in, uz=uz_in)
        else:
            f = far_field_bc_3d(f, u_in=u_in)

        # 5. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        # 6. Compute drag (non-equilibrium ME, post-stream post-BC)
        if cfg["ref_name"] != "xr_h":
            cd_val = drag_neq(f, near, dpS, device, solid)
        else:
            cd_val = 0.0

        if step > warmup and math.isfinite(cd_val):
            cd_window.append(cd_val)

        # Check for divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            break

        if step % 200 == 0:
            cd_avg = sum(cd_window) / max(len(cd_window), 1) if cd_window else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd={cd_val:.6f} avg={cd_avg:.6f} "
                  f"n={len(cd_window)} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # Final results
    if diverged:
        cd_mean = float("nan")
        cd_std = float("nan")
        status = "DIV"
    elif cfg["ref_name"] == "xr_h":
        # Measure reattachment length for backward step
        rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
        step_h = cfg["geom_params"]["step_h"]
        x_step = nx // 4
        # Find first x where ux at y=1 (first fluid row above bottom) becomes positive
        ux_bottom = ux_f[0, 1, x_step:].cpu().numpy()
        xr_h = 0.0
        for i, u in enumerate(ux_bottom):
            if u > 0:
                xr_h = float(i) / step_h
                break
        cd_mean = xr_h
        cd_std = 0.0
        status = "OK" if math.isfinite(xr_h) else "DIV"
    else:
        cd_mean = sum(cd_window) / max(len(cd_window), 1) if cd_window else float("nan")
        cd_std = (sum((c - cd_mean) ** 2 for c in cd_window) /
                  max(len(cd_window) - 1, 1)) ** 0.5 if len(cd_window) > 1 else 0.0
        status = "OK" if math.isfinite(cd_mean) else "DIV"

    ref = cfg["ref"]
    ref_name = cfg["ref_name"]
    if ref_name == "xr_h":
        err_pct = abs(cd_mean - ref) / ref * 100 if ref > 0 and math.isfinite(cd_mean) else float("nan")
    else:
        err_pct = abs(cd_mean - ref) / ref * 100 if ref > 0 and math.isfinite(cd_mean) else float("nan")

    result = {
        "case": case_name,
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": f"MRT+Smag(Cs={cs_smag})",
        "boundary": "bounce_back+ME(Ladd1994)+farfield",
        "grid": f"{nx}x{ny}x{nz}",
        "Re": re,
        "nu": nu,
        "tau": tau,
        "u_in": u_in,
        "char_len": char_len,
        "n_steps": n_steps,
        "warmup": warmup,
        "window": window,
        "solid_cells": n_solid,
        "near_wall_cells": n_near,
        "S_ref": S_ref,
        "dpS": dpS,
        "alpha_deg": cfg["alpha_deg"],
        f"{ref_name}_mean": cd_mean,
        f"{ref_name}_std": cd_std,
        f"{ref_name}_ref": ref,
        "error_pct": err_pct,
        "status": status,
        "n_samples": len(cd_window),
        "finite": bool(torch.isfinite(f).all().item()) if not diverged else False,
        "elapsed_s": elapsed,
    }

    print(f"{tag} DONE {ref_name}={cd_mean:.6f} (ref={ref}) err={err_pct:.1f}% "
          f"status={status} time={elapsed:.0f}s", flush=True)

    Path(output_path).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    case_name = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3]
    run_case(case_name, device_id, output_path)
