#!/usr/bin/env python3
"""AllBenchs worker: D3Q19 MRT+Smag(Cs=0.05) + wallfn log-law + farfield BC.

Usage:
    python _bench_worker.py <device_id> <case> <output_path>

Cases:
    naca0012_a0      NACA 0012, Re=6e6, α=0°, 300x100x4
    square_prism      Square prism D=30, 200x80x4, Re=22000
    backward_step     Backward step h=20 H=60, 200x80x4, Re=5000
    rect_prism_2_1_1  Rect prism 2:1:1, D=30, 200x120x4, Re=200000
    s809_a0           S809 airfoil, Re=2e6, α=0°, 300x100x4
    ahmed_25deg       Ahmed body 25°, 300x120x100, Re=2e6
    naca4412_a5       NACA 4412, α=5°, Re=3e6, 300x100x4
    sphere_d48        Sphere D=48, 240x120x120, Re=1000
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d, sphere_mask
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d

# ── NACA 4-digit airfoil profile ────────────────────────────────────────────

def _naca_thickness(x_norm: torch.Tensor, t: float) -> torch.Tensor:
    """NACA 4-digit thickness distribution. x_norm in [0,1], t = max thickness ratio."""
    sqrt_x = torch.sqrt(x_norm.clamp(min=0.0))
    return 5.0 * t * (
        0.2969 * sqrt_x
        - 0.1260 * x_norm
        - 0.3516 * x_norm ** 2
        + 0.2843 * x_norm ** 3
        - 0.1015 * x_norm ** 4
    )


def _naca_camber(x_norm: torch.Tensor, m: float, p: float) -> torch.Tensor:
    """NACA 4-digit mean camber line. m = max camber, p = position of max camber."""
    if m == 0.0 or p == 0.0:
        return torch.zeros_like(x_norm)
    yc = torch.zeros_like(x_norm)
    mask_fwd = x_norm <= p
    x_fwd = x_norm[mask_fwd]
    x_aft = x_norm[~mask_fwd]
    if mask_fwd.any():
        yc[mask_fwd] = m / (p * p) * (2.0 * p * x_fwd - x_fwd * x_fwd)
    if (~mask_fwd).any():
        yc[~mask_fwd] = m / ((1.0 - p) ** 2) * (
            (1.0 - 2.0 * p) + 2.0 * p * x_aft - x_aft * x_aft
        )
    return yc


def _naca_camber_slope(x_norm: torch.Tensor, m: float, p: float) -> torch.Tensor:
    """d(yc)/dx for NACA 4-digit."""
    if m == 0.0 or p == 0.0:
        return torch.zeros_like(x_norm)
    dyc = torch.zeros_like(x_norm)
    mask_fwd = x_norm <= p
    x_fwd = x_norm[mask_fwd]
    x_aft = x_norm[~mask_fwd]
    if mask_fwd.any():
        dyc[mask_fwd] = (2.0 * m) / (p * p) * (p - x_fwd)
    if (~mask_fwd).any():
        dyc[~mask_fwd] = (2.0 * m) / ((1.0 - p) ** 2) * (p - x_aft)
    return dyc


def build_naca_mask(
    nx: int,
    ny: int,
    nz: int,
    chord: float,
    aoa_deg: float,
    m: float,
    p: float,
    t: float,
    cx: float,
    cy: float,
    device: torch.device,
) -> torch.Tensor:
    """Build a 3-D NACA airfoil mask extruded along z (nz layers)."""
    aoa = math.radians(aoa_deg)
    cos_a, sin_a = math.cos(aoa), math.sin(aoa)

    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )  # (ny, nx)
    dx = xx - cx
    dy = yy - cy
    # Rotate to airfoil coords
    xr = (dx * cos_a + dy * sin_a) / chord   # normalized chord coord
    yr = (-dx * sin_a + dy * cos_a) / chord  # normalized normal coord

    xclamp = xr.clamp(0.0, 1.0)
    yt = _naca_thickness(xclamp, t)
    yc = _naca_camber(xclamp, m, p)
    theta = torch.atan(_naca_camber_slope(xclamp, m, p))

    upper = yc + yt * torch.cos(theta)
    lower = yc - yt * torch.cos(theta)
    inside = (xr >= 0.0) & (xr <= 1.0) & (yr >= lower) & (yr <= upper)

    # Extrude along z
    solid = inside.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


# ── S809 airfoil (NREL thick laminar-flow, t/c=0.21) ────────────────────────

# Tabulated S809 coordinates (x/c, y/c upper, y/c lower) — 26 stations
_S809_COORDS = [
    (0.00000,  0.00000,  0.00000),
    (0.00126,  0.00578, -0.00578),
    (0.00514,  0.01141, -0.01141),
    (0.01314,  0.01815, -0.01815),
    (0.02514,  0.02582, -0.02508),
    (0.04514,  0.03480, -0.03148),
    (0.07514,  0.04412, -0.03638),
    (0.10014,  0.04912, -0.03938),
    (0.12514,  0.05308, -0.04108),
    (0.15014,  0.05600, -0.04200),
    (0.17514,  0.05812, -0.04240),
    (0.20014,  0.05956, -0.04200),
    (0.22514,  0.06030, -0.04100),
    (0.25014,  0.06044, -0.03960),
    (0.27514,  0.06004, -0.03790),
    (0.30014,  0.05916, -0.03590),
    (0.32514,  0.05786, -0.03360),
    (0.35014,  0.05620, -0.03120),
    (0.37514,  0.05420, -0.02860),
    (0.40014,  0.05192, -0.02592),
    (0.42514,  0.04940, -0.02320),
    (0.45014,  0.04668, -0.02050),
    (0.47514,  0.04378, -0.01790),
    (0.50014,  0.04074, -0.01540),
    (0.55014,  0.03428, -0.01088),
    (0.60014,  0.02748, -0.00688),
    (0.65014,  0.02056, -0.00358),
    (0.70014,  0.01380, -0.00108),
    (0.75014,  0.00746,  0.00060),
    (0.80014,  0.00200,  0.00142),
    (0.85014, -0.00204,  0.00140),
    (0.90014, -0.00440,  0.00080),
    (0.95014, -0.00450,  0.00010),
    (1.00000,  0.00000,  0.00000),
]


def build_s809_mask(
    nx: int,
    ny: int,
    nz: int,
    chord: float,
    aoa_deg: float,
    cx: float,
    cy: float,
    device: torch.device,
) -> torch.Tensor:
    """Build S809 airfoil mask via tabulated coordinate interpolation."""
    aoa = math.radians(aoa_deg)
    cos_a, sin_a = math.cos(aoa), math.sin(aoa)

    # Build interpolation tables on CPU first (small data)
    xs = torch.tensor([c[0] for c in _S809_COORDS], dtype=torch.float32)
    yu = torch.tensor([c[1] for c in _S809_COORDS], dtype=torch.float32)
    yl = torch.tensor([c[2] for c in _S809_COORDS], dtype=torch.float32)

    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    dx = xx - cx
    dy = yy - cy
    xr = (dx * cos_a + dy * sin_a) / chord
    yr = (-dx * sin_a + dy * cos_a) / chord
    xr_clamp = xr.clamp(0.0, 1.0)

    # Linear interpolation of upper/lower surfaces onto the grid
    xs_dev = xs.to(device)
    yu_dev = yu.to(device)
    yl_dev = yl.to(device)

    # Find right bin via searchsorted
    idx = torch.searchsorted(xs_dev, xr_clamp.flatten()).clamp(1, len(xs_dev) - 1)
    x_left = xs_dev[idx - 1]
    x_right = xs_dev[idx]
    yu_left = yu_dev[idx - 1]
    yu_right = yu_dev[idx]
    yl_left = yl_dev[idx - 1]
    yl_right = yl_dev[idx]

    denom = (x_right - x_left).clamp(min=1e-12)
    t = ((xr_clamp.flatten() - x_left) / denom).clamp(0.0, 1.0)
    y_upper = yu_left + t * (yu_right - yu_left)
    y_lower = yl_left + t * (yl_right - yl_left)

    upper = y_upper.reshape(ny, nx)
    lower = y_lower.reshape(ny, nx)

    inside = (xr >= 0.0) & (xr <= 1.0) & (yr >= lower) & (yr <= upper)
    solid = inside.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


# ── Primitive shape masks ───────────────────────────────────────────────────

def build_rect_mask(
    nx: int, ny: int, nz: int,
    x0: float, y0: float,
    width: float, height: float,
    device: torch.device,
) -> torch.Tensor:
    """Solid mask for a rectangular block extruded along z."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    inside = (xx >= x0) & (xx < x0 + width) & (yy >= y0) & (yy < y0 + height)
    solid = inside.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_step_mask(
    nx: int, ny: int, nz: int,
    step_x: float, step_h: float,
    device: torch.device,
) -> torch.Tensor:
    """Backward-facing step: solid from x=0..step_x, y=0..step_h."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    inside = (xx < step_x) & (yy < step_h)
    solid = inside.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_ahmed_mask(
    nx: int, ny: int, nz: int,
    L: float, W: float, H_body: float,
    slant_deg: float,
    cx: float, cy: float, cz_base: float,
    device: torch.device,
) -> torch.Tensor:
    """Ahmed body: box + 25° slant at rear top.

    Body sits with base at cz_base, centered at cx along x, cy along y.
    L = length (x), W = width (y), H_body = height (z).
    The slant starts at x = cx + L*0.7 (30% rear overhang with slant).
    """
    import math as _m
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x0 = cx - L / 2.0
    x1 = cx + L / 2.0
    y0 = cy - W / 2.0
    y1 = cy + W / 2.0

    # Slant starts at 70% of body length from front
    slant_start = x0 + L * 0.7
    slant_rad = _m.radians(slant_deg)

    # Full-height region: x from x0 to slant_start, full H_body
    box = (xx >= x0) & (xx < slant_start) & (yy >= y0) & (yy < y1) & (zz >= cz_base) & (zz < cz_base + H_body)

    # Slanted region: x from slant_start to x1, height decreases linearly
    slant_region = (xx >= slant_start) & (xx <= x1) & (yy >= y0) & (yy < y1)
    if slant_region.any():
        dist_from_slant = xx - slant_start  # distance into slant region
        slant_height = H_body - dist_from_slant * _m.tan(slant_rad)
        slant_height = slant_height.clamp(min=0.0)
        slant_solid = slant_region & (zz >= cz_base) & (zz < cz_base + slant_height)
        box = box | slant_solid

    return box


# ── Main ────────────────────────────────────────────────────────────────────

# Each case: (grid, geometry params, u_in, reference dict, postprocess key)
CASE_CONFIGS = {
    "naca0012_a0": {
        "grid": (300, 100, 4),
        "u_in": 0.08,
        "ref": {"Cd": 0.008},
        "chord": 200.0,
        "aoa_deg": 0.0,
        "m": 0.0, "p": 0.0, "t": 0.12,
        "cx_frac": 0.33, "cy_frac": 0.5,
        "reynolds": 6e6,
        "area_ref": "chord * nz",  # Cd = drag / (0.5*rho*u^2*chord*nz)
        "post": "cd",
    },
    "square_prism": {
        "grid": (200, 80, 4),
        "u_in": 0.08,
        "ref": {"Cd": 2.1},
        "D": 30.0,
        "cx_frac": 0.25, "cy_frac": 0.5,
        "reynolds": 22000,
        "area_ref": "D * nz",
        "post": "cd",
    },
    "backward_step": {
        "grid": (200, 80, 4),
        "u_in": 0.08,
        "ref": {"xr_h": 6.5},
        "step_h": 20.0, "step_x": 50.0,
        "H": 60.0,
        "reynolds": 5000,
        "area_ref": "step_h * nz",  # for Cd
        "post": "reattach",
    },
    "rect_prism_2_1_1": {
        "grid": (200, 120, 4),
        "u_in": 0.08,
        "ref": {"Cd": 1.3},
        "D": 30.0,  # streamwise depth = 2*D, cross-section = D x D
        "cx_frac": 0.25, "cy_frac": 0.5,
        "aspect": 2.0,  # depth/width = 2:1:1 => L=2D, W=D, H=D
        "reynolds": 200000,
        "area_ref": "D * nz",
        "post": "cd",
    },
    "s809_a0": {
        "grid": (300, 100, 4),
        "u_in": 0.08,
        "ref": {"Cd": 0.007},
        "chord": 200.0,
        "aoa_deg": 0.0,
        "cx_frac": 0.33, "cy_frac": 0.5,
        "reynolds": 2e6,
        "area_ref": "chord * nz",
        "post": "cd",
    },
    "ahmed_25deg": {
        "grid": (300, 120, 100),
        "u_in": 0.08,
        "ref": {"Cd": 0.25},
        "L": 260.0, "W": 100.0, "H_body": 70.0,
        "slant_deg": 25.0,
        "cx_frac": 0.4, "cy_frac": 0.5, "cz_base_frac": 0.1,
        "reynolds": 2e6,
        "area_ref": "W * H_body",  # frontal area
        "post": "cd",
    },
    "naca4412_a5": {
        "grid": (300, 100, 4),
        "u_in": 0.08,
        "ref": {"Cd": 0.007, "Cl": 0.8},
        "chord": 200.0,
        "aoa_deg": 5.0,
        "m": 0.04, "p": 0.4, "t": 0.12,
        "cx_frac": 0.33, "cy_frac": 0.5,
        "reynolds": 3e6,
        "area_ref": "chord * nz",
        "post": "cl_cd",
    },
    "sphere_d48": {
        "grid": (240, 120, 120),
        "u_in": 0.08,
        "ref": {"Cd": 0.47},
        "D": 48.0,
        "cx_frac": 0.25, "cy_frac": 0.5, "cz_frac": 0.5,
        "reynolds": 1000,
        "area_ref": "pi * (D/2)**2",
        "post": "cd",
    },
}


def _sn_cd(re: float) -> float:
    """Schiller-Naumann drag coefficient for sphere."""
    return (24.0 / re) * (1.0 + 0.15 * re ** 0.687) if re > 0 else float("nan")


def main():
    device_id = int(sys.argv[1])
    case = sys.argv[2]
    output_path = sys.argv[3]

    cfg = CASE_CONFIGS[case]
    nx, ny, nz = cfg["grid"]
    u_in = cfg["u_in"]
    re = cfg["reynolds"]
    n_steps = 2000
    warmup = 300  # sliding window 300
    cs_smag = 0.05

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[SDAA:{device_id} {case} Re={re:.0f}]"

    # ── Build geometry ──────────────────────────────────────────────────────
    t0 = time.time()

    if case == "naca0012_a0":
        chord = cfg["chord"]
        cx = nx * cfg["cx_frac"]
        cy = ny * cfg["cy_frac"]
        solid = build_naca_mask(
            nx, ny, nz, chord, cfg["aoa_deg"],
            cfg["m"], cfg["p"], cfg["t"],
            cx, cy, device,
        )
        A_ref = chord * nz

    elif case == "square_prism":
        D = cfg["D"]
        cx = nx * cfg["cx_frac"]
        cy = ny * cfg["cy_frac"]
        solid = build_rect_mask(nx, ny, nz, cx - D/2, cy - D/2, D, D, device)
        A_ref = D * nz

    elif case == "backward_step":
        step_h = cfg["step_h"]
        step_x = cfg["step_x"]
        solid = build_step_mask(nx, ny, nz, step_x, step_h, device)
        A_ref = step_h * nz

    elif case == "rect_prism_2_1_1":
        D = cfg["D"]
        cx = nx * cfg["cx_frac"]
        cy = ny * cfg["cy_frac"]
        depth = 2.0 * D  # 2:1 streamwise
        solid = build_rect_mask(nx, ny, nz, cx - depth/2, cy - D/2, depth, D, device)
        A_ref = D * nz  # frontal area = D * nz

    elif case == "s809_a0":
        chord = cfg["chord"]
        cx = nx * cfg["cx_frac"]
        cy = ny * cfg["cy_frac"]
        solid = build_s809_mask(nx, ny, nz, chord, cfg["aoa_deg"], cx, cy, device)
        A_ref = chord * nz

    elif case == "ahmed_25deg":
        L = cfg["L"]
        W = cfg["W"]
        H_body = cfg["H_body"]
        cx = nx * cfg["cx_frac"]
        cy = ny * cfg["cy_frac"]
        cz_base = nz * cfg["cz_base_frac"]
        solid = build_ahmed_mask(
            nx, ny, nz, L, W, H_body, cfg["slant_deg"],
            cx, cy, cz_base, device,
        )
        A_ref = W * H_body

    elif case == "naca4412_a5":
        chord = cfg["chord"]
        cx = nx * cfg["cx_frac"]
        cy = ny * cfg["cy_frac"]
        solid = build_naca_mask(
            nx, ny, nz, chord, cfg["aoa_deg"],
            cfg["m"], cfg["p"], cfg["t"],
            cx, cy, device,
        )
        A_ref = chord * nz

    elif case == "sphere_d48":
        D = cfg["D"]
        radius = D / 2.0
        cx_s = nx * cfg["cx_frac"]
        cy_s = ny * cfg["cy_frac"]
        cz_s = nz * cfg["cz_frac"]
        solid = sphere_mask(nx, ny, nz, cx_s, cy_s, cz_s, radius, device=device)
        A_ref = math.pi * radius ** 2

    else:
        raise ValueError(f"Unknown case: {case}")

    nu = u_in * (A_ref / nz if "airfoil" in case or "s809" in case else
                  (cfg.get("D", cfg.get("step_h", 1.0)))) / re
    # Correct nu computation: nu = u_in * L_char / Re
    if case in ("naca0012_a0", "s809_a0", "naca4412_a5"):
        L_char = cfg["chord"]
    elif case == "sphere_d48":
        L_char = cfg["D"]
    elif case == "backward_step":
        L_char = cfg["step_h"]
    elif case == "ahmed_25deg":
        L_char = cfg["H_body"]
    elif case == "square_prism":
        L_char = cfg["D"]
    elif case == "rect_prism_2_1_1":
        L_char = cfg["D"]
    else:
        L_char = 1.0
    nu = u_in * L_char / re
    tau = 3.0 * nu + 0.5
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_ref

    solid_cells = int(solid.sum().item())
    print(f"{tag} nx={nx} ny={ny} nz={nz} L_char={L_char:.1f} u_in={u_in} nu={nu:.6e} "
          f"tau={tau:.6f} Cs={cs_smag} solid_cells={solid_cells} A_ref={A_ref:.1f}",
          flush=True)

    # ── Initialize flow field ───────────────────────────────────────────────
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    initial_mass = float(torch.ones_like(rho0).sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s)", flush=True)

    # ── Accumulators ────────────────────────────────────────────────────────
    cd_hist = []    # total drag coefficient
    cl_hist = []    # lift coefficient (for airfoils)
    do_cl = case in ("naca4412_a5",)

    for step in range(1, n_steps + 1):
        # 1. Collision: MRT + Smagorinsky LES
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)

        # 2. Stream
        f = stream3d(f)

        # 3. Wall function (body force + drag computation)
        f, drag_fric, drag_pres = wall_function_3d(f, solid, nu, y_val=0.5)

        # 4. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in)

        # 5. Mass correction every 100 steps
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        # Compute Cd
        cd_total = (drag_fric + drag_pres) / dyn_p if dyn_p > 0 else 0.0

        if step > warmup and math.isfinite(cd_total):
            cd_hist.append(cd_total)

        # Compute Cl for airfoils
        if do_cl and step > warmup:
            rho, ux, uy, uz = macroscopic3d(f)
            p = (rho - 1.0) / 3.0
            # Lift from y-direction pressure integral
            sn = torch.roll(solid, 1, dims=1)   # solid at y+1
            sp = torch.roll(solid, -1, dims=1)  # solid at y-1
            fluid = ~solid
            lift_pres = float((-p * (sn.float() - sp.float()) * fluid.float()).sum().item())
            cl_hist.append(lift_pres / dyn_p if dyn_p > 0 else 0.0)

        # Check for divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 200 == 0:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            cl_avg = sum(cl_hist) / max(len(cl_hist), 1) if cl_hist else 0.0
            elapsed = time.time() - t0
            extra = f" Cl_avg={cl_avg:.4f}" if do_cl else ""
            print(f"{tag} step={step} Cd={cd_total:.4f} Cd_avg={cd_avg:.4f}{extra} ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0

    # ── Final statistics ────────────────────────────────────────────────────
    is_finite = bool(torch.isfinite(f).all().item())

    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
    n_samples = len(cd_hist)
    cd_std = math.sqrt(
        sum((c - cd_mean) ** 2 for c in cd_hist) / max(n_samples - 1, 1)
    ) if n_samples > 1 else 0.0

    cl_mean = float("nan")
    cl_std = 0.0
    if cl_hist:
        cl_mean = sum(cl_hist) / len(cl_hist)
        cl_std = math.sqrt(
            sum((c - cl_mean) ** 2 for c in cl_hist) / max(len(cl_hist) - 1, 1)
        ) if len(cl_hist) > 1 else 0.0

    # Reference values
    ref_cd = cfg["ref"].get("Cd", float("nan"))
    ref_cl = cfg["ref"].get("Cl", float("nan"))
    error_cd = abs(cd_mean - ref_cd) / ref_cd * 100 if (
        math.isfinite(ref_cd) and ref_cd > 0 and math.isfinite(cd_mean)
    ) else float("nan")
    error_cl = abs(cl_mean - ref_cl) / ref_cl * 100 if (
        math.isfinite(ref_cl) and ref_cl > 0 and math.isfinite(cl_mean)
    ) else float("nan")

    # ── Special: reattachment length for backward step ──────────────────────
    xr_h = float("nan")
    if case == "backward_step" and is_finite:
        rho_f, ux_f, _, _ = macroscopic3d(f)
        step_x = cfg["step_x"]
        step_h = cfg["step_h"]
        # Extract ux along bottom wall (y=1 after step) at mid-z
        ux_bottom = ux_f[nz // 2, 1, int(step_x):]  # from step to outlet
        # Find first cell where ux > 0 (reattachment)
        pos = (ux_bottom > 0).nonzero(as_tuple=False)
        if pos.numel() > 0:
            xr = float(pos[0].item())  # cells from step
            xr_h = xr / step_h

    # Reference for step
    ref_xr = cfg["ref"].get("xr_h", float("nan"))
    error_xr = abs(xr_h - ref_xr) / ref_xr * 100 if (
        math.isfinite(ref_xr) and ref_xr > 0 and math.isfinite(xr_h)
    ) else float("nan")

    result = {
        "case": case,
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q19",
        "collision": f"MRT+Smag(Cs={cs_smag})",
        "boundary": "wall_function_3d(log-law)+farfield",
        "grid": f"{nx}x{ny}x{nz}",
        "L_char": L_char,
        "u_in": u_in,
        "Re": re,
        "nu": nu,
        "tau": tau,
        "A_ref": A_ref,
        "solid_cells": solid_cells,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cd_ref": ref_cd,
        "Cd_error_pct": error_cd,
        "cd_samples": n_samples,
        "Cl_mean": cl_mean if do_cl else None,
        "Cl_std": cl_std if do_cl else None,
        "Cl_ref": ref_cl if do_cl else None,
        "Cl_error_pct": error_cl if do_cl else None,
        "xr_h": xr_h if case == "backward_step" else None,
        "xr_h_ref": ref_xr if case == "backward_step" else None,
        "xr_h_error_pct": error_xr if case == "backward_step" else None,
        "status": "DIV" if not is_finite else "OK",
        "finite": is_finite,
        "elapsed_s": elapsed,
    }

    status_str = "DIVERGED" if not is_finite else "OK"
    cd_str = f"Cd={cd_mean:.4f}" if math.isfinite(cd_mean) else "Cd=DIV"
    err_str = f"err={error_cd:.1f}%" if math.isfinite(error_cd) else ""
    cl_str = f" Cl={cl_mean:.4f} err={error_cl:.1f}%" if do_cl and math.isfinite(cl_mean) else ""
    xr_str = f" xr/h={xr_h:.2f} err={error_xr:.1f}%" if case == "backward_step" and math.isfinite(xr_h) else ""
    print(f"{tag} DONE {cd_str}{cl_str}{xr_str} {err_str} {status_str} wall={elapsed:.0f}s",
          flush=True)

    Path(output_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
