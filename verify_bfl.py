"""BFL (Bouzidi-Firdaouss-Lallemand) verification script.

Tests:
  Step 1: BFL q=0.5 on Poiseuille — cross-validate with standard BB
  Step 2: BFL q=0.5 on Couette (moving wall) — cross-validate with standard BB
  Step 3: BFL on cylinder (varying q) — compare drag with standard BB
  Step 4: BFL + momentum exchange on cylinder — compare with standard BB+ME

Uses the verified-correct main loop:
  collide → NoDynamics → BB/BFL(after stream) → far_field_bc → correct_mass

BFL with q=0.5 (flat wall) should be IDENTICAL to standard half-way bounce-back.
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d
from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.bfl_d3q19 import compute_q_cylinder_d3q19, bouzidi_bounce_back_d3q19
from tensorlbm.solver3d import collide_bgk3d, stream3d, correct_mass3d
from tensorlbm.drag_pressure import SurfaceMesh, drag_pressure_integration, drag_friction_integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_flat_wall_boundary_mask(solid, device):
    """Create per-direction fluid boundary mask for flat walls.

    For each direction d, mask[d] is True where:
      - current cell is fluid
      - neighbor in direction d is solid
    """
    nz, ny, nx = solid.shape
    c = C.to(device)
    mask = torch.zeros((19, nz, ny, nx), dtype=torch.bool, device=device)
    for d in range(1, 19):
        dcx, dcy, dcz = int(c[d, 0]), int(c[d, 1]), int(c[d, 2])
        nb_solid = torch.roll(solid, shifts=(-dcz, -dcy, -dcx), dims=(0, 1, 2))
        boundary = (~solid) & nb_solid
        mask[d] = boundary
    return mask


def apply_body_force(f, fx, fy=0.0, fz=0.0):
    """Guo body-force forcing."""
    device = f.device
    c = C.to(device).float()
    w = W.to(device).view(19, 1, 1, 1)
    cs2 = 1.0 / 3.0
    rho, ux, uy, uz = macroscopic3d(f)
    cu = c[:, 0].view(19, 1, 1, 1) * ux + c[:, 1].view(19, 1, 1, 1) * uy + c[:, 2].view(19, 1, 1, 1) * uz
    cF = c[:, 0].view(19, 1, 1, 1) * fx + c[:, 1].view(19, 1, 1, 1) * fy + c[:, 2].view(19, 1, 1, 1) * fz
    forcing = w * (1.0 + cu / cs2) * cF / cs2
    return f + forcing


def moving_wall_bb(f, solid_top, u_w, rho_w=1.0):
    """Moving-wall bounce-back at solid cells (top wall).

    f[q] = f[opp_q] + 6 * w_q * rho * (c_q · u_w)  [D3Q19, cs²=1/3]
    """
    device = f.device
    opp = OPPOSITE.to(device)
    c = C.to(device).float()
    w = W.to(device).view(19, 1, 1, 1)
    # Correction: 6 * w_q * rho * (c_q · u_w)
    correction = 6.0 * w * rho_w * (
        c[:, 0].view(19, 1, 1, 1) * u_w[0] +
        c[:, 1].view(19, 1, 1, 1) * u_w[1] +
        c[:, 2].view(19, 1, 1, 1) * u_w[2]
    )
    # Apply at top wall solid cells only
    mask = solid_top.unsqueeze(0)  # (1, nz, ny, nx)
    f_new = torch.where(mask, f[opp] + correction, f)
    return f_new


def moving_wall_bfl(f, f_prev, fluid_boundary_mask, q_field, solid_top_mask, u_w, rho_w=1.0):
    """BFL with moving-wall correction at top wall boundary cells.

    For q=0.5: f_bc = f_opp + 6 * w_d * rho * (-c_d · u_w)
    (correction for the bounced population f[opp_d])
    """
    device = f.device
    opp = OPPOSITE.to(device)
    c = C.to(device).float()
    w = W.to(device)
    f_out = f.clone()

    for d in range(1, 19):
        opp_d = int(opp[d].item())
        mask = fluid_boundary_mask[d] & solid_top_mask  # only top wall
        if not mask.any():
            continue

        q_cell = q_field[d][mask]
        f_opp = f[opp_d][mask]
        fp_d = f_prev[d][mask]
        fp_opp = f_prev[opp_d][mask]

        # BFL formula
        mask_lin = q_cell < 0.5
        mask_quad = ~mask_lin
        f_bc_lin = 2.0 * q_cell * f_opp + (1.0 - 2.0 * q_cell) * fp_d
        safe_q = torch.where(mask_quad, q_cell, torch.ones_like(q_cell))
        f_bc_quad = f_opp / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
        f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

        # Moving wall correction for bounced population f[opp_d]:
        # correction = 6 * w_opp_d * rho * (c_opp_d · u_w)
        #            = 6 * w_d * rho * (-c_d · u_w)  (w_opp=w_d, c_opp=-c_d)
        w_d = w[d]
        c_d = c[d]
        correction = 6.0 * w_d * rho_w * (
            -c_d[0] * u_w[0] - c_d[1] * u_w[1] - c_d[2] * u_w[2]
        )
        f_bc = f_bc + correction

        target = f_out[opp_d].clone()
        target[mask] = f_bc
        f_out[opp_d] = target

    # Also apply standard BFL (q=0.5) at bottom wall (stationary)
    for d in range(1, 19):
        opp_d = int(opp[d].item())
        mask = fluid_boundary_mask[d] & (~solid_top_mask)
        if not mask.any():
            continue

        q_cell = q_field[d][mask]
        f_opp = f[opp_d][mask]
        fp_d = f_prev[d][mask]
        fp_opp = f_prev[opp_d][mask]

        mask_lin = q_cell < 0.5
        mask_quad = ~mask_lin
        f_bc_lin = 2.0 * q_cell * f_opp + (1.0 - 2.0 * q_cell) * fp_d
        safe_q = torch.where(mask_quad, q_cell, torch.ones_like(q_cell))
        f_bc_quad = f_opp / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
        f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

        target = f_out[opp_d].clone()
        target[mask] = f_bc
        f_out[opp_d] = target

    return f_out


def compute_me_linkwise(f_post_stream, f_pre_stream, solid, device):
    """Link-wise momentum exchange (Ladd 1994) for standard BB.

    F = Σ (f_in + f_out) * c_q for fluid→solid links
    where f_in = f_streamed[q] at solid, f_out = f_after_BB[opp_q] at solid.
    """
    c = C.to(device).float()
    opp = OPPOSITE.to(device)
    fluid = ~solid
    fx = torch.tensor(0.0, device=device)

    for q in range(1, 19):
        cqx = int(c[q, 0].item())
        q_opp = int(opp[q].item())
        # f_in: streamed from fluid to solid
        # Check if source (solid - c_q) is fluid
        fluid_shifted = torch.roll(fluid, shifts=(cqx,), dims=(2,))  # simplified for x-only
        # Actually, need full 3D roll
        cz, cy, cx = int(c[q, 2]), int(c[q, 1]), int(c[q, 0])
        fluid_shifted = torch.roll(fluid, shifts=(cz, cy, cx), dims=(0, 1, 2))
        is_fluid_link = fluid_shifted[solid]
        if not is_fluid_link.any():
            continue
        f_in = f_post_stream[q][solid]
        # f_out: after BB, f[opp_q] at solid = f[q] at solid (before BB)
        # With NoDynamics, f[q] at solid before BB = feq[q] at solid
        # But we need the actual value. Use f_pre_stream[q] at solid (post-collision = feq with NoDynamics)
        f_out = f_pre_stream[q][solid]  # post-collision at solid = feq (NoDynamics)
        contrib = (f_in + f_out) * is_fluid_link.to(f_in.dtype)
        fx = fx + float(cqx) * contrib.sum()

    return float(fx.item())


def compute_me_bfl(f_post_bfl, f_pre_stream, fluid_boundary_mask, q_field, solid, device):
    """BFL momentum exchange: F = Σ (f_pre[d] + f_bc) * c_d / q.

    Uses the BFL-interpolated f_bc (already applied to f_post_bfl[opp_d])
    and pre-stream f_pre[d] at the fluid boundary cell.
    """
    c = C.to(device).float()
    opp = OPPOSITE.to(device)
    fx = torch.tensor(0.0, device=device)

    for d in range(1, 19):
        opp_d = int(opp[d].item())
        mask = fluid_boundary_mask[d]
        if not mask.any():
            continue

        q_cell = q_field[d][mask].clamp(min=0.01)
        # f_pre[d] at fluid boundary cell (pre-stream, post-collision)
        fp_d = f_pre_stream[d][mask]
        # f_bc = f_post_bfl[opp_d] at fluid boundary cell (the BFL-reconstructed value)
        f_bc = f_post_bfl[opp_d][mask]

        c_d_x = float(c[d, 0].item())
        # F = (f_pre[d] + f_bc) * c_d_x / q
        contrib = (fp_d + f_bc) * c_d_x / q_cell
        fx = fx + contrib.sum()

    return float(fx.item())


# ---------------------------------------------------------------------------
# Step 1: Poiseuille flow (q=0.5, flat wall, cross-validate BB vs BFL)
# ---------------------------------------------------------------------------

def run_poiseuille(device, n_steps=3000):
    """Poiseuille flow: body force driven, stationary walls, q=0.5.

    BFL with q=0.5 should give IDENTICAL velocity profile as standard BB.
    """
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0  # 1/6
    H = ny - 2  # fluid gap = 10 (walls at y=0.5 and y=10.5)
    U_max_target = 0.05
    # G = 8*nu*U_max / H^2  (Poiseuille: u_max = G*H^2/(8*nu))
    G = 8.0 * nu * U_max_target / (H ** 2)

    tag = f"[Poiseuille {device}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.4f} H={H} U_max={U_max_target} G={G:.6e}", flush=True)

    # Solid mask: walls at y=0 and y=ny-1
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True
    solid[:, -1, :] = True

    # Boundary mask and q=0.5 field
    bfl_mask = make_flat_wall_boundary_mask(solid, device)
    q_field = torch.full((19, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    # Equilibrium for solid (NoDynamics: u=0)
    rho0 = torch.ones(nz, ny, nx, device=device)
    f_eq_solid = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))

    results = {}

    for method in ["standard_BB", "BFL_q05"]:
        f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))
        initial_mass = float(rho0.sum().item())

        for step in range(1, n_steps + 1):
            # 1. NoDynamics: reset solid to equilibrium
            f[:, solid] = f_eq_solid[:, solid]
            # 2. Collide
            f = collide_bgk3d(f, tau=tau)
            # 3. Body force
            f = apply_body_force(f, fx=G)
            # 4. Save pre-stream
            f_pre = f.clone()
            # 5. Stream
            f = stream3d(f)
            # 6. Wall BC
            if method == "standard_BB":
                f = bounce_back_cells_3d(f, solid)
            else:
                f = bouzidi_bounce_back_d3q19(f, f_pre, bfl_mask, q_field)
            # 7. Mass correction
            if step % 100 == 0:
                f = correct_mass3d(f, initial_mass)

            if not torch.isfinite(f).all():
                print(f"{tag} {method} DIVERGED at step {step}", flush=True)
                break

        # Analyze: velocity profile
        rho, ux, uy, uz = macroscopic3d(f)
        # Average over x and z (periodic directions)
        ux_profile = ux[:, 1:-1, :].mean(dim=(0, 2))  # (ny-2,)
        u_max_sim = float(ux_profile.max().item())
        # Analytical: u(y) = G/(2*nu) * y * (H - y), y in [0, H]
        y = torch.arange(1, ny - 1, dtype=torch.float32, device=device) - 0.5  # cell centres
        u_analytical = G / (2 * nu) * y * (H - y)
        u_err = float(((ux_profile - u_analytical).abs() / u_analytical.max()).mean().item() * 100)

        results[method] = {
            "u_max_sim": u_max_sim,
            "u_max_analytical": float(u_analytical.max().item()),
            "u_err_pct": u_err,
        }
        print(f"{tag} {method}: u_max={u_max_sim:.6f} (analytical={float(u_analytical.max().item()):.6f}) u_err={u_err:.2f}%", flush=True)

    # Cross-validation
    diff = abs(results["standard_BB"]["u_max_sim"] - results["BFL_q05"]["u_max_sim"])
    diff_pct = diff / max(abs(results["standard_BB"]["u_max_sim"]), 1e-12) * 100
    print(f"{tag} CROSS-VALIDATION: u_max diff = {diff:.2e} ({diff_pct:.4f}%)", flush=True)
    results["cross_val_diff_pct"] = diff_pct
    results["match"] = diff_pct < 0.01
    return results


# ---------------------------------------------------------------------------
# Step 2: Couette flow (q=0.5, moving wall, cross-validate BB vs BFL)
# ---------------------------------------------------------------------------

def run_couette(device, n_steps=3000):
    """Couette flow: moving top wall, stationary bottom, q=0.5.

    BFL with q=0.5 should give IDENTICAL Cf as standard BB.
    """
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    U = 0.05  # top wall velocity
    H = ny - 2  # fluid gap = 10

    tag = f"[Couette {device}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} tau={tau} nu={nu:.4f} U={U} H={H}", flush=True)

    # Solid mask
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    solid[:, 0, :] = True   # bottom (stationary)
    solid[:, -1, :] = True  # top (moving at U)
    solid_top = torch.zeros_like(solid)
    solid_top[:, -1, :] = True

    # Boundary mask and q=0.5 field
    bfl_mask = make_flat_wall_boundary_mask(solid, device)
    q_field = torch.full((19, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    rho0 = torch.ones(nz, ny, nx, device=device)
    f_eq_solid = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))

    # Analytical Cf for Couette: Cf = 2*nu / (U * H)
    Cf_analytical = 2.0 * nu / (U * H)

    results = {}

    for method in ["standard_BB", "BFL_q05"]:
        f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))
        initial_mass = float(rho0.sum().item())

        for step in range(1, n_steps + 1):
            # 1. NoDynamics
            f[:, solid] = f_eq_solid[:, solid]
            # 2. Collide
            f = collide_bgk3d(f, tau=tau)
            # 3. Save pre-stream
            f_pre = f.clone()
            # 4. Stream
            f = stream3d(f)
            # 5. Wall BC
            if method == "standard_BB":
                # Standard BB at all solid cells, then moving wall correction at top
                f = bounce_back_cells_3d(f, solid)
                # Moving wall correction: add momentum at top wall
                f = moving_wall_bb(f, solid_top, (U, 0.0, 0.0))
            else:
                # BFL with moving wall correction
                f = moving_wall_bfl(f, f_pre, bfl_mask, q_field, solid_top, (U, 0.0, 0.0))
            # 6. Mass correction
            if step % 100 == 0:
                f = correct_mass3d(f, initial_mass)

            if not torch.isfinite(f).all():
                print(f"{tag} {method} DIVERGED at step {step}", flush=True)
                break

        # Analyze: velocity profile (linear for Couette)
        rho, ux, uy, uz = macroscopic3d(f)
        ux_profile = ux[:, 1:-1, :].mean(dim=(0, 2))  # (ny-2,)
        # Analytical: u(y) = U * y / H, y in [0, H]
        y = torch.arange(1, ny - 1, dtype=torch.float32, device=device) - 0.5
        u_analytical = U * y / H
        u_err = float(((ux_profile - u_analytical).abs() / max(U, 1e-12)).mean().item() * 100)

        # Cf from velocity gradient at wall
        # du/dy at bottom wall ≈ ux[1] / 0.5 (distance from wall to first cell centre)
        du_dy = float(ux[:, 1, :].mean().item()) / 0.5
        Cf_sim = 2.0 * nu * du_dy / (U ** 2)  # Wait, Cf = tau_w / (0.5*rho*U^2) = 2*nu*du/dy / U^2
        # Actually: tau_w = nu * du/dy, Cf = tau_w / (0.5*rho*U^2) = 2*nu*du/dy/U^2
        # But for Couette, du/dy = U/H, so Cf = 2*nu*U/(H*U^2) = 2*nu/(U*H)
        # Let me use the direct formula
        Cf_sim = 2.0 * nu * du_dy / (U ** 2)

        results[method] = {
            "u_max_sim": float(ux_profile.max().item()),
            "u_max_analytical": float(u_analytical.max().item()),
            "u_err_pct": u_err,
            "Cf_sim": Cf_sim,
            "Cf_analytical": Cf_analytical,
        }
        print(f"{tag} {method}: Cf={Cf_sim:.4f} (analytical={Cf_analytical:.4f}) u_err={u_err:.2f}%", flush=True)

    diff = abs(results["standard_BB"]["Cf_sim"] - results["BFL_q05"]["Cf_sim"])
    diff_pct = diff / max(abs(results["standard_BB"]["Cf_sim"]), 1e-12) * 100
    print(f"{tag} CROSS-VALIDATION: Cf diff = {diff:.2e} ({diff_pct:.4f}%)", flush=True)
    results["cross_val_diff_pct"] = diff_pct
    results["match"] = diff_pct < 0.01
    return results


# ---------------------------------------------------------------------------
# Step 3 & 4: Cylinder (varying q, BFL vs BB, + momentum exchange)
# ---------------------------------------------------------------------------

def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def run_cylinder(device, n_steps=3000, use_bfl=False, compute_me=False):
    """Cylinder flow: Re=200, D=24, far-field BC.

    Compare standard BB vs BFL (varying q) drag.
    Optionally compute momentum exchange drag.
    """
    nx, ny, nz = 200, 80, 4
    diameter = 24.0
    radius = diameter / 2.0
    u_in = 0.08
    Re = 200.0
    nu = u_in * diameter / Re
    tau = 3.0 * nu + 0.5

    tag = f"[Cylinder {device} {'BFL' if use_bfl else 'BB'}{'+ME' if compute_me else ''}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} nu={nu:.6e} tau={tau:.4f}", flush=True)

    device_t = torch.device(device)
    torch.sdaa.set_device(device_t)

    cx_cyl = nx * 0.25
    cy_cyl = ny * 0.5
    solid = build_cylinder_mask(nx, ny, nz, cx_cyl, cy_cyl, radius, device_t)

    A_frontal = diameter * nz
    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_frontal

    # BFL q-field
    bfl_mask = None
    q_field = None
    if use_bfl:
        bfl_mask, q_field = compute_q_cylinder_d3q19(nx, ny, nz, cx_cyl, cy_cyl, radius, device_t)
        n_links = int(bfl_mask.sum().item())
        q_min = float(q_field[bfl_mask].min().item())
        q_max = float(q_field[bfl_mask].max().item())
        q_mean = float(q_field[bfl_mask].mean().item())
        print(f"{tag} BFL q-field: {n_links} links, q=[{q_min:.4f}, {q_max:.4f}], mean={q_mean:.4f}", flush=True)

    # Surface mesh for pressure drag
    from tensorlbm.drag_pressure import get_near_wall_2d
    near = get_near_wall_2d(solid)
    mesh = SurfaceMesh.from_cylinder(solid, near, cx_cyl, cy_cyl, radius)

    # Initialize
    rho0 = torch.ones(nz, ny, nx, device=device_t)
    ux0 = torch.full((nz, ny, nx), u_in, device=device_t)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device_t)
    initial_mass = float(rho0.sum().item())

    # NoDynamics equilibrium for solid
    f_eq_solid = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0), device=device_t)

    t0 = time.time()
    cd_hist = []
    cd_me_hist = []

    for step in range(1, n_steps + 1):
        # 1. NoDynamics
        f[:, solid] = f_eq_solid[:, solid]
        # 2. Collide
        f = collide_bgk3d(f, tau=tau)
        # 3. Save pre-stream
        f_pre = f.clone()
        # 4. Stream
        f = stream3d(f)
        # 5. Far-field BC
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)
        # 6. Wall BC
        if use_bfl and bfl_mask is not None:
            f = bouzidi_bounce_back_d3q19(f, f_pre, bfl_mask, q_field)
        else:
            f = bounce_back_cells_3d(f, solid)
        # 7. Mass correction
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        # Compute drag
        if step > n_steps // 2:  # second half
            # Pressure + friction drag
            cd_p, _ = drag_pressure_integration(f, mesh, dyn_p)
            cd_f = drag_friction_integration(f, mesh, dyn_p, nu)
            cd_total = cd_p + cd_f
            if math.isfinite(cd_total):
                cd_hist.append(cd_total)

            # Momentum exchange drag
            if compute_me:
                if use_bfl:
                    fx_me = compute_me_bfl(f, f_pre, bfl_mask, q_field, solid, device_t)
                else:
                    fx_me = compute_me_linkwise(f, f_pre, solid, device_t)
                cd_me = fx_me / dyn_p
                if math.isfinite(cd_me):
                    cd_me_hist.append(cd_me)

        if step % 500 == 0:
            cd_avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd={cd_avg:.4f} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    cd_mean = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
    cd_std = (sum((c - cd_mean) ** 2 for c in cd_hist) / max(len(cd_hist) - 1, 1)) ** 0.5 if len(cd_hist) > 1 else 0.0

    result = {
        "method": "BFL" if use_bfl else "standard_BB",
        "compute_me": compute_me,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cd_ref": 1.30,  # Re=200 cylinder
        "n_samples": len(cd_hist),
        "elapsed_s": elapsed,
    }

    if compute_me and cd_me_hist:
        cd_me_mean = sum(cd_me_hist) / max(len(cd_me_hist), 1)
        result["Cd_ME_mean"] = cd_me_mean
        print(f"{tag} DONE Cd_pf={cd_mean:.4f} Cd_ME={cd_me_mean:.4f} (ref=1.30) time={elapsed:.0f}s", flush=True)
    else:
        print(f"{tag} DONE Cd={cd_mean:.4f} (ref=1.30) time={elapsed:.0f}s", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device_id = sys.argv[1] if len(sys.argv) > 1 else "12"
    test = sys.argv[2] if len(sys.argv) > 2 else "all"
    device = f"sdaa:{device_id}"

    all_results = {}

    if test in ("all", "poiseuille"):
        all_results["step1_poiseuille"] = run_poiseuille(device, n_steps=3000)

    if test in ("all", "couette"):
        all_results["step2_couette"] = run_couette(device, n_steps=3000)

    if test in ("all", "cylinder_bb"):
        all_results["step3_cylinder_BB"] = run_cylinder(device, n_steps=3000, use_bfl=False, compute_me=False)

    if test in ("all", "cylinder_bfl"):
        all_results["step3_cylinder_BFL"] = run_cylinder(device, n_steps=3000, use_bfl=True, compute_me=False)

    if test in ("all", "cylinder_bb_me"):
        all_results["step4_cylinder_BB_ME"] = run_cylinder(device, n_steps=3000, use_bfl=False, compute_me=True)

    if test in ("all", "cylinder_bfl_me"):
        all_results["step4_cylinder_BFL_ME"] = run_cylinder(device, n_steps=3000, use_bfl=True, compute_me=True)

    print("\n" + "=" * 70)
    print("BFL VERIFICATION SUMMARY")
    print("=" * 70)
    for key, val in all_results.items():
        print(f"\n{key}:")
        if isinstance(val, dict):
            for k, v in val.items():
                print(f"  {k}: {v}")

    # Save results
    out_path = Path(f"/root/TensorLBM_dev/bfl_verification_results_sdaa{device_id}.json")
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
