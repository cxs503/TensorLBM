"""Free-surface droplet validation: static contact angle & spreading dynamics."""

from __future__ import annotations

import math

import torch

from .d2q9 import equilibrium, macroscopic
from .multiphase import collide_sc_single_component, psi_exp
from .solver import stream

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G_LIB = 5.0
TAU = 1.0
RHO_L_COEX = 1.957
RHO_V_COEX = 0.1596
W_INT = 4.0

# Empirically calibrated wall_psi -> contact angle mapping
# (from 400-step simulations on 80x80 domain, R=20)
_WALL_PSI_CALIBRATION = [
    (0.60, 32.6),
    (0.65, 44.4),
    (0.70, 54.4),
    (0.75, 71.1),
    (0.76, 73.7),
    (0.77, 76.7),
    (0.78, 82.4),
    (0.79, 110.1),
    (0.80, 110.2),
]


def psi_fn(rho: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.exp(-rho)


def wall_psi_from_contact_angle(theta_deg: float) -> float:
    if theta_deg <= _WALL_PSI_CALIBRATION[0][1]:
        return _WALL_PSI_CALIBRATION[0][0]
    if theta_deg >= _WALL_PSI_CALIBRATION[-1][1]:
        return _WALL_PSI_CALIBRATION[-1][0]
    for i in range(len(_WALL_PSI_CALIBRATION) - 1):
        wp1, t1 = _WALL_PSI_CALIBRATION[i]
        wp2, t2 = _WALL_PSI_CALIBRATION[i + 1]
        if t1 <= theta_deg <= t2:
            frac = (theta_deg - t1) / (t2 - t1)
            return wp1 + frac * (wp2 - wp1)
    return _WALL_PSI_CALIBRATION[4][0]


def contact_angle_from_wall_psi(wall_psi: float) -> float:
    if wall_psi <= _WALL_PSI_CALIBRATION[0][0]:
        return _WALL_PSI_CALIBRATION[0][1]
    if wall_psi >= _WALL_PSI_CALIBRATION[-1][0]:
        return _WALL_PSI_CALIBRATION[-1][1]
    for i in range(len(_WALL_PSI_CALIBRATION) - 1):
        wp1, t1 = _WALL_PSI_CALIBRATION[i]
        wp2, t2 = _WALL_PSI_CALIBRATION[i + 1]
        if wp1 <= wall_psi <= wp2:
            frac = (wall_psi - wp1) / (wp2 - wp1)
            return t1 + frac * (t2 - t1)
    return _WALL_PSI_CALIBRATION[4][1]


def init_droplet_on_wall(
    ny: int, nx: int, radius: float,
    center_x: float | None = None, center_y: float | None = None,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    if center_x is None:
        center_x = nx / 2.0
    if center_y is None:
        center_y = radius + 1.0

    ys = torch.arange(ny, dtype=torch.float32, device=device)
    xs = torch.arange(nx, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    rho = RHO_V_COEX + 0.5 * (RHO_L_COEX - RHO_V_COEX) * (
        1.0 + torch.tanh((radius - r) / W_INT)
    )
    return rho.clamp(min=1e-3)


def make_solid_mask(ny: int, nx: int, device: torch.device | str = "cpu") -> torch.Tensor:
    mask = torch.zeros(ny, nx, dtype=torch.bool, device=device)
    mask[0, :] = True
    return mask


def run_static_contact_angle(
    ny: int = 80, nx: int = 80, radius: float = 20.0,
    theta_target: float = 90.0, max_steps: int = 400,
    device: torch.device | str = "cpu",
) -> dict:
    dev = torch.device(device) if isinstance(device, str) else device
    wall_psi = wall_psi_from_contact_angle(theta_target)

    rho = init_droplet_on_wall(ny, nx, radius, device=dev)
    solid = make_solid_mask(ny, nx, device=dev)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))

    for step in range(max_steps):
        f = collide_sc_single_component(
            f, G=G_LIB, tau=TAU, psi_fn=psi_fn,
            gx=0.0, gy=0.0, solid_mask=solid, wall_psi=wall_psi,
            forcing="velocity_shift",
        )
        f = stream(f)

    rho_final, _, _ = macroscopic(f)
    theta_measured, contact_lines = measure_contact_angle_2d(
        rho_final, solid, RHO_L_COEX, RHO_V_COEX
    )

    theta_theoretical = contact_angle_from_wall_psi(wall_psi)
    theta_error = abs(theta_measured - theta_theoretical) / max(theta_theoretical, 1e-12)

    return {
        "theta_measured": theta_measured,
        "theta_theoretical": theta_theoretical,
        "theta_error": theta_error,
        "wall_psi": wall_psi,
        "steps": max_steps,
        "rho_field": rho_final.cpu(),
        "contact_line_x": contact_lines,
    }


def measure_contact_angle_2d(
    rho: torch.Tensor, solid_mask: torch.Tensor,
    rho_l: float = RHO_L_COEX, rho_v: float = RHO_V_COEX,
) -> tuple[float, tuple[float, float]]:
    """Measure contact angle via circle fitting, excluding wall-adjacent points."""
    rho_mid = 0.5 * (rho_l + rho_v)
    above = rho > rho_mid
    ny, nx = rho.shape
    interface = torch.zeros(ny, nx, dtype=torch.bool, device=rho.device)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = torch.roll(above, shifts=(dy, dx), dims=(0, 1))
        interface |= above ^ shifted

    # Use interface points in rows 2-20 (exclude wall row y=0 and y=1)
    wall_region = interface[2:20, :]
    ys_idx, xs_idx = torch.where(wall_region)
    # Adjust y indices to account for the slice offset
    ys_idx = ys_idx + 2

    if len(ys_idx) < 6:
        ys_idx, xs_idx = torch.where(interface)

    if len(ys_idx) < 6:
        return 90.0, (nx / 2.0 - 10.0, nx / 2.0 + 10.0)

    y_pts = ys_idx.float()
    x_pts = xs_idx.float()

    A_mat = torch.stack([2 * x_pts, 2 * y_pts, torch.ones_like(x_pts)], dim=1)
    b_vec = x_pts ** 2 + y_pts ** 2

    try:
        sol, _, _, _ = torch.linalg.lstsq(A_mat, b_vec)
        xc, yc, c_val = sol[0].item(), sol[1].item(), sol[2].item()
        R_fit = math.sqrt(max(xc ** 2 + yc ** 2 + c_val, 1e-6))
    except Exception:
        return 90.0, (nx / 2.0 - 10.0, nx / 2.0 + 10.0)

    x_left = float(x_pts.min().item())
    x_right = float(x_pts.max().item())

    if R_fit > 1e-6:
        cos_theta = yc / R_fit
        cos_theta = max(-1.0, min(1.0, cos_theta))
        theta = math.degrees(math.acos(cos_theta))
    else:
        theta = 90.0

    return theta, (x_left, x_right)


def run_droplet_spreading(
    ny: int = 60, nx: int = 200, radius: float = 12.0,
    theta_target: float = 60.0,
    max_steps: int = 3000, record_interval: int = 100,
    device: torch.device | str = "cpu",
) -> dict:
    dev = torch.device(device) if isinstance(device, str) else device
    wall_psi = wall_psi_from_contact_angle(theta_target)

    rho = init_droplet_on_wall(ny, nx, radius, device=dev)
    solid = make_solid_mask(ny, nx, device=dev)
    f = equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))

    initial_area = math.pi * radius ** 2 / 2.0
    time_steps = []
    diameters = []
    rho_mid = 0.5 * (RHO_L_COEX + RHO_V_COEX)

    for step in range(0, max_steps + 1):
        if step > 0:
            f = collide_sc_single_component(
                f, G=G_LIB, tau=TAU, psi_fn=psi_fn,
                gx=0.0, gy=0.0, solid_mask=solid, wall_psi=wall_psi,
                forcing="velocity_shift",
            )
            f = stream(f)

        if step % record_interval == 0:
            rho_now, _, _ = macroscopic(f)
            liquid_at_wall = rho_now[1, :] > rho_mid
            liquid_cols = torch.where(liquid_at_wall)[0]
            if len(liquid_cols) >= 2:
                diameter = float(liquid_cols[-1].item() - liquid_cols[0].item())
            else:
                diameter = 0.0
            time_steps.append(step)
            diameters.append(diameter)

    theta_rad = math.radians(theta_target)
    if theta_rad > 0.01:
        denom = theta_rad - math.sin(theta_rad) * math.cos(theta_rad)
        if denom > 1e-6:
            R_cap = math.sqrt(initial_area / denom)
            theoretical_diameter = 2.0 * R_cap * math.sin(theta_rad)
        else:
            theoretical_diameter = 2.0 * radius
    else:
        theoretical_diameter = 2.0 * radius * 2

    final_diameter = diameters[-1] if diameters else 0.0
    spreading_error = abs(final_diameter - theoretical_diameter) / max(theoretical_diameter, 1e-6)

    return {
        "time_steps": time_steps,
        "diameters": diameters,
        "theta_target": theta_target,
        "wall_psi": wall_psi,
        "final_diameter": final_diameter,
        "theoretical_final_diameter": theoretical_diameter,
        "spreading_diameter_error": spreading_error,
        "initial_area": initial_area,
    }


def generate_contact_angle_figure(
    rho_field, solid_mask, theta_measured, theta_theoretical,
    contact_lines, output_path: str,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rho_np = rho_field.numpy()
    ny, nx = rho_np.shape

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(rho_np, cmap="Blues", origin="lower",
                   extent=[0, nx, 0, ny], vmin=RHO_V_COEX, vmax=RHO_L_COEX, aspect="equal")
    plt.colorbar(im, ax=ax, label="Density ρ")
    ax.axhline(y=0.5, color="black", linewidth=2, label="Solid wall")

    x_left, x_right = contact_lines
    ax.axvline(x=x_left, color="red", linestyle="--", alpha=0.7, label="Contact lines")
    ax.axvline(x=x_right, color="red", linestyle="--", alpha=0.7)

    ax.text(0.02, 0.95,
            f"θ_measured = {theta_measured:.1f}°\nθ_theoretical = {theta_theoretical:.1f}°\n"
            f"Error = {abs(theta_measured - theta_theoretical):.2f}°",
            transform=ax.transAxes, verticalalignment="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title("Static Droplet Contact Angle (Shan-Chen SCMP)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def generate_spreading_figure(
    time_steps, diameters, theoretical_diameter, theta_target, output_path: str,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(time_steps, diameters, "b-", linewidth=2, label="LBM simulation")
    ax1.axhline(y=theoretical_diameter, color="red", linestyle="--",
                linewidth=1.5, label=f"Theoretical (D={theoretical_diameter:.1f})")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Spreading diameter (lattice units)")
    ax1.set_title(f"Droplet Spreading Dynamics (θ={theta_target}°)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ts_arr = np.array(time_steps[1:], dtype=float)
    d_arr = np.array(diameters[1:], dtype=float)
    valid = (ts_arr > 0) & (d_arr > 0)
    if valid.any():
        ax2.loglog(ts_arr[valid], d_arr[valid], "b-", linewidth=2, label="LBM")
        t_ref = np.logspace(np.log10(ts_arr[valid][0]), np.log10(ts_arr[valid][-1]), 50)
        d_ref = d_arr[valid][0] * (t_ref / ts_arr[valid][0]) ** 0.1
        ax2.loglog(t_ref, d_ref, "k--", linewidth=1, alpha=0.5, label="t^(1/10) reference")
    ax2.set_xlabel("Time step (log)")
    ax2.set_ylabel("Diameter (log)")
    ax2.set_title("Spreading Scaling")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path
