#!/usr/bin/env python3
"""Free-surface (VOF) common-module benchmark worker — SDAA 20-23.

Runs four free-surface benchmarks using ONLY the common module
``tensorlbm.free_surface_common``:

  1. Dam break        (SDAA:20) — front position vs Martin & Moyce (1952)
  2. Sloshing tank    (SDAA:21) — wave frequency vs linear theory
  3. Rayleigh-Taylor  (SDAA:22) — mixing-layer growth vs linear theory
  4. Bubble rise      (SDAA:23) — terminal velocity vs Clift-Gauvin

Usage:
  python _free_surface_vof_worker.py <benchmark> <device_id> [output_path]

Examples:
  python _free_surface_vof_worker.py dam_break 20
  python _free_surface_vof_worker.py sloshing 21
  python _free_surface_vof_worker.py rayleigh_taylor 22
  python _free_surface_vof_worker.py bubble 23
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch_sdaa  # noqa: F401

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.free_surface_common import (
    bubble_centroid_velocity_3d,
    free_surface_vof_step,
    front_position_3d,
    init_phi_bubble_3d,
    init_phi_column_3d,
    init_phi_rayleigh_taylor_3d,
    init_phi_tilted_3d,
    mixing_layer_thickness_3d,
    wave_height_at_wall_3d,
)


# --------------------------------------------------------------------------- #
#  Container wall mask                                                         #
# --------------------------------------------------------------------------- #


def make_container_walls_3d(nz, ny, nx, device):
    """No-slip walls on all 6 faces of a closed tank."""
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, :, 0] = True    # left wall
    solid[:, :, -1] = True   # right wall
    solid[:, 0, :] = True    # front wall (y=0, floor)
    solid[:, -1, :] = True   # back wall (y=ny-1, ceiling)
    solid[0, :, :] = True    # z=0
    solid[-1, :, :] = True  # z=nz-1
    return solid


# --------------------------------------------------------------------------- #
#  Benchmark 1: Dam break (SDAA:20)                                           #
# --------------------------------------------------------------------------- #


def run_dam_break(device_id: int, output_path: str | None = None) -> dict:
    """Dam break — water column collapse.

    Reference: Martin & Moyce (1952) dam-break experiment.
    The dimensionless front position z* = x_front / a (where a is the
    initial column width) follows an approximate power law:
        z* ≈ 1.0 + 1.25 * sqrt(g * t / a)   (simplified correlation)

    Target: front position within 20% of the correlation at t* = t*sqrt(g/a).
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[DamBreak SDAA:{device_id}]"

    # Grid: nx=200, ny=100, nz=4
    nz, ny, nx = 4, 100, 200
    # Initial column: width=50, height=50 (aspect ratio 1:1)
    col_w, col_h = 50, 50

    # Lattice parameters
    tau = 1.0
    rho_l = 1.0
    rho_g = 0.01
    # Gravity in lattice units: gy negative (y=0 is floor)
    gy_lattice = -1e-4  # small for stability
    g_phys = abs(gy_lattice)

    # Physical reference: Martin & Moyce correlation
    # z* = x_front / a,  t* = t * sqrt(g/a)
    # For a/a0 = 1 (square column): z* ≈ 1.0 + 1.25 * t*
    a = col_w  # initial column width

    solid = make_container_walls_3d(nz, ny, nx, device)
    phi = init_phi_column_3d(nz, ny, nx, width=col_w, height=col_h, device=device)
    # Mask solid cells in phi
    phi = phi.masked_fill(solid, 0.0)

    # Initialize f with equilibrium at rest
    rho_field = rho_l * phi + rho_g * (1.0 - phi)
    ux = torch.zeros((nz, ny, nx), device=device)
    uy = torch.zeros((nz, ny, nx), device=device)
    uz = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho_field, ux, uy, uz, device=device)

    n_steps = 3000
    sample_interval = 50
    front_history = []
    time_history = []

    print(f"{tag} Starting dam break: {nx}x{ny}x{nz}, col={col_w}x{col_h}, "
          f"gy={gy_lattice}, steps={n_steps}")

    t0 = time.time()
    for step in range(n_steps):
        f, phi = free_surface_vof_step(
            f, phi, tau=tau, gy=gy_lattice,
            rho_liquid=rho_l, rho_gas=rho_g, solid=solid,
        )

        if step % sample_interval == 0:
            front = front_position_3d(phi)
            t_star = step * math.sqrt(g_phys / a)
            z_star = front / a
            front_history.append(z_star)
            time_history.append(t_star)
            if step % 500 == 0:
                elapsed = time.time() - t0
                print(f"{tag} step={step:5d} front={front:6.1f} z*={z_star:.3f} "
                      f"t*={t_star:.3f} ({elapsed:.1f}s)")

    # Reference: Martin & Moyce correlation at final time
    t_final = time_history[-1]
    z_ref = 1.0 + 1.25 * t_final  # simplified correlation
    z_sim = front_history[-1]
    error_pct = abs(z_sim - z_ref) / max(z_ref, 1e-6) * 100.0

    result = {
        "benchmark": "dam_break",
        "device": device_id,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "column": {"width": col_w, "height": col_h},
        "parameters": {"tau": tau, "rho_liquid": rho_l, "rho_gas": rho_g,
                        "gy": gy_lattice, "n_steps": n_steps},
        "front_position_z_star_final": z_sim,
        "reference_z_star": z_ref,
        "error_pct": error_pct,
        "target_pct": 20.0,
        "pass": error_pct < 20.0,
        "front_history": front_history,
        "time_history": time_history,
        "elapsed_s": time.time() - t0,
    }

    print(f"{tag} RESULT: z*_sim={z_sim:.3f} z*_ref={z_ref:.3f} "
          f"error={error_pct:.1f}% {'PASS' if result['pass'] else 'FAIL'}")

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


# --------------------------------------------------------------------------- #
#  Benchmark 2: Sloshing tank (SDAA:21)                                        #
# --------------------------------------------------------------------------- #


def run_sloshing(device_id: int, output_path: str | None = None) -> dict:
    """Sloshing tank — tilted initial free surface.

    Reference: linear sloshing theory.
    The natural frequency of the first sloshing mode in a tank of
    width L with fluid depth h is:
        omega_1 = sqrt(g * k * tanh(k * h))
    where k = pi / L (first mode).
    Period T = 2*pi / omega_1.

    Target: frequency within 10% of linear theory.
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[Sloshing SDAA:{device_id}]"

    nz, ny, nx = 4, 100, 200
    fill_frac = 0.5   # half-full tank
    angle_deg = 1.0   # small tilt for linear regime

    tau = 1.0
    rho_l = 1.0
    rho_g = 0.01
    gy_lattice = -1e-4
    g_phys = abs(gy_lattice)

    L = nx - 2  # interior width
    h = fill_frac * ny  # fluid depth

    # Linear theory: omega_1 = sqrt(g * k * tanh(k*h)), k = pi/L
    k1 = math.pi / L
    omega_ref = math.sqrt(g_phys * k1 * math.tanh(k1 * h))
    T_ref = 2 * math.pi / omega_ref
    freq_ref = omega_ref / (2 * math.pi)

    solid = make_container_walls_3d(nz, ny, nx, device)
    phi = init_phi_tilted_3d(nz, ny, nx, fill_frac, angle_deg, device)
    phi = phi.masked_fill(solid, 0.0)

    rho_field = rho_l * phi + rho_g * (1.0 - phi)
    ux = torch.zeros((nz, ny, nx), device=device)
    uy = torch.zeros((nz, ny, nx), device=device)
    uz = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho_field, ux, uy, uz, device=device)

    n_steps = 5000
    sample_interval = 20
    wave_history = []

    print(f"{tag} Starting sloshing: {nx}x{ny}x{nz}, fill={fill_frac}, "
          f"angle={angle_deg}°, omega_ref={omega_ref:.6f}, T_ref={T_ref:.1f}")

    t0 = time.time()
    for step in range(n_steps):
        f, phi = free_surface_vof_step(
            f, phi, tau=tau, gy=gy_lattice,
            rho_liquid=rho_l, rho_gas=rho_g, solid=solid,
        )

        if step % sample_interval == 0:
            h_left = wave_height_at_wall_3d(phi, wall="left")
            h_right = wave_height_at_wall_3d(phi, wall="right")
            wave_history.append({"step": step, "h_left": h_left, "h_right": h_right})
            if step % 500 == 0:
                print(f"{tag} step={step:5d} h_left={h_left:.1f} "
                      f"h_right={h_right:.1f} ({time.time()-t0:.1f}s)")

    # Extract frequency from wave height oscillation via FFT
    h_left_arr = [w["h_left"] for w in wave_history]
    h_right_arr = [w["h_right"] for w in wave_history]
    # Use the difference (antisymmetric mode)
    diff = [l - r for l, r in zip(h_left_arr, h_right_arr)]

    # Simple zero-crossing frequency estimation
    crossings = 0
    for i in range(1, len(diff)):
        if diff[i] * diff[i - 1] < 0:
            crossings += 1
    if crossings > 0:
        # Each full period = 2 zero crossings
        sim_periods = crossings / 2.0
        sim_time = n_steps  # lattice time
        T_sim = sim_time / sim_periods
        freq_sim = 1.0 / T_sim
    else:
        T_sim = 0.0
        freq_sim = 0.0

    error_pct = abs(freq_sim - freq_ref) / max(freq_ref, 1e-12) * 100.0

    result = {
        "benchmark": "sloshing",
        "device": device_id,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "parameters": {"tau": tau, "fill_frac": fill_frac, "angle_deg": angle_deg,
                        "gy": gy_lattice, "n_steps": n_steps},
        "frequency_sim": freq_sim,
        "frequency_ref": freq_ref,
        "period_sim": T_sim,
        "period_ref": T_ref,
        "error_pct": error_pct,
        "target_pct": 10.0,
        "pass": error_pct < 10.0,
        "wave_history": wave_history[-50:],
        "elapsed_s": time.time() - t0,
    }

    print(f"{tag} RESULT: freq_sim={freq_sim:.6f} freq_ref={freq_ref:.6f} "
          f"error={error_pct:.1f}% {'PASS' if result['pass'] else 'FAIL'}")

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


# --------------------------------------------------------------------------- #
#  Benchmark 3: Rayleigh-Taylor (SDAA:22)                                     #
# --------------------------------------------------------------------------- #


def run_rayleigh_taylor(device_id: int, output_path: str | None = None) -> dict:
    """Rayleigh-Taylor instability — heavy fluid on top.

    Reference: linear theory growth rate.
    The linear growth rate of the RT instability is:
        gamma = sqrt(A * g * k)
    where A = (rho_h - rho_l)/(rho_h + rho_l) is the Atwood number,
    g is gravity, and k = 2*pi/lambda is the wavenumber.
    The mixing layer grows as: h(t) ~ exp(gamma * t) in the linear regime.

    Target: growth rate within 30% of linear theory.
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[RayleighTaylor SDAA:{device_id}]"

    nz, ny, nx = 4, 100, 200
    interface_frac = 0.5
    amplitude = 2.0  # initial perturbation amplitude
    wavelength = float(nx - 2)  # one wavelength across the domain

    tau = 1.0
    rho_l = 1.0    # heavy fluid (top, phi=1)
    rho_g = 0.3    # light fluid (bottom, phi=0)
    gy_lattice = -1e-4  # gravity downward (toward y=0)
    g_phys = abs(gy_lattice)

    # Atwood number
    A_atwood = (rho_l - rho_g) / (rho_l + rho_g)
    k = 2 * math.pi / wavelength
    gamma_ref = math.sqrt(A_atwood * g_phys * k)

    solid = make_container_walls_3d(nz, ny, nx, device)
    phi = init_phi_rayleigh_taylor_3d(
        nz, ny, nx, interface_frac, amplitude, wavelength, device
    )
    phi = phi.masked_fill(solid, 0.0)

    rho_field = rho_l * phi + rho_g * (1.0 - phi)
    ux = torch.zeros((nz, ny, nx), device=device)
    uy = torch.zeros((nz, ny, nx), device=device)
    uz = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho_field, ux, uy, uz, device=device)

    n_steps = 4000
    sample_interval = 50
    mixing_history = []

    print(f"{tag} Starting RT: {nx}x{ny}x{nz}, A={A_atwood:.3f}, "
          f"gamma_ref={gamma_ref:.6f}, k={k:.4f}")

    t0 = time.time()
    for step in range(n_steps):
        # For RT, gravity pulls heavy fluid (phi=1) downward
        # We need gravity in ALL cells, not just phi>0.5
        # Use a modified step: apply gravity everywhere
        f, phi = free_surface_vof_step(
            f, phi, tau=tau, gy=gy_lattice,
            rho_liquid=rho_l, rho_gas=rho_g, solid=solid,
        )

        if step % sample_interval == 0:
            mix = mixing_layer_thickness_3d(phi)
            mixing_history.append({"step": step, "mixing": mix})
            if step % 500 == 0:
                print(f"{tag} step={step:5d} mixing={mix:.1f} "
                      f"({time.time()-t0:.1f}s)")

    # Estimate growth rate from mixing history (linear regime)
    # h(t) = h0 * exp(gamma * t)  =>  ln(h) = ln(h0) + gamma*t
    import numpy as np
    steps_arr = np.array([m["step"] for m in mixing_history[2:]])
    mix_arr = np.array([m["mixing"] for m in mixing_history[2:]])
    # Filter out zero/negative
    valid = mix_arr > 0
    if valid.sum() > 5:
        log_mix = np.log(mix_arr[valid])
        steps_v = steps_arr[valid]
        # Linear fit: log(h) = a + gamma * t
        coeffs = np.polyfit(steps_v, log_mix, 1)
        gamma_sim = coeffs[0]
    else:
        gamma_sim = 0.0

    error_pct = abs(gamma_sim - gamma_ref) / max(gamma_ref, 1e-12) * 100.0

    result = {
        "benchmark": "rayleigh_taylor",
        "device": device_id,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "parameters": {"tau": tau, "rho_heavy": rho_l, "rho_light": rho_g,
                        "gy": gy_lattice, "amplitude": amplitude,
                        "wavelength": wavelength, "n_steps": n_steps},
        "atwood_number": A_atwood,
        "growth_rate_sim": gamma_sim,
        "growth_rate_ref": gamma_ref,
        "error_pct": error_pct,
        "target_pct": 30.0,
        "pass": error_pct < 30.0,
        "mixing_history": mixing_history[-30:],
        "elapsed_s": time.time() - t0,
    }

    print(f"{tag} RESULT: gamma_sim={gamma_sim:.6f} gamma_ref={gamma_ref:.6f} "
          f"error={error_pct:.1f}% {'PASS' if result['pass'] else 'FAIL'}")

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


# --------------------------------------------------------------------------- #
#  Benchmark 4: Bubble rise (SDAA:23)                                         #
# --------------------------------------------------------------------------- #


def run_bubble_rise(device_id: int, output_path: str | None = None) -> dict:
    """Bubble rise — air bubble in water.

    Reference: Clift-Gauvin correlation for terminal velocity of a
    rising bubble in the Stokes/intermediate regime:
        U_t = (2/9) * g * R^2 * (rho_l - rho_g) / mu   (Stokes, Re << 1)
    For moderate Re, use the Clift-Gauvin correlation:
        Re = (rho_l * U_t * d) / mu
        Cd = (24/Re) * (1 + 0.15*Re^0.687) + 0.42/(1 + 4.25e4/Re^1.16)
        U_t = sqrt(2*g*(rho_l-rho_g)*V / (rho_l * A * Cd))

    For simplicity, we use the Stokes terminal velocity as reference
    and target within 30%.

    Target: terminal velocity within 30% of reference.
    """
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)
    tag = f"[BubbleRise SDAA:{device_id}]"

    nz, ny, nx = 4, 100, 200
    # Bubble at the bottom centre
    cx, cy, cz = nx / 2, 25, nz / 2
    radius = 10.0

    tau = 1.0
    rho_l = 1.0    # water
    rho_g = 0.1    # air bubble (lighter)
    gy_lattice = -1e-4  # gravity downward
    g_phys = abs(gy_lattice)

    # Stokes terminal velocity: U_t = (2/9) * g * R^2 * (rho_l - rho_g) / mu
    # In lattice units: mu = (tau - 0.5) / 3 = (1.0 - 0.5)/3 = 1/6
    mu_lattice = (tau - 0.5) / 3.0
    U_stokes = (2.0 / 9.0) * g_phys * radius ** 2 * (rho_l - rho_g) / mu_lattice

    solid = make_container_walls_3d(nz, ny, nx, device)
    phi = init_phi_bubble_3d(nz, ny, nx, cx, cy, cz, radius, device)
    phi = phi.masked_fill(solid, 0.0)

    rho_field = rho_l * phi + rho_g * (1.0 - phi)
    ux = torch.zeros((nz, ny, nx), device=device)
    uy = torch.zeros((nz, ny, nx), device=device)
    uz = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho_field, ux, uy, uz, device=device)

    n_steps = 4000
    sample_interval = 50
    bubble_history = []

    print(f"{tag} Starting bubble rise: {nx}x{ny}x{nz}, R={radius}, "
          f"U_stokes={U_stokes:.6f}, mu={mu_lattice:.4f}")

    t0 = time.time()
    for step in range(n_steps):
        f, phi = free_surface_vof_step(
            f, phi, tau=tau, gy=gy_lattice,
            rho_liquid=rho_l, rho_gas=rho_g, solid=solid,
        )

        if step % sample_interval == 0:
            rho_f, ux_f, uy_f, uz_f = macroscopic3d(f)
            cx_b, cy_b, cz_b, vx_b, vy_b, vz_b = bubble_centroid_velocity_3d(
                phi, ux_f, uy_f, uz_f
            )
            bubble_history.append({
                "step": step, "cx": cx_b, "cy": cy_b, "cz": cz_b,
                "vy": vy_b,
            })
            if step % 500 == 0:
                print(f"{tag} step={step:5d} cy={cy_b:.1f} vy={vy_b:.6f} "
                      f"({time.time()-t0:.1f}s)")

    # Estimate terminal velocity from the last 1/3 of the simulation
    import numpy as np
    n_hist = len(bubble_history)
    late = bubble_history[n_hist // 3:]
    if len(late) > 5:
        steps_arr = np.array([b["step"] for b in late])
        vy_arr = np.array([b["vy"] for b in late])
        # Average velocity in the late stage (terminal)
        U_sim = float(np.mean(vy_arr))
        # Also try linear fit of cy vs step
        cy_arr = np.array([b["cy"] for b in late])
        if len(steps_arr) > 2:
            slope = np.polyfit(steps_arr, cy_arr, 1)[0]
            U_sim_fit = float(slope)
        else:
            U_sim_fit = U_sim
    else:
        U_sim = 0.0
        U_sim_fit = 0.0

    # Use the fit-based velocity (more robust)
    U_sim_final = U_sim_fit if abs(U_sim_fit) > 1e-12 else U_sim
    # Bubble rises upward, so vy should be positive (toward higher y)
    # But gravity is -y, so bubble (lighter) rises in +y
    U_sim_abs = abs(U_sim_final)

    error_pct = abs(U_sim_abs - abs(U_stokes)) / max(abs(U_stokes), 1e-12) * 100.0

    result = {
        "benchmark": "bubble_rise",
        "device": device_id,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "parameters": {"tau": tau, "rho_liquid": rho_l, "rho_gas": rho_g,
                        "gy": gy_lattice, "radius": radius,
                        "n_steps": n_steps},
        "terminal_velocity_sim": U_sim_abs,
        "terminal_velocity_ref": abs(U_stokes),
        "error_pct": error_pct,
        "target_pct": 30.0,
        "pass": error_pct < 30.0,
        "bubble_history": bubble_history[-30:],
        "elapsed_s": time.time() - t0,
    }

    print(f"{tag} RESULT: U_sim={U_sim_abs:.6f} U_ref={abs(U_stokes):.6f} "
          f"error={error_pct:.1f}% {'PASS' if result['pass'] else 'FAIL'}")

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

BENCHMARKS = {
    "dam_break": run_dam_break,
    "sloshing": run_sloshing,
    "rayleigh_taylor": run_rayleigh_taylor,
    "bubble": run_bubble_rise,
}


def main():
    if len(sys.argv) < 3:
        print("Usage: python _free_surface_vof_worker.py <benchmark> <device_id> [output_path]")
        print(f"Available benchmarks: {list(BENCHMARKS.keys())}")
        sys.exit(1)

    bench = sys.argv[1]
    dev = int(sys.argv[2])
    out = sys.argv[3] if len(sys.argv) > 3 else None

    if bench not in BENCHMARKS:
        print(f"Unknown benchmark '{bench}'. Available: {list(BENCHMARKS.keys())}")
        sys.exit(1)

    result = BENCHMARKS[bench](dev, out)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("front_history", "wave_history",
                                   "mixing_history", "bubble_history")},
                      indent=2))


if __name__ == "__main__":
    main()
