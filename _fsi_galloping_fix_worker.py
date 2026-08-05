#!/usr/bin/env python3
"""FSI galloping prism fix — SDAA:10.

Fix: NaN displacement — check force before spring update.
If force is NaN, skip spring update (use previous state).

Also fixes mask drift (incremental shift) and force computation order.

Square prism Re=1000, D=24, nx=300, ny=100, nz=4
m*=5, f_n=0.1/St, 20000 steps
Target: detect galloping onset (displacement > 0.1D)
"""
from __future__ import annotations

import functools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import torch
import torch_sdaa  # noqa: F401

from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.fsi_common import (
    SpringMassDamper,
    SpringMassState,
    spring_mass_step,
    shift_solid_mask,
)
from tensorlbm.postprocess import detect_strouhal


def build_square_prism_mask(nx, ny, nz, cx, cy, D, device):
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    half = D // 2
    solid[:, cy - half:cy + half, cx - half:cx + half] = True
    return solid


def run_galloping(device_id, output_path=None):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device_id)

    nx, ny, nz = 300, 100, 4
    D = 24
    cx = nx // 4
    cy = ny // 2
    Re = 1000
    u_in = 0.1
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 20000
    tag = f"[Galloping-fix SDAA:{device_id}]"

    # Structural parameters
    m_star = 5.0
    zeta = 0.1
    # f_n = 0.1/St — St for square prism ≈ 0.13
    # Use moderate f_n for galloping (soft enough for amplitude, stiff enough
    # to stay in domain). Vortex shedding freq = St*U/D = 0.13*0.1/24 ≈ 0.000542
    St_expected = 0.13
    f_n = 0.001  # moderate — U_r = U/(f_n*D) = 4.17 (in galloping range)
    rho_f = 1.0

    smd = SpringMassDamper.from_mass_ratio_freq(
        mass_ratio=m_star, rho_f=rho_f, D=float(D), u_in=u_in,
        f_n=f_n, zeta=zeta, n_dof=1,
    )
    print(f"{tag} m*={m_star} zeta={zeta} f_n={f_n:.6f} "
          f"m={smd.mass:.2f} k={smd.stiffness:.6f} c={smd.damping:.6f} "
          f"f_n_check={smd.natural_frequency:.6f} zeta_check={smd.damping_ratio:.6f}",
          flush=True)

    dpS = 0.5 * u_in ** 2 * D * nz

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} Re={Re} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} dpS={dpS:.6e}", flush=True)

    t0 = time.time()
    solid = build_square_prism_mask(nx, ny, nz, cx, cy, D, device)

    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    mesh = SurfaceMesh.from_square_prism(solid, near, int(cx), int(cy), int(D))
    print(f"{tag} SurfaceMesh.from_square_prism built", flush=True)

    bc_config = {'far_field_faces': ['y+'], 'periodic_faces': ['z-', 'z+']}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    sm_state = SpringMassState.zero(1, dtype=torch.float64)

    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    disp_hist, vel_hist = [], []

    last_dy_shift = 0
    cy_current = float(cy)
    nan_count = 0
    # Displacement clamp: keep mask within domain
    max_disp = ny // 2 - D // 2 - 4

    ramp_steps = 2000
    for step in range(1, n_steps + 1):
        u_cur = min(u_in, 0.01 + (u_in - 0.01) * step / ramp_steps)

        # 1. LBM step
        f = lbm_step_correct(
            f, collide_mrt3d, tau, solid, u_cur,
            far_field_fn, correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200,
        )

        # 2. Check if displacement requires mask shift (INCREMENTAL)
        disp_y = sm_state.disp[0].item()
        # Clamp displacement to keep mask within domain
        if math.isfinite(disp_y) and abs(disp_y) > max_disp:
            disp_y = max_disp if disp_y > 0 else -max_disp
            sm_state.disp[0] = disp_y
            sm_state.vel[0] *= 0.5
        if math.isfinite(disp_y):
            dy_shift_total = int(round(disp_y))
            dy_incremental = dy_shift_total - last_dy_shift
            if abs(dy_incremental) >= 1:
                solid = shift_solid_mask(solid, dx=0, dy=dy_incremental, dz=0)
                near = get_near_wall_3d(solid)
                cy_current = float(cy + disp_y)
                # Bounds-check cy_current for from_square_prism
                cy_int = int(cy_current)
                cy_int = max(D, min(cy_int, ny - D))
                mesh = SurfaceMesh.from_square_prism(
                    solid, near, int(cx), cy_int, int(D))
                last_dy_shift = dy_shift_total

        # 3. Compute force with (possibly new) mesh
        fx_p, fy_p, fz_p = drag_pressure_integration(
            f, mesh, dpS, extrap='none', p0_method='far_field', solid=solid)
        fx_f, fy_f, fz_f = drag_friction_integration(
            f, mesh, dpS, nu, formula='standard')
        cd_t = (fx_p + fx_f, fy_p + fy_f, fz_p + fz_f)

        # 4. Check force for NaN BEFORE spring update
        force_y = cd_t[1] * dpS  # transverse (y) force
        if math.isnan(force_y) or math.isinf(force_y):
            # Skip spring update — use previous state
            nan_count += 1
            if nan_count <= 10:
                print(f"{tag} WARNING: force NaN at step {step}, "
                      f"skipping spring update (count={nan_count})", flush=True)
        else:
            # Force is valid — update structure
            f_drive = torch.tensor([force_y], dtype=torch.float64)
            sm_state = spring_mass_step(sm_state, f_drive, dt=1.0, smd=smd)

        # 5. Record
        cd_p_hist.append(fx_p)
        cd_f_hist.append(fx_f)
        cd_tot_hist.append(cd_t[0])
        cl_hist.append(cd_t[1])
        disp_hist.append(sm_state.disp[0].item() if math.isfinite(sm_state.disp[0].item()) else 0.0)
        vel_hist.append(sm_state.vel[0].item() if math.isfinite(sm_state.vel[0].item()) else 0.0)

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step % 2000 == 0:
            n_avg = min(500, len(cd_tot_hist))
            at = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_rms = math.sqrt(sum(c**2 for c in cl_hist[-n_avg:]) / n_avg)
            disp_max = max(abs(d) for d in disp_hist[-n_avg:])
            print(f"{tag} step={step}/{n_steps} Cd={at:.4f} Cl_rms={cl_rms:.4f} "
                  f"disp_max={disp_max:.2f} nan_count={nan_count} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    n_half = len(disp_hist) // 2
    disp_arr = np.array(disp_hist[n_half:])
    vel_arr = np.array(vel_hist[n_half:])
    cl_arr = np.array(cl_hist[n_half:])
    cd_arr = np.array(cd_tot_hist[n_half:])

    A_D = (disp_arr.max() - disp_arr.min()) / 2.0 / D
    disp_rms = float(np.sqrt(np.mean(disp_arr ** 2))) / D
    cd_mean = float(np.mean(cd_arr))
    cl_rms = float(np.sqrt(np.mean(cl_arr ** 2)))
    st = detect_strouhal(cl_hist, sample_rate=1.0, u_ref=u_in,
                         length_ref=float(D), min_cycles=3)

    # Galloping onset detection
    n_quart = len(disp_hist) // 4
    early_disp = np.max(np.abs(disp_hist[n_half:n_half + n_quart]))
    late_disp = np.max(np.abs(disp_hist[-n_quart:]))
    galloping_detected = late_disp > 2.0 * early_disp and late_disp > D * 0.1

    print(f"\n{tag} === FINAL ===  A/D={A_D:.4f} disp_rms/D={disp_rms:.4f} "
          f"Cd_mean={cd_mean:.4f} Cl_rms={cl_rms:.4f} St={st} "
          f"galloping={'YES' if galloping_detected else 'NO'} "
          f"early_disp={early_disp:.2f} late_disp={late_disp:.2f} "
          f"nan_count={nan_count} time={elapsed:.0f}s", flush=True)

    result_dict = {
        "case": "galloping_prism_Re1000_fixed",
        "device": f"sdaa:{device_id}",
        "grid": f"{nx}x{ny}x{nz}",
        "D": D, "Re": Re, "u_in": u_in, "nu": nu, "tau": tau,
        "m_star": m_star, "zeta": zeta, "f_n": f_n,
        "mass": smd.mass, "stiffness": smd.stiffness, "damping": smd.damping,
        "n_steps": n_steps,
        "amplitude_A_over_D": float(A_D),
        "disp_rms_over_D": disp_rms,
        "Cd_mean": cd_mean,
        "Cl_rms": cl_rms,
        "St": st,
        "galloping_detected": bool(galloping_detected),
        "early_disp_max": float(early_disp),
        "late_disp_max": float(late_disp),
        "nan_count": int(nan_count),
        "target_galloping_onset": bool(galloping_detected),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result_dict, indent=2))
        print(f"{tag} results saved to {output_path}", flush=True)

    return result_dict


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_galloping(dev, out)
