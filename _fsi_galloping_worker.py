#!/usr/bin/env python3
"""FSI Test 2: Galloping of a square prism.

SDAA:25  |  Re=1000, D=24, m*=5, zeta=0.1, f_n=0.1

Pipeline (common-interface only):
  solid → get_near_wall_3d → SurfaceMesh.from_square_prism →
  lbm_step_correct → fsi_step_drag (drag_pressure + drag_friction → SMD) →
  shift_solid_mask (moving boundary) → detect_strouhal

Galloping is a self-excited oscillation driven by negative aerodynamic
damping (Den Hartog criterion).  The transverse force slope dCl/dalpha < 0
at alpha=0 triggers galloping when the structural damping is insufficient.

Target: detect galloping onset (transverse displacement grows without bound
or reaches a large-amplitude limit cycle).

Usage:
  python _fsi_galloping_worker.py <device_id> [output_path]
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

# ---- Common interface imports (ONLY these modules) ----
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d
from tensorlbm.lbm_step_correct import lbm_step_correct
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
)
from tensorlbm.fsi_common import (
    SpringMassDamper,
    SpringMassState,
    fsi_step_drag,
    shift_solid_mask,
)
from tensorlbm.postprocess import detect_strouhal


def build_square_prism_mask(nx, ny, nz, cx, cy, D, device):
    """Square prism (D×D cross-section, extruded along z)."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    half = D // 2
    solid[:, cy - half:cy + half, cx - half:cx + half] = True
    return solid


def run_galloping(device_id, output_path=None):
    """Galloping prism: Re=1000, D=24, m*=5, zeta=0.1, f_n=0.1."""
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    # --- Grid & flow parameters ---
    nx, ny, nz = 300, 100, 4
    D = 24                          # prism side length (lattice units)
    cx = nx // 4
    cy = ny // 2
    Re = 1000
    u_in = 0.1
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    n_steps = 30000
    tag = f"[Galloping SDAA:{device_id}]"

    # --- Structural parameters ---
    m_star = 5.0
    zeta = 0.1
    f_n = 0.1
    rho_f = 1.0

    smd = SpringMassDamper.from_mass_ratio_freq(
        mass_ratio=m_star, rho_f=rho_f, D=float(D), u_in=u_in,
        f_n=f_n, zeta=zeta, n_dof=1,
    )
    print(f"{tag} m*={m_star} zeta={zeta} f_n={f_n} "
          f"m={smd.mass:.2f} k={smd.stiffness:.4f} c={smd.damping:.4f} "
          f"f_n_check={smd.natural_frequency:.6f} zeta_check={smd.damping_ratio:.6f}",
          flush=True)

    dpS = 0.5 * u_in ** 2 * D * nz

    print(f"{tag} nx={nx} ny={ny} nz={nz} D={D} Re={Re} u_in={u_in} "
          f"nu={nu:.6e} tau={tau:.6f} dpS={dpS:.6e}", flush=True)

    t0 = time.time()
    solid = build_square_prism_mask(nx, ny, nz, cx, cy, D, device)

    # --- Common interface: get_near_wall_3d ---
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells={n_near}", flush=True)

    # --- Common interface: SurfaceMesh.from_square_prism ---
    mesh = SurfaceMesh.from_square_prism(solid, near, float(cx), float(cy), float(D))
    print(f"{tag} SurfaceMesh.from_square_prism built", flush=True)

    # --- Far-field BC ---
    bc_config = {'far_field_faces': ['y+'], 'periodic_faces': ['z-', 'z+']}
    far_field_fn = functools.partial(far_field_bc_3d, bc_config=bc_config)

    # --- Initialise flow ---
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time()-t0:.1f}s)", flush=True)

    # --- Structure state ---
    sm_state = SpringMassState.zero(1, dtype=torch.float64)

    # --- Histories ---
    cd_p_hist, cd_f_hist, cd_tot_hist, cl_hist = [], [], [], []
    disp_hist, vel_hist = [], []

    # --- Main loop ---
    ramp_steps = 1000
    for step in range(1, n_steps + 1):
        u_cur = min(u_in, 0.02 + (u_in - 0.02) * step / ramp_steps)

        f = lbm_step_correct(
            f, collide_mrt3d, tau, solid, u_cur,
            far_field_fn, correct_mass_fn=correct_mass3d, target_mass=im,
            step=step, mass_interval=200,
        )

        result = fsi_step_drag(
            f, mesh, dpS, nu, sm_state,
            smd=smd, dt=1.0, force_axis=1,
            extrap='none', p0_method='far_field', solid=solid,
            friction_formula='standard',
        )
        sm_state = result.structure_updated

        # Moving boundary
        disp_y = sm_state.disp[0].item()
        dy_shift = int(round(disp_y))
        if abs(dy_shift) >= 1:
            solid = shift_solid_mask(solid, dx=0, dy=dy_shift, dz=0)
            near = get_near_wall_3d(solid)
            mesh = SurfaceMesh.from_square_prism(
                solid, near, float(cx), float(cy + disp_y), float(D))

        cd_p = result.cd_pressure[0]
        cd_f = result.cd_friction[0]
        cd_tot = result.cd_total[0]
        cl = result.cd_total[1]
        cd_p_hist.append(cd_p)
        cd_f_hist.append(cd_f)
        cd_tot_hist.append(cd_tot)
        cl_hist.append(cl)
        disp_hist.append(disp_y)
        vel_hist.append(sm_state.vel[0].item())

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break
        if step % 2000 == 0:
            n_avg = min(500, len(cd_tot_hist))
            at = sum(cd_tot_hist[-n_avg:]) / n_avg
            cl_rms = math.sqrt(sum(c**2 for c in cl_hist[-n_avg:]) / n_avg)
            disp_max = max(abs(d) for d in disp_hist[-n_avg:])
            print(f"{tag} step={step}/{n_steps} Cd={at:.4f} Cl_rms={cl_rms:.4f} "
                  f"disp_max={disp_max:.2f} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # --- Post-processing ---
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

    # Galloping onset detection: check if displacement grows monotonically
    # in the last 25% of the simulation (vs first 25% after transient)
    n_quart = len(disp_hist) // 4
    early_disp = np.max(np.abs(disp_hist[n_half:n_half + n_quart]))
    late_disp = np.max(np.abs(disp_hist[-n_quart:]))
    galloping_detected = late_disp > 2.0 * early_disp and late_disp > D * 0.1

    print(f"\n{tag} === FINAL ===  A/D={A_D:.4f} disp_rms/D={disp_rms:.4f} "
          f"Cd_mean={cd_mean:.4f} Cl_rms={cl_rms:.4f} St={st} "
          f"galloping={'YES' if galloping_detected else 'NO'} "
          f"early_disp={early_disp:.2f} late_disp={late_disp:.2f} "
          f"time={elapsed:.0f}s", flush=True)

    result_dict = {
        "case": "galloping_prism_Re1000",
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
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": float(elapsed),
        "modules_used": [
            "drag_pressure.get_near_wall_3d",
            "drag_pressure.SurfaceMesh.from_square_prism",
            "lbm_step_correct.lbm_step_correct",
            "fsi_common.fsi_step_drag",
            "fsi_common.SpringMassDamper",
            "fsi_common.shift_solid_mask",
            "postprocess.detect_strouhal",
        ],
    }

    if output_path:
        Path(output_path).write_text(json.dumps(result_dict, indent=2))
        print(f"{tag} results saved to {output_path}", flush=True)

    return result_dict


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_galloping(dev, out)
