"""W9-B P1-AC: Allen-Cahn module probe — static stability, spurious currents,
gravity rise plateau quality, and lambda-reachability. First-ever exercise of
allen_cahn_lbm (no tests/benchmarks use it).

AC-1: static lambda=0.5 (r=0.5): volume drift, aspect, spurious max|u|, rho consistency
AC-0: static lambda in {0.1, 10.0}: reachability (3000 steps, stability only)
AC-2: gravity lambda=0.5 r=0.5 Re~0.05: U(t) centroid rise, plateau, travel budget
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))
from tensorlbm.allen_cahn_lbm import allen_cahn_step  # noqa: E402
from tensorlbm.boundaries3d import (  # noqa: E402
    make_channel_wall_mask_3d,
    make_tank_wall_mask_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402

OUT = Path(__file__).resolve().parent
DEV = torch.device("cuda")
L, A_RAD = 96, 12
W_INT = 4.0
RHO_O, NU_O = 1.0, 1.0 / 6.0  # ambient (phi=+1)
RHO_I = 0.5  # droplet (phi=-1)

z, y, x = torch.meshgrid(
    torch.arange(L, device=DEV, dtype=torch.float32),
    torch.arange(L, device=DEV, dtype=torch.float32),
    torch.arange(L, device=DEV, dtype=torch.float32),
    indexing="ij",
)
r2 = (x - L / 2 + 0.5) ** 2 + (y - L / 4 + 0.5) ** 2 + (z - L / 2 + 0.5) ** 2
PHI0 = torch.tanh((r2.sqrt() - A_RAD) / (W_INT / 2))  # -1 inside droplet (light, rho_l)
EMPTY = torch.zeros((L, L, L), dtype=torch.bool, device=DEV)
WALL = make_channel_wall_mask_3d(L, L, L, EMPTY, DEV) | make_tank_wall_mask_3d(L, L, L, EMPTY, DEV)
YGRID = y


def init_fields(nu_i):
    rho = RHO_I + (RHO_O - RHO_I) * (1.0 + PHI0) / 2.0  # phi=-1 -> rho_l droplet
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, zero, zero, zero)
    # g = phase-field equilibrium at rest (same form as library init)
    from tensorlbm.allen_cahn_lbm import _phase_field_equilibrium

    g = _phase_field_equilibrium(PHI0, zero, zero, zero, DEV)
    return f, g, PHI0.clone()


def diags(f, phi, step, nu_i):
    rho, ux, uy, uz = macroscopic3d(f)
    u = (ux**2 + uy**2 + uz**2).sqrt()
    fluid = ~WALL
    drop = phi < -0.5
    nd = int(drop.sum())
    w = torch.clamp(-phi, min=0.0)
    y_com = float((w * YGRID).sum() / w.sum())
    if nd > 0:
        ys, xs = drop.any(-1).any(-1), drop.any(0).any(-1)
        y_ext = int(ys.sum())
        x_ext = int(xs.sum())
        aspect = y_ext / max(x_ext, 1)
        u_in = float(uy[drop].mean())
        rho_in = float(rho[drop].mean())
    else:
        aspect, u_in, rho_in = -1.0, float("nan"), float("nan")
    r_eq = (3.0 * nd / (4.0 * math.pi)) ** (1.0 / 3.0)
    return {
        "step": step,
        "R_eq": r_eq,
        "y_com": y_com,
        "aspect": aspect,
        "max_u": float(u[fluid].max()),
        "u_y_in": u_in,
        "rho_in": rho_in,
        "rho_mix_in": RHO_I,
    }


def run_static(lam, n_steps, tag):
    nu_i = lam * NU_O * RHO_O / RHO_I  # lambda = rho_i nu_i / (rho_o nu_o)
    f, g, phi = init_fields(nu_i)
    trace = [diags(f, phi, 0, nu_i)]
    t0 = time.time()
    for step in range(1, n_steps + 1):
        f, g, phi = allen_cahn_step(
            f,
            g,
            phi,
            rho_h=RHO_O,
            rho_l=RHO_I,
            nu_h=NU_O,
            nu_l=nu_i,
            sigma=0.0,
            W=W_INT,
            solid_mask=WALL,
        )
        if step % 250 == 0:
            if torch.isnan(f).any() or torch.isnan(phi).any():
                print(f"[{tag}] NaN at {step}", flush=True)
                break
            trace.append(diags(f, phi, step, nu_i))
            d = trace[-1]
            print(
                f"[{tag}] {step:6d} R_eq={d['R_eq']:.3f} aspect={d['aspect']:.3f} "
                f"max_u={d['max_u']:.2e} u_y_in={d['u_y_in']:.2e} rho_in={d['rho_in']:.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return trace


def run_gravity(lam, n_steps, g_mag, tag):
    nu_i = lam * NU_O * RHO_O / RHO_I
    f, g, phi = init_fields(nu_i)
    trace = [diags(f, phi, 0, nu_i)]
    t0 = time.time()
    for step in range(1, n_steps + 1):
        f, g, phi = allen_cahn_step(
            f,
            g,
            phi,
            rho_h=RHO_O,
            rho_l=RHO_I,
            nu_h=NU_O,
            nu_l=nu_i,
            sigma=0.0,
            W=W_INT,
            gx=0.0,
            gy=-g_mag,
            gz=0.0,
            solid_mask=WALL,
        )
        if step % 250 == 0:
            if torch.isnan(f).any() or torch.isnan(phi).any():
                print(f"[{tag}] NaN at {step}", flush=True)
                break
            trace.append(diags(f, phi, step, nu_i))
            d = trace[-1]
            print(
                f"[{tag}] {step:6d} y_com={d['y_com']:.3f} R_eq={d['R_eq']:.3f} "
                f"aspect={d['aspect']:.3f} max_u={d['max_u']:.2e} u_y_in={d['u_y_in']:.2e} "
                f"rho_in={d['rho_in']:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    return trace


if __name__ == "__main__":
    results = {}
    results["AC1_static_lam0.5"] = run_static(0.5, 8000, "AC1")
    results["AC0_static_lam0.1"] = run_static(0.1, 3000, "AC0a")
    results["AC0_static_lam10"] = run_static(10.0, 3000, "AC0b")
    # U_t = 123.4286*g ; Re=0.05 -> U=3.472e-4 -> g = 2.8133e-6
    results["AC2_grav_lam0.5"] = run_gravity(0.5, 40000, 2.8133e-6, "AC2")
    with open(OUT / "p1_ac.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print("P1-AC complete")
