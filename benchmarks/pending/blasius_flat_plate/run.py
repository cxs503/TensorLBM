#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B25: Blasius flat-plate laminar boundary layer (D2Q9) — mid-domain thin plate.

Physics: uniform stream U over a semi-infinite flat plate (leading edge at
x=x0, no-slip wall), laminar steady boundary layer. Compared against the
Blasius similarity solution f''' + f*f''/2 = 0:

    eta = (y - y_w) * sqrt(U/(nu*(x - x0)))          (y_w = py +/- 0.5)
    u/U = f'(eta)
    C_f  = 2*tau_w/(rho*U^2) = 0.664/sqrt(Re_x),  Re_x = U*(x-x0)/nu

FIX (2026-08-19, mid-plate design): the earlier bottom-wall plate put the
no-slip wall at y=0.5 while the symmetry plane sat at y=0 — a half-cell
'step' at the leading edge that produced a systematic C_f overshoot of
~+50%. Here the plate is a ONE-ROW THIN SOLID at the domain MIDDLE
(plate_y = ny/2), with fluid on both sides and mirror (specular) top/bottom
boundaries far away. Both surfaces use the fluid-side-reflection half-way
bounce-back (periodic stream() contaminates solid-row self-values; the
B13 pre-streaming f_pre[OPPOSITE] variant is only exact for symmetric
channels). Leading edge at x=20; outlet uses the library Zou-He pressure
BC (rho=1), verified zero mass drift at tau=0.53 (see
references/blasius-flat-plate-b25.md session 2).

True simulation, no extrapolation:
  - library primitives only: d2q9.equilibrium/macroscopic,
    solver.collide_bgk/stream, boundaries.zou_he_outlet_pressure
  - Blasius reference solved numerically (RK4 shooting) inside this script
  - no correction factors, no result tuning, extrap: none

Grids (>=2 to prove convergence; plate length 200/400, probe fixed at x=200):
  - plate200: nx=240 ny=1400 plate_y=700 le=20 L=200 probe=200 (x'=180)
  - plate400: nx=440 ny=1400 plate_y=700 le=20 L=400 probe=200 (x'=180)
  - plate400_y1600 (y-refinement): nx=440 ny=1600 plate_y=800 le=20 L=400 probe=200

Notes:
  * U=0.1 (Ma=0.173) makes the Zou-He outlet diverge at tau=0.53 (~800 steps,
    NaN) — verified on both ny=100 and ny=600. U=0.05 is the validated stable
    point (zero mass drift, session-2 record).
  * The mid plate develops boundary layers on BOTH sides, so blockage
    (u_edge/U - 1 ~= 2*delta*/ny) is twice that of the bottom-wall plate;
    delta* ~= 10.3 @ x'=180, so ny=1400 keeps blockage ~1.5%.

Usage:
    run.py <out.json> [--grid plate200|plate400|plate400_y1600] [--U 0.05]
           [--nu 0.01] [--steps 30000] [--smoke 3000] [--device cuda:1]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
import types as _types
from pathlib import Path

_SRC = "/home/wxsc/cxs/TensorLBM/src"
sys.path.insert(0, _SRC)

# NOTE(2026-08-19): tensorlbm/__init__.py currently imports .dg_lbm -> .physics ->
# ..thermal which is mid-refactor by a parallel agent (C_D2Q5 renamed to C5) and
# raises ImportError on package import. d2q9/boundaries/solver depend only on each
# other and torch, so load them directly via importlib. Still the exact library
# primitives — no hand-written kernels.


def _load_submodule(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, f"{_SRC}/tensorlbm/{rel_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_tlbm_pkg = _types.ModuleType("tensorlbm")
_tlbm_pkg.__path__ = [f"{_SRC}/tensorlbm"]
sys.modules["tensorlbm"] = _tlbm_pkg
_load_submodule("tensorlbm.d2q9", "d2q9.py")
_load_submodule("tensorlbm.boundaries", "boundaries.py")
_load_submodule("tensorlbm.solver", "solver.py")

import numpy as np
import torch

from tensorlbm.boundaries import zou_he_outlet_pressure
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream

torch.set_num_threads(32)

# Mirror (specular) direction map for symmetry BC: cy -> -cy.
# D2Q9: 0:(0,0) 1:(1,0) 2:(0,1) 3:(-1,0) 4:(0,-1) 5:(1,1) 6:(-1,1) 7:(-1,-1) 8:(1,-1)
SPEC = torch.tensor([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=torch.int64)

GRIDS = {
    "plate200": dict(nx=240, ny=1400, le=20, plate_len=200, probe=200),
    "plate400": dict(nx=440, ny=1400, le=20, plate_len=400, probe=200),
    "plate400_y1600": dict(nx=440, ny=1600, le=20, plate_len=400, probe=200),
}


def blasius_table(eta_max: float = 10.0, h: float = 0.005) -> tuple[np.ndarray, np.ndarray, float]:
    """Numerically solve f''' + 0.5*f*f'' = 0, f(0)=f'(0)=0, f'(inf)=1 (RK4 + bisection)."""

    def rhs(s):
        f, fp, fpp = s
        return np.array([fp, fpp, -0.5 * f * fpp])

    def rk4(s, dt):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * dt * k1)
        k3 = rhs(s + 0.5 * dt * k2)
        k4 = rhs(s + dt * k3)
        return s + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    n = int(round(eta_max / h))

    def shoot(fpp0):
        s = np.array([0.0, 0.0, fpp0])
        for _ in range(n):
            s = rk4(s, h)
        return s[1]

    lo, hi = 0.30, 0.37
    flo = shoot(lo) - 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        fm = shoot(mid) - 1.0
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    fpp0 = 0.5 * (lo + hi)

    etas = np.zeros(n + 1)
    fprimes = np.zeros(n + 1)
    s = np.array([0.0, 0.0, fpp0])
    for i in range(1, n + 1):
        s = rk4(s, h)
        etas[i] = i * h
        fprimes[i] = s[1]
    return etas, fprimes, fpp0


def plate_hwbb_mid(f: torch.Tensor, py: int, mask: torch.Tensor) -> torch.Tensor:
    """Fluid-side-reflection half-way bounce-back for a thin plate row y=py.

    Solid row y=py, fluid rows y<py and y>py. Wall positions py-0.5 (lower
    surface) and py+0.5 (upper surface). Periodic stream() fills solid-row
    self-values from the opposite boundary, so the standard post-stream
    bounce-back / B13 pre-streaming f_pre[OPPOSITE] variant are wrong here;
    the solid row is built from the adjacent fluid rows (post-collision)
    with the correct opposite mapping (opp(7)=5, opp(8)=6) including the
    x-shifts:

      upper surface (fluid above):  f2(py,x)=f4(py+1,x)
                                    f5(py,x)=f7(py+1,x+1)  f6(py,x)=f8(py+1,x-1)
      lower surface (fluid below):  f4(py,x)=f2(py-1,x)
                                    f7(py,x)=f5(py-1,x+1)  f8(py,x)=f6(py-1,x-1)
    """
    f = f.clone()
    # upper surface (fluid above, wall at py+0.5) — same form as run3's
    # bottom-wall plate: full velocity reversal (5<->7, 6<->8) with x-shifts.
    f[2, py, :] = torch.where(mask, f[4, py + 1, :], f[2, py, :])
    f[5, py, :-1] = torch.where(mask[:-1], f[7, py + 1, 1:], f[5, py, :-1])
    f[6, py, 1:] = torch.where(mask[1:], f[8, py + 1, :-1], f[6, py, 1:])
    # lower surface (fluid below, wall at py-0.5)
    f[4, py, :] = torch.where(mask, f[2, py - 1, :], f[4, py, :])
    f[7, py, 1:] = torch.where(mask[1:], f[5, py - 1, :-1], f[7, py, 1:])
    f[8, py, :-1] = torch.where(mask[:-1], f[6, py - 1, 1:], f[8, py, :-1])
    return f


def run_case(
    grid_name: str, U: float, nu: float, n_steps: int, out_path: str, device_str: str = "cpu"
) -> dict:
    g = GRIDS[grid_name]
    nx, ny = g["nx"], g["ny"]
    le, plate_len, probe = g["le"], g["plate_len"], g["probe"]
    plate_end = le + plate_len
    py = ny // 2
    tau = 3.0 * nu + 0.5
    device = torch.device(device_str)
    torch.manual_seed(0)

    plate = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    plate[py, le:plate_end] = True
    plate_row = plate[py]  # (nx,)

    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.full((ny, nx), U, device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, ux0, uy0, device=device)
    initial_mass = float(f.sum().item())

    spec = SPEC.to(device)
    # inlet free-stream (equilibrium Dirichlet, rho=1, ux=U) — precomputed
    feq_in = equilibrium(
        torch.ones((ny, 1), device=device),
        torch.full((ny, 1), U, device=device),
        torch.zeros((ny, 1), device=device),
    )[:, :, 0].contiguous()

    def step_fn(f_):
        f_ = collide_bgk(f_, tau)
        f_ = plate_hwbb_mid(f_, py, plate_row)
        # mirror ghost rows (slip/symmetry) before streaming: bottom + top
        f_ = f_.clone()
        f_[:, 0, :] = f_[:, 1, :][spec]
        f_[:, -1, :] = f_[:, -2, :][spec]
        f_ = stream(f_)
        # inlet: uniform free-stream Dirichlet
        f_ = f_.clone()
        f_[:, :, 0] = feq_in
        # outlet: Zou-He pressure (rho=1), library BC
        f_ = zou_he_outlet_pressure(f_, 1.0)
        return f_

    t0 = time.time()
    umax_hist: list[float] = []
    for step in range(1, n_steps + 1):
        f = step_fn(f)
        if step % 200 == 0:
            _, ux, _ = macroscopic(f)
            umax_hist.append(float(ux.max().item()))
    elapsed = time.time() - t0

    umax_arr = np.array(umax_hist)
    tail = umax_arr[-10:] if len(umax_arr) >= 10 else umax_arr
    umax_drift = (float(tail.max()) - float(tail.min())) / max(abs(float(tail.mean())), 1e-12)

    prof_up = torch.zeros(ny, device=device)
    prof_lo = torch.zeros(ny, device=device)
    for _ in range(200):
        f = step_fn(f)
        _, ux, _ = macroscopic(f)
        prof_up += ux[:, probe]
        prof_lo += ux[:, probe]  # same column; lower side read via mirror below
    prof_up /= 200.0

    rho, ux_f, uy_f = macroscopic(f)
    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0
    finite = bool(torch.isfinite(f).all().item())

    x_eff = probe - le
    Rex = U * x_eff / nu
    scale = math.sqrt(nu * x_eff / U)
    etas, fprimes, fpp0 = blasius_table()

    u_sim_up = prof_up.cpu().numpy()
    # upper surface: wall at y_w = py + 0.5, fluid rows py+1 .. ny-1
    y_w = py + 0.5
    y_hi = np.arange(py + 1, ny, dtype=np.float64) - y_w  # 0.5 .. ny-1.5-py
    eta_hi = y_hi / scale
    fp_ref_hi = np.interp(eta_hi, etas, fprimes)
    u_ref_hi = U * fp_ref_hi
    u_hi = u_sim_up[py + 1 :]

    m = (eta_hi > 0.05) & (eta_hi < 5.0) & (fp_ref_hi > 0.02) & (fp_ref_hi < 0.995)
    rel_err = np.abs(u_hi - u_ref_hi) / np.maximum(u_ref_hi, 1e-12)
    l2_rel = (
        float(np.linalg.norm(rel_err[m]) / np.linalg.norm(np.ones(m.sum())))
        if m.any()
        else float("nan")
    )
    max_rel_pct = float(rel_err[m].max() * 100.0) if m.any() else float("nan")

    # symmetry check: lower surface (rows 49..1, dist 0.5..48.5 from lower wall)
    # vs upper surface (rows 51..99, dist 0.5..48.5 from upper wall)
    u_lo = u_sim_up[:py]  # rows 0..py-1 (lower half of the domain, same column)
    sym_err_pct = float(
        np.max(np.abs(u_lo[: py - 1][::-1] - u_hi[: py - 1]) / np.maximum(U, 1e-12)) * 100.0
    )

    u_edge_probe = float(ux_f[ny - 2, probe].item())

    tab = []
    for eta_t in [1.0, 2.0, 3.0, 4.0, 5.0]:
        y_t = eta_t * scale + y_w
        u_interp = float(np.interp(y_t, np.arange(py + 1, ny, dtype=np.float64), u_hi))
        fp_ref = float(np.interp(eta_t, etas, fprimes))
        tab.append(
            {
                "eta": eta_t,
                "y_row": round(y_t, 3),
                "u_sim_over_U": round(u_interp / U, 5),
                "fprime_ref": round(fp_ref, 5),
                "rel_err_pct": round(abs(u_interp / U - fp_ref) / fp_ref * 100.0, 3),
            }
        )

    # wall shear: 2nd-order one-sided difference at s = 0.5/1.5/2.5 from wall
    s_dist = np.array([0.5, 1.5, 2.5])
    u123 = np.array([u_hi[0], u_hi[1], u_hi[2]])
    A = np.vstack([np.ones(3), s_dist, s_dist**2]).T
    w_der = np.linalg.solve(A, np.array([0.0, 1.0, 0.0]))
    dudy = float(w_der @ u123)
    tau_w = nu * dudy
    Cf_sim = 2.0 * tau_w / (U * U)
    Cf_ref = 0.664 / math.sqrt(Rex)
    Cf_err_pct = (Cf_sim - Cf_ref) / Cf_ref * 100.0

    delta_star_meas = float(np.trapezoid(1.0 - u_hi / max(u_edge_probe, 1e-12), y_hi))
    delta_star_ref = 1.7208 * x_eff / math.sqrt(Rex)

    profile_rows = []
    for i, eta_v in enumerate(eta_hi):
        profile_rows.append(
            {
                "y": int(py + 1 + i),
                "eta": round(float(eta_v), 4),
                "u_sim": round(float(u_hi[i]), 7),
                "u_blasius": round(float(u_ref_hi[i]), 7),
                "rel_err_pct": round(float(rel_err[i] * 100.0), 3),
            }
        )

    result = {
        "case": "B25_blasius_flat_plate",
        "grid": grid_name,
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": "equilibrium free-stream inlet + Zou-He pressure outlet (rho=1) + mirror(slip) top/bottom + mid-domain thin plate (fluid-side-reflection half-way BB, both surfaces)",
        "extrap": "none",
        "nx": nx,
        "ny": ny,
        "plate_y": py,
        "plate_len": plate_len,
        "U": U,
        "nu": nu,
        "tau": tau,
        "x0_leading_edge": le,
        "x_probe": probe,
        "x_eff": x_eff,
        "Rex": Rex,
        "Ma": U / math.sqrt(1.0 / 3.0),
        "n_steps": n_steps,
        "umax_drift_last_2000": umax_drift,
        "umax_history": [round(float(v), 6) for v in umax_arr[::5]],
        "blasius_fpp0_shooting": fpp0,
        "blasius_fpp0_ref": 0.332057,
        "l2_rel_err_profile": l2_rel,
        "max_rel_err_profile_pct": max_rel_pct,
        "eta_tab_comparison": tab,
        "u_edge_over_U_probe": round(u_edge_probe / U, 5),
        "symmetry_max_dev_over_U_pct": sym_err_pct,
        "delta_star_meas": round(delta_star_meas, 4),
        "delta_star_ref_1p7208_x_over_sqrtRex": round(delta_star_ref, 4),
        "Cf_sim": Cf_sim,
        "Cf_ref_0p664_over_sqrtRex": Cf_ref,
        "Cf_err_pct": Cf_err_pct,
        "tau_w_sim": tau_w,
        "dudy_wall_2nd_order": dudy,
        "mass_drift_pct": mass_drift_pct,
        "finite": finite,
        "elapsed_s": round(elapsed, 1),
        "profile": profile_rows,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="B25 Blasius flat-plate, mid-domain thin plate (D2Q9)")
    ap.add_argument("out_json", type=str)
    ap.add_argument("--grid", type=str, default="plate200", choices=list(GRIDS))
    ap.add_argument("--U", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    steps = args.smoke if args.smoke > 0 else args.steps
    r = run_case(args.grid, args.U, args.nu, steps, args.out_json, args.device)
    keys = [
        "grid",
        "nx",
        "ny",
        "plate_y",
        "U",
        "nu",
        "tau",
        "x_eff",
        "Rex",
        "n_steps",
        "umax_drift_last_2000",
        "l2_rel_err_profile",
        "max_rel_err_profile_pct",
        "eta_tab_comparison",
        "u_edge_over_U_probe",
        "symmetry_max_dev_over_U_pct",
        "delta_star_meas",
        "delta_star_ref_1p7208_x_over_sqrtRex",
        "Cf_sim",
        "Cf_ref_0p664_over_sqrtRex",
        "Cf_err_pct",
        "mass_drift_pct",
        "finite",
        "elapsed_s",
    ]
    print(json.dumps({k: r[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
