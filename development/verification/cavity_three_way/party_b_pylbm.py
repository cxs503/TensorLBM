"""Party B -- independent open-source LBM (pylbm) at the same specification.

Survey note: pylbm ``demo/2D/lid_driven_cavity.py`` builds the cavity with the
d'Humieres moment set and *independent* relaxation rates for the energy and
deviatoric-moment families (i.e. MRT), Bouzidi-Firdaouss-Lallemand bounce-back
on the walls and a prescribed-moment (conserved variables) lid condition.  This
driver keeps that recipe and only parameterises the geometry and the
viscosity so that it matches the shared spec:

    N cells of size dx = 1/N      -> cavity side N lattice spacings (h = 1)
    U = spec.u_lid                 (lattice units)
    nu_lattice = U*N/Re            -> sigma_2 = 3*nu = tau - 1/2
    energy modes                    sigma_1 = 100 * sigma_2 (demo's zeta = 100 mu)

Note the pylbm viscosity mapping: with ``dummy = 3/(la*rho_o*space_step)`` and
``s_2 = 1/(0.5 + dummy*mu)``, the shear rate is ``dummy*mu = 3*nu_lattice`` when
``mu = nu_lattice*dx``.  That identity is asserted in ``_self_check`` below so a
future pylbm version that changes the convention cannot silently change Re.

Serial execution needs ``mpi4py``; the shim in ``mpi4py_shim/`` supplies the
single-rank subset pylbm imports (see ``README.md``).  Only rank 0 exists, so
this is *not* an MPI-scale run -- it is the same solver, one process.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp

from cavity_spec import CavitySpec, centreline_metrics, load_ghia, write_case_output

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "mpi4py_shim"))

import pylbm  # noqa: E402  (needs the shim on sys.path first)

X, Y = sp.symbols("X, Y")
RHO, QX, QY = sp.symbols("rho, qx, qy")
LA = sp.symbols("lambda", constants=True)


def _scheme(N: int, re: float, u_lid: float):
    """d'Humieres MRT moment set with the viscosity pinned to the shared spec."""
    dx = 1.0 / N
    nu = u_lid * N / re
    sigma_2 = 3.0 * nu                      # = tau - 1/2
    dummy = 3.0 / (1.0 * 1.0 * dx)          # la = rho_o = 1
    mu = sigma_2 / dummy                    # == nu * dx
    zeta = 100.0 * mu
    s_1 = 1.0 / (0.5 + dummy * zeta)        # energy / energy-flux family
    s_2 = 1.0 / (0.5 + dummy * mu)          # deviatoric (shear) family

    _self_check(dummy * mu, 3.0 * nu)

    QX2 = QX**2
    QY2 = QY**2
    Q2 = QX2 + QY2
    QXY = QX * QY
    polynomials = [
        1,
        X,
        Y,
        3 * (X**2 + Y**2) - 4 * LA**2,
        0.5 * (9 * (X**2 + Y**2) ** 2 - 21 * (X**2 + Y**2) * LA**2 + 8 * LA**4),
        3 * X * (X**2 + Y**2) - 5 * X * LA**2,
        3 * Y * (X**2 + Y**2) - 5 * Y * LA**2,
        X**2 - Y**2,
        X * Y,
    ]
    equilibrium = [
        RHO,
        QX,
        QY,
        -2 * RHO * LA**2 + 3 * Q2,
        RHO * LA**2 - 3 * Q2,
        -QX * LA**2,
        -QY * LA**2,
        QX2 - QY2,
        QXY,
    ]
    s = [0.0, 0.0, 0.0, s_1, s_1, s_1, s_1, s_2, s_2]
    return polynomials, equilibrium, s, 1.0 / (0.5 + sigma_2)


def _self_check(got: float, want: float) -> None:
    if abs(got - want) > 1e-9 * max(1.0, abs(want)):
        raise RuntimeError(
            f"pylbm viscosity mapping changed: 3*nu = {want}, relaxation gives {got}"
        )


def bc_up(f, m, x, y, rho_o, driven_velocity):
    """lid: prescribed conserved moments (pylbm demo recipe)"""
    m[RHO] = rho_o
    m[QX] = rho_o * driven_velocity
    m[QY] = 0.0


def bevelled_skip():
    return None


def run_case(spec: CavitySpec, N: int, generator: str = "numpy") -> dict:
    polynomials, equilibrium, s, tau = _scheme(N, spec.re, spec.u_lid)
    dx = 1.0 / N
    rho_o = 1.0

    cfg = {
        "parameters": {LA: 1.0},
        "box": {"x": [0.0, 1.0], "y": [0.0, 1.0], "label": [0, 0, 0, 1]},
        "space_step": dx,
        "scheme_velocity": LA,
        "schemes": [
            {
                "velocities": list(range(9)),
                "polynomials": polynomials,
                "relaxation_parameters": s,
                "equilibrium": equilibrium,
                "conserved_moments": [RHO, QX, QY],
            }
        ],
        "init": {RHO: rho_o, QX: 0.0, QY: 0.0},
        "boundary_conditions": {
            0: {"method": {0: pylbm.bc.BFLBounceBack}},
            1: {
                "method": {0: pylbm.bc.BFLBounceBack},
                "value": (bc_up, (rho_o, spec.u_lid)),
            },
        },
        "generator": generator,
        "relative_velocity": [QX / RHO, QY / RHO],
    }
    sim = pylbm.Simulation(cfg)

    # steady-state monitor: pylbm stores physical time; dt = dx/la
    dt = dx
    ux_prev = None
    resid = float("inf")
    t0 = time.time()
    check_every = 2000
    steps = 0
    hist = []
    while steps < spec.max_steps:
        for _ in range(check_every):
            sim.one_time_step()
        steps += check_every
        ux = np.array(sim.m[QX] / sim.m[RHO])
        uy = np.array(sim.m[QY] / sim.m[RHO])
        if ux_prev is None:
            ux_prev = (ux.copy(), uy.copy())
            continue
        resid = max(np.abs(ux - ux_prev[0]).max(), np.abs(uy - ux_prev[1]).max()) / spec.u_lid
        hist.append([steps, float(resid)])
        ux_prev = (ux.copy(), uy.copy())
        if resid < spec.resid_tol:
            break
    elapsed = time.time() - t0

    rho = np.array(sim.m[RHO])
    ux = np.array(sim.m[QX] / sim.m[RHO]) / spec.u_lid
    uy = np.array(sim.m[QY] / sim.m[RHO]) / spec.u_lid
    if (rho < 0.5).any() or not np.isfinite(ux).all():
        raise RuntimeError(f"pylbm diverged at N={N} Re={spec.re}")

    # pylbm cell i sits at (i+0.5)/N inside [0,1]; walls are the cell faces.
    n_cells = ux.shape[0]
    y = (np.arange(n_cells) + 0.5) / n_cells
    x = (np.arange(n_cells) + 0.5) / n_cells
    # pylbm stores fields as [x, y]: u varies along y at fixed x, v along x at
    # fixed y (getting this transposed silently compares the wrong profiles).
    col = n_cells // 2
    u_cl = ux[col, :]
    v_cl = uy[:, col]

    # no-slip / lid values are imposed at the faces -> add them explicitly
    yy = np.concatenate(([0.0], y, [1.0]))
    uu = np.concatenate(([0.0], u_cl, [1.0]))
    xx = np.concatenate(([0.0], x, [1.0]))
    vv = np.concatenate(([0.0], v_cl, [0.0]))

    speed2 = ux**2 + uy**2
    inner = np.zeros_like(speed2, dtype=bool)
    lo, hi = int(0.30 * n_cells), int(0.90 * n_cells)
    inner[lo:hi, lo:hi] = True          # primary vortex window (as in the repo case)
    s2 = np.where(inner, speed2, np.inf)
    ix, iy = np.unravel_index(np.argmin(s2), s2.shape)   # first axis is x

    def refine2(a, b, c):
        den = a - 2 * b + c
        return 0.5 * (a - c) / den if abs(den) > 1e-30 else 0.0

    try:
        cx = (ix + refine2(s2[ix - 1, iy], s2[ix, iy], s2[ix + 1, iy])) / n_cells
        cy = (iy + refine2(s2[ix, iy - 1], s2[ix, iy], s2[ix, iy + 1])) / n_cells
    except (IndexError, ValueError):
        cx = (ix + 0.5) / n_cells
        cy = (iy + 0.5) / n_cells

    ghia = load_ghia(int(spec.re))
    met = centreline_metrics(yy, uu, xx, vv, ghia, spec.u_lid, label=f"pylbm_N{N}")
    met["vortex_center"] = [float(cx), float(cy)]
    met["geometry"] = {"W_lattice": float(N), "H_lattice": float(N), "aspect": 1.0}
    met["wall_convention"] = (
        "BFL bounce-back (distance 0.5) on cell faces; lid = prescribed moments"
    )
    met["collision"] = "d'Humieres MRT (energy sigma_1 = 100*sigma_2)"

    return {
        "party": "B",
        "tool": "pylbm",
        "tool_version": getattr(pylbm, "__version__", "unknown"),
        "spec": spec.as_dict(),
        "n_cells": int(n_cells),
        "tau_shear_rate": 1.0 / max(tau, 1e-30),
        "generator": generator,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "final_resid": float(resid),
        "resid_history": hist[-6:],
        "metrics": met,
        "profile": {"y": yy.tolist(), "u": uu.tolist(),
                    "x": xx.tolist(), "v": vv.tolist()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--re", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=128, help="cavity side in cells")
    ap.add_argument("--u-lid", type=float, default=0.06)
    ap.add_argument("--max-steps", type=int, default=400_000)
    ap.add_argument("--generator", default="numpy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    spec = CavitySpec(re=a.re, n=a.n, u_lid=a.u_lid, max_steps=a.max_steps)
    res = run_case(spec, a.n, generator=a.generator)
    write_case_output(Path(a.out), res)
    m = res["metrics"]
    print(
        f"[B N={a.n} Re={a.re:g}] steps={res['steps']} resid={res['final_resid']:.1e} "
        f"u_min={m['u_min']:+.4f} (Ghia {m['ghia_u_min']:+.4f}) "
        f"v_max={m['v_max']:+.4f} (Ghia {m['ghia_v_max']:+.4f}) "
        f"vortex=({m['vortex_center'][0]:.3f},{m['vortex_center'][1]:.3f}) "
        f"rmse_u={m['rmse_u']:.4f} max_dev={m['max_abs_dev_pct']:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()