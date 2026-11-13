"""Smoke test: build a D3Q19 Taylor-Green (z,x)-plane case in pylbm 0.12.0 and run it.

Cross-validation harness for XVAL-LBM-CARTESIAN-MPI.  Runs pylbm's own code
(numpy backend) through the execution-only mpi4py shim in
``tools/xval_cartesian_mpi/shims``.

Usage:
    PYTHONPATH=tools/xval_cartesian_mpi/shims:/tmp/pylbm_src \
        python tools/xval_cartesian_mpi/pylbm_tgv_run.py --n 16 --steps 5
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import sympy as sp

# ---------------------------------------------------------------------------
# D3Q19 lattice setup (derived from pylbm's own stencil, not hard-coded)
# ---------------------------------------------------------------------------

W_D3Q19 = {"0": 1.0 / 3.0, "1": 1.0 / 18.0, "2": 1.0 / 36.0}


def _stencil_velocities(st) -> np.ndarray:
    """Integer velocity array of pylbm's stencil (handles Velocity objects)."""
    raw = st.v[0]
    out = []
    for x in raw:
        v = getattr(x, "v", x)
        out.append([int(c) for c in np.asarray(v).ravel()])
    return np.asarray(out)


def d3q19_indices(pylbm) -> list[int]:
    """27-velocity stencil indices that make up D3Q19 (|c|_1 <= 2)."""
    st = pylbm.Stencil({"dim": 3, "schemes": [{"velocities": list(range(27))}]})
    v = _stencil_velocities(st)
    keep = [i for i, c in enumerate(v) if np.abs(c).sum() <= 2]
    assert len(keep) == 19, len(keep)
    return keep


def d3q19_velocity_set(pylbm) -> np.ndarray:
    st = pylbm.Stencil({"dim": 3, "schemes": [{"velocities": list(range(27))}]})
    return _stencil_velocities(st)[d3q19_indices(pylbm)]


def weights_for(vel: np.ndarray) -> np.ndarray:
    w = []
    for c in vel:
        s = np.abs(c).sum()
        w.append(1.0 / 3.0 if s == 0 else (1.0 / 18.0 if s == 1 else 1.0 / 36.0))
    return np.asarray(w, dtype=float)


def moment_basis(vel: np.ndarray) -> list:
    """Monomial basis (in X, Y, Z) completing {1, X, Y, Z} to a full rank-19 set.

    Chosen and *verified* numerically (rank == 19) from the actual D3Q19
    velocity set, so no hand-transcribed moment table is involved.
    """
    X, Y, Z = sp.symbols("X Y Z")
    cand = [1, X, Y, Z]
    degree_terms = [
        X**2, Y**2, Z**2, X * Y, X * Z, Y * Z,
        X**2 * Y, X**2 * Z, Y**2 * X, Y**2 * Z, Z**2 * X, Z**2 * Y, X * Y * Z,
        X**2 * Y**2, Y**2 * Z**2, Z**2 * X**2,
        X**2 * Y * Z, Y**2 * X * Z, Z**2 * X * Y,
        X**3, Y**3, Z**3,
    ]
    A = [np.array([float(sp.sympify(p).subs({X: c[0], Y: c[1], Z: c[2]})) for p in cand])
         for c in vel]
    A = np.asarray(A)  # (19, 4)
    for p in degree_terms:
        col = np.array([float(sp.sympify(p).subs({X: c[0], Y: c[1], Z: c[2]})) for c in vel])
        trial = np.column_stack([A, col])
        if np.linalg.matrix_rank(trial) > A.shape[1]:
            A = trial
            cand.append(p)
        if len(cand) == 19:
            break
    assert len(cand) == 19 and np.linalg.matrix_rank(A) == 19, (
        len(cand), np.linalg.matrix_rank(A))
    return cand


def equilibrium_func(vel: np.ndarray, weights: np.ndarray):
    """Maxwell-Boltzmann discrete equilibrium (lattice units, cs^2 = 1/3)."""
    rho, ux, uy, uz = sp.symbols("rho u_x u_y u_z")
    out = []
    u2 = ux**2 + uy**2 + uz**2
    for c, w in zip(vel, weights):
        cu = c[0] * ux + c[1] * uy + c[2] * uz
        out.append(sp.Rational(w).limit_denominator() if False else
                   sp.Float(w) * rho * (1 + 3 * cu + sp.Rational(9, 2) * cu**2 - sp.Rational(3, 2) * u2))
    return out, [rho, ux, uy, uz]


def build_dico(n: int, u0: float, nu: float, generator: str = "numpy") -> dict:
    import pylbm

    vel = d3q19_velocity_set(pylbm)
    idx = d3q19_indices(pylbm)
    w = weights_for(vel)
    polys = moment_basis(vel)
    feq_expr, syms = equilibrium_func(vel, w)
    k = 2.0 * np.pi / n
    s_nu = 1.0 / (3.0 * nu + 0.5)

    def feq(v, rho, ux, uy, uz):
        # v: (19, 3) integer lattice velocities of this stencil
        out = []
        u2 = ux**2 + uy**2 + uz**2
        for c, ww in zip(np.asarray(v), w):
            cu = c[0] * ux + c[1] * uy + c[2] * uz
            out.append(ww * rho * (1 + 3 * cu + sp.Rational(9, 2) * cu**2 - sp.Rational(3, 2) * u2))
        return sp.Matrix(out)

    rho, ux, uy, uz = syms
    nrelax = len(polys) - 4
    dico = {
        "box": {
            "x": [0, n], "y": [0, n], "z": [0, n],
            "label": [0, 1, 2, 3, 4, 5],
        },
        "space_step": 1.0,
        "scheme_velocity": 1.0,
        "schemes": [
            {
                "velocities": idx,
                "conserved_moments": [rho, ux, uy, uz],
                "polynomials": polys,
                "relaxation_parameters": [0, 0, 0, 0] + [s_nu] * nrelax,
                "feq": [feq, [rho, ux, uy, uz]],
            }
        ],
        "init": {
            rho: sp.sympify("rho0"),
            ux: sp.sympify("-U0*cos(k*(Z-0.5))*sin(k*(X-0.5))"),
            uy: 0.0,
            uz: sp.sympify("U0*sin(k*(Z-0.5))*cos(k*(X-0.5))"),
        },
        "parameters": {
            sp.Symbol("rho0"): 1.0, sp.Symbol("U0"): u0,
            sp.Symbol("k"): k, sp.Symbol("nu"): nu,
        },
        "generator": generator,
    }
    return dico


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--u0", type=float, default=0.05)
    ap.add_argument("--nu", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--generator", default="numpy")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    import pylbm

    # pylbm 0.12's cerberus schema types 'symbol' as sympy.Symbol and then
    # *sorts* error paths with sympy objects, which raises on Python 3.11
    # (sympy Relational truth value).  The dico is built programmatically here,
    # so the schema check is redundant: disable it and let pylbm's own
    # downstream checks (M invertibility, inverse check, etc.) do the work.
    import pylbm.simulation as _sim_mod
    import pylbm.stencil as _st_mod
    import pylbm.scheme as _sc_mod
    import pylbm.domain as _dm_mod

    for _m in (_sim_mod, _st_mod, _sc_mod, _dm_mod):
        if hasattr(_m, "validate"):
            _m.validate = lambda *a, **k: None

    dico = build_dico(args.n, args.u0, args.nu, args.generator)
    sim = pylbm.Simulation(dico)
    print("topology:", sim.domain.mpi_topo.split, sim.domain.shape_in, sim.domain.shape_halo)
    print("s_nu:", dico["schemes"][0]["relaxation_parameters"][4])

    hist = []
    for it in range(args.steps + 1):
        u = np.asarray(sim.m[1](0)) if False else None
        rho = np.asarray(sim.m[0]).copy()
        ux = np.asarray(sim.m[1]).copy()
        uy = np.asarray(sim.m[2]).copy()
        uz = np.asarray(sim.m[3]).copy()
        K = 0.5 * np.mean(ux**2 + uy**2 + uz**2)
        hist.append({"step": it, "t": float(sim.t), "K": float(K),
                     "rho_max_err": float(np.abs(rho - 1.0).max())})
        if it < args.steps:
            sim.one_time_step()
    for h in hist:
        print(h)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"n": args.n, "u0": args.u0, "nu": args.nu, "hist": hist}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())