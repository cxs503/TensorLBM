"""Run the **unmodified** OSS ``Maarten-vd-Sande/lbm`` snapshot as the XVAL oracle.

Numba is unavailable in this image, so a minimal stand-in module is injected
into ``sys.modules`` before importing the OSS package: ``jit`` becomes an
identity decorator.  The OSS source files are otherwise imported and called
exactly as shipped (``lbm.init`` / ``lbm.time_evolve`` / ``lbm.util`` /
``examples/dambreak.py`` initialisation), which is what makes the port
equivalence test meaningful: the oracle here is upstream code, not a copy.

The snapshot path is configurable (``--oss-root``); the default is the
inventoried ecosystem cache ``/root/oss-cache/Maarten-vd-Sande_lbm``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

OSS_ROOT = Path("/root/oss-cache/Maarten-vd-Sande_lbm")


def install_numba_shim() -> None:
    """``from numba import jit`` -> identity decorator."""
    if "numba" in sys.modules:
        return
    mod = types.ModuleType("numba")

    def jit(*args, **kwargs):  # noqa: D401  (mimics numba.jit)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def deco(fn):
            return fn

        return deco

    mod.jit = jit
    sys.modules["numba"] = mod


def import_oss(oss_root: Path = OSS_ROOT):
    """Import the OSS package with the numba shim in place."""
    install_numba_shim()
    root = str(Path(oss_root).resolve())
    # a foreign ``lbm`` package (e.g. repo ``src/lbm``) must not shadow the OSS one
    for name in [m for m in list(sys.modules) if m == "lbm" or m.startswith("lbm.")]:
        if not str(Path(getattr(sys.modules[name], "__file__", "")).resolve()).startswith(root):
            del sys.modules[name]
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    import lbm.util as U  # noqa: WPS433
    import lbm.init as INIT  # noqa: WPS433
    import lbm.streamcollide as SC  # noqa: WPS433
    from lbm.time_evolve import evolve  # noqa: WPS433

    return {"util": U, "init": INIT, "streamcollide": SC, "evolve": evolve}


def make_state(nx: int, ny: int, a: int, h0: int, oss):
    """``examples/dambreak.py`` initialisation, generalised block size.

    Upstream block construction is::

        minx, maxx = 0, 70 ; miny, maxy = 40, ny
        cell_type[xx, yy] = CT.INTERFACE
        cell_type[xx[1:-1,1:-1], yy[1:-1,1:-1]] = CT.FLUID
        mass[FLUID] = 1 ; mass[INTERFACE] = 0.5 ; rho += mass
        set_equi(...) ; fin = equi.copy()

    i.e. the block starts at column 0 / row 40 and its outer ring is
    INTERFACE.  Here the block spans columns ``0..a-1`` and rows
    ``ny-h0..ny-1`` (bottom-left origin), the walls are re-stamped on all four
    sides exactly like the example, and ``rho += mass`` is kept.
    """
    CT, v, t = oss["util"].CT, oss["util"].v, oss["util"].t
    fin, fout, equi, fdist, inlet, u, rho, mass, cell_type = oss["init"].init(nx, ny)
    minx, maxx = 0, a
    miny, maxy = ny - h0, ny
    xx, yy = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy))
    cell_type[xx, yy] = CT.INTERFACE
    cell_type[xx[1:-1, 1:-1], yy[1:-1, 1:-1]] = CT.FLUID
    cell_type[:, (0, -1)] = CT.OBSTACLE
    cell_type[(0, -1), :] = CT.OBSTACLE
    mass[cell_type == CT.FLUID] = 1
    mass[cell_type == CT.INTERFACE] = 0.5
    rho += mass
    sc = oss["streamcollide"]
    for ix in range(rho.shape[0]):
        for iy in range(rho.shape[1]):
            sc.set_equi(ix, iy, equi, rho[ix, iy], u[:, ix, iy], v, t)
    fin[:] = equi
    return fin, fout, equi, fdist, inlet, u, rho, mass, cell_type


def run_oss(nx: int = 40, ny: int = 24, a: int = 8, h0: int = 16,
            steps: int = 5, omega: float = 1.0, gravity=(0.0, 0.0),
            oss=None, trace_every: int = 1, quiet: bool = True):
    """Run upstream ``evolve`` and return the inventory trace + final fields."""
    oss = oss or import_oss()
    CT = oss["util"].CT
    fin, fout, equi, fdist, inlet, u, rho, mass, ct = make_state(nx, ny, a, h0, oss)

    def inventory(rho, mass, ct):
        # cell_type is a bitmask (NO_*/TO_* bits), so compare bitwise
        return float(rho[(ct & CT.FLUID) != 0].sum() + mass[(ct & CT.INTERFACE) != 0].sum())

    trace = [{"step": 0, "inventory": inventory(rho, mass, ct),
              "front_cells": int(np.flatnonzero(((ct & (CT.FLUID | CT.INTERFACE)) != 0)
                                                .any(axis=1))[-1])}]
    orig_stdout = sys.stdout
    if quiet:
        sys.stdout = open("/dev/null", "w")
    try:
        for k in range(1, steps + 1):
            # Q6: ``evolve`` returns ``velocities`` (a (T,2,nx,ny) stack)
            # under the name ``u`` -- the example's ``u, cell_types = evolve()``
            # therefore hands back a 4-D array.  Take the last frame so the
            # state can be advanced in single steps for tracing.
            u_stack, cell_types = oss["evolve"](1, fin, fout, equi, fdist, inlet, u, rho,
                                                mass, ct, np.array([omega]),
                                                oss["util"].v, oss["util"].v_inv,
                                                oss["util"].t, np.array(gravity))
            u = u_stack[-1]
            if k % trace_every == 0 or k == steps:
                trace.append({
                    "step": k,
                    "inventory": inventory(rho, mass, ct),
                    "front_cells": int(np.flatnonzero(((ct & (CT.FLUID | CT.INTERFACE)) != 0)
                                                      .any(axis=1))[-1]),
                    "u_max": float(np.abs(u).max()),
                })
    finally:
        if quiet:
            sys.stdout.close()
            sys.stdout = orig_stdout
    return {"trace": trace,
            "fields": {"fin": fin.copy(), "mass": mass.copy(), "rho": rho.copy(),
                       "u": u.copy(), "cell_type": ct.copy()}}


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=40)
    ap.add_argument("--ny", type=int, default=24)
    ap.add_argument("--a", type=int, default=8)
    ap.add_argument("--h0", type=int, default=16)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--gravity", type=float, default=0.0)
    ap.add_argument("--oss-root", default=str(OSS_ROOT))
    args = ap.parse_args()
    out = run_oss(args.nx, args.ny, args.a, args.h0, args.steps,
                  gravity=(0.0, -args.gravity), oss=import_oss(Path(args.oss_root)))
    print(json.dumps(out["trace"], indent=1))