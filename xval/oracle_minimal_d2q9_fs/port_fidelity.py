"""Port-equivalence test: ``oss_oracle_port`` vs the **unmodified** upstream code.

The XVAL oracle must be upstream, not a re-implementation.  This script drives
the shipped ``lbm`` package (through a numba stand-in, see
:mod:`oss_original`) and the port from identical initial states and reports,
step by step, the maximum absolute difference on ``fin``, ``mass``, ``rho``,
``u`` and the cell-type field, plus the liquid inventory both sides.

Outcome recorded for the anchor (2026-11-13): agreement to float32 round-off
(``< 1e-6`` on every field, identical cell types) for as long as upstream
remains finite.  Upstream itself raises ``RuntimeWarning: invalid value
encountered in divide`` in ``update_rho_u`` once an interface cell's
``sum(fin)`` hits zero, after which it is all-NaN; the port guards that
division (documented deviation Q7) and therefore keeps running -- so the
comparison is truncated at the first non-finite upstream field.

Usage
    python xval/oracle_minimal_d2q9_fs/port_fidelity.py --case block_g1e-3 --steps 10
    python xval/oracle_minimal_d2q9_fs/port_fidelity.py --case dam_break_a16 --steps 400
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oss_oracle_port as P  # noqa: E402
import oss_original as O  # noqa: E402

CASES = {
    # 40x24 / a=8 / h0=16 -- the case used while bisecting the port
    "block_g1e-3": dict(nx=40, ny=24, a=8, h0=16, g=1e-3),
    "block_static": dict(nx=40, ny=24, a=8, h0=16, g=0.0),
    # reference resolution, mask/ablation case
    "dam_break_a16": dict(nx=80, ny=48, a=16, h0=32, g=5e-5),
}


def compare(case: str, steps: int, trace_every: int = 1):
    cfg = CASES[case]
    oss = O.import_oss()
    CT = oss["util"].CT
    nx, ny, a, h0, g = cfg["nx"], cfg["ny"], cfg["a"], cfg["h0"], cfg["g"]

    fin, fout, equi, fdist, inlet, u, rho, mass, ct = O.make_state(nx, ny, a, h0, oss)
    st = P.make_oss_block(nx=nx, ny=ny, a=a, h0=h0)

    def uinv(rho, mass, ct):
        return float(rho[(ct & CT.FLUID) != 0].sum() + mass[(ct & CT.INTERFACE) != 0].sum())

    rows = []
    t0 = time.time()
    first_nan = None
    for step in range(1, steps + 1):
        with contextlib.redirect_stdout(io.StringIO()):
            vel, _ = oss["evolve"](1, fin, fout, equi, fdist, inlet, u, rho, mass, ct,
                                   np.array([1.0]), oss["util"].v, oss["util"].v_inv,
                                   oss["util"].t, np.array([0.0, -g]))
        u = vel[-1]
        P.step(st, omega=1.0, gravity=(0.0, -g))
        finite = bool(np.isfinite(rho).all() and np.isfinite(mass).all()
                      and np.isfinite(fin).all())
        if not finite and first_nan is None:
            first_nan = step
            break
        if step % trace_every == 0 or step == 1:
            d = {
                "step": step,
                "ct_equal": bool((ct == st.flags()).all()),
                "fin_maxdiff": float(np.abs(fin - P.I(st.fin)).max()),
                "mass_maxdiff": float(np.abs(mass - P.I(st.mass)).max()),
                "rho_maxdiff": float(np.abs(rho - P.I(st.rho)).max()),
                "u_maxdiff": float(np.abs(u - P.I(st.u)).max()),
                "inventory_upstream": uinv(rho, mass, ct),
                "inventory_port": st.liquid_inventory(),
                "front_upstream": int(np.flatnonzero(
                    ((ct & (CT.FLUID | CT.INTERFACE)) != 0).any(axis=1))[-1]),
                "front_port": st.front_cells(),
            }
            rows.append(d)
            print(json.dumps(d), flush=True)
    summary = {
        "case": case, "cfg": cfg, "steps_requested": steps,
        "steps_compared": rows[-1]["step"] if rows else 0,
        "upstream_first_nonfinite_step": first_nan,
        "max_ct_mismatch_steps": sum(0 if r["ct_equal"] else 1 for r in rows),
        "max_fin_abs_diff": max((r["fin_maxdiff"] for r in rows), default=None),
        "max_mass_abs_diff": max((r["mass_maxdiff"] for r in rows), default=None),
        "max_u_abs_diff": max((r["u_maxdiff"] for r in rows), default=None),
        "front_mismatch_steps": sum(0 if r["front_upstream"] == r["front_port"] else 1
                                    for r in rows),
        "wall_seconds": round(time.time() - t0, 2),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    return {"summary": summary, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="block_g1e-3", choices=sorted(CASES))
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--trace-every", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--all-cases", action="store_true")
    args = ap.parse_args()
    out = {}
    if args.all_cases:
        for case, cfg in CASES.items():
            steps = 40 if case == "block_static" else (10 if "g1e-3" in case
                                                      else args.steps)
            out[case] = compare(case, steps, args.trace_every)
    else:
        out[args.case] = compare(args.case, args.steps, args.trace_every)
    path = Path(args.out) if args.out else HERE / "artifacts" / "port_fidelity.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())