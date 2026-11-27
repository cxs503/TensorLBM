#!/usr/bin/env python3
"""Scratch diagnostic 3: is the spurious quadrature force a *compressibility*
(acoustic) artefact?  At fixed KC and beta, lower u0 -> lower omega -> smaller
omega*L/c_s.  Also varies the box ratio at fixed u0."""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "/srv/tensor/agent/work/m20245/src")

from tensorlbm.oscillatory_cylinder_inertia import ArmCase, run_arm

CS = 1.0 / math.sqrt(3.0)


def row(tag, c):
    r, _tr = run_arm(c, threads=4)
    k = r["kpi"]["E1_mem_area"]
    n = r["numerics"]
    ref = r["reference"]["potential_same_mask_area"]
    mag = math.hypot(k["C_I"], k["C_Q"])
    phi = math.degrees(math.atan2(k["C_Q"], k["C_I"]))
    ac = n["omega"] * n["nx"] / CS
    print(f"{tag:>14s} u0={c.u0:<6g} box={c.box:<5g} T={n['T_steps']:>6.1f} "
          f"wL/cs={ac:>7.2f} Ma={n['Ma']:.4f} C_I={k['C_I']:>+7.3f} C_Q={k['C_Q']:>+7.3f} "
          f"|C|={mag:>6.3f} phi={phi:>+7.1f} ref={ref:.3f} errI={100*(k['C_I']-ref)/ref:>+7.1f}% "
          f"errMod={100*(mag-ref)/ref:>+7.1f}% resid={r['harmonic']['E1_mem']['resid_rel']:.2f}",
          flush=True)
    return r


if __name__ == "__main__":
    print("# KC=0.2, D=16, beta=100, transient=4, measure=4")
    for u0 in (0.04, 0.02, 0.01, 0.005):
        row(f"u0={u0:g}", ArmCase(f"u{u0:g}", D=16, kc=0.2, beta=100.0, u0=u0,
                                box=12.0, transient=4.0, measure=4.0))
    for box in (6.0, 4.0):
        row(f"box={box:g}", ArmCase(f"b{box:g}", D=16, kc=0.2, beta=100.0, u0=0.04,
                                   box=box, transient=4.0, measure=4.0))