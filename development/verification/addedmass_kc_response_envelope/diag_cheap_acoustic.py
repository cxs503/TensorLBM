#!/usr/bin/env python3
"""Scratch diagnostic 4: shrink omega*L/c_s by lowering u0 at fixed KC (cheap
box=6 geometry).  Prediction (H1): the quadrature contamination collapses."""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "/srv/tensor/agent/work/m20245/src")

from tensorlbm.oscillatory_cylinder_inertia import ArmCase, run_arm

CS = 1.0 / math.sqrt(3.0)


def row(c):
    r, _ = run_arm(c, threads=4)
    k = r["kpi"]["E1_mem_area"]
    n = r["numerics"]
    ref = r["reference"]["potential_same_mask_area"]
    mag = math.hypot(k["C_I"], k["C_Q"])
    print(f"u0={c.u0:<6g} T={n['T_steps']:>6.1f} wL/cs={n['omega']*n['nx']/CS:>6.2f} "
          f"tau={n['tau']:.4f} nu={n['nu']:.2e} C_I={k['C_I']:>+7.3f} C_Q={k['C_Q']:>+7.3f} "
          f"|C|={mag:>6.3f} phi={math.degrees(math.atan2(k['C_Q'],k['C_I'])):>+7.1f} "
          f"ref={ref:.4f} errI={100*(k['C_I']-ref)/ref:>+7.1f}% "
          f"errMod={100*(mag-ref)/ref:>+7.1f}% resid={r['harmonic']['E1_mem']['resid_rel']:.3f}",
          flush=True)


if __name__ == "__main__":
    print("# D=16 box=6 KC=0.2 beta=200 transient=3 measure=3")
    for u0 in (0.02, 0.01):
        row(ArmCase(f"a{u0:g}", D=16, kc=0.2, beta=200.0, u0=u0, box=6.0, transient=3.0, measure=3.0))