#!/usr/bin/env python3
"""Scratch diagnostic 2: beta sweep at fixed KC (is the quadrature force a
viscous effect?  Does the inviscid limit recover in-phase added mass?)."""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "/srv/tensor/agent/work/m20245/src")

from tensorlbm.oscillatory_cylinder_inertia import ArmCase, run_arm


def main(betas, kc=0.2, D=16, transient=4.0, measure=4.0, box=12.0):
    print(f"# KC={kc} D={D} box={box} transient={transient} measure={measure}")
    print(f"{'beta':>9} {'tau':>8} {'nu':>10} {'d/R':>7} {'C_I':>8} {'C_Q':>8} {'|C|':>8} "
          f"{'resid':>9} {'ref_area':>9} {'errI%':>8} {'errMod%':>8}")
    for b in betas:
        c = ArmCase(f"b{b:g}", D=D, kc=kc, beta=float(b), box=box,
                    transient=transient, measure=measure)
        r, _tr = run_arm(c, threads=4)
        k = r["kpi"]["E1_mem_area"]
        nu = r["numerics"]["nu"]
        T = r["numerics"]["T_steps"]
        dR = math.sqrt(nu * T / math.pi) / (r["geometry"]["D_area"] / 2)
        ref = r["reference"]["potential_same_mask_area"]
        mag = math.hypot(k["C_I"], k["C_Q"])
        print(f"{b:>9g} {r['numerics']['tau']:>8.5f} {nu:>10.3e} {dR:>7.4f} "
              f"{k['C_I']:>+8.4f} {k['C_Q']:>+8.4f} {mag:>8.4f} "
              f"{r['harmonic']['E1_mem']['resid_rel']:>9.2e} {ref:>9.4f} "
              f"{100*(k['C_I']-ref)/ref:>+8.2f} {100*(mag-ref)/ref:>+8.2f}")


if __name__ == "__main__":
    main([1e6, 1600.0, 400.0, 200.0, 100.0, 50.0])