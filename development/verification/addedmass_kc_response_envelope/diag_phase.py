#!/usr/bin/env python3
"""Scratch diagnostic: why does the in-phase/quadrature split of the
oscillatory-cylinder force drift at low KC?  Prints the full harmonic
decomposition of F (both estimators) and of the box-mean velocity, plus a
window-resolved harmonic analysis (does the phase settle?)."""
from __future__ import annotations

import math
import sys

sys.path.insert(0, "/srv/tensor/agent/work/m20245/src")
import numpy as np

from tensorlbm.oscillatory_cylinder_inertia import ArmCase, harmonic_decompose, run_arm


def windowed(trace, t0, nwin, period):
    out = []
    for k in range(nwin):
        a = t0 + k * period
        b = a + period
        d = harmonic_decompose(trace["t"], trace["F_mem"], trace["omega"], t0=a, t1=b, nh=2)
        du = harmonic_decompose(trace["t"], trace["U"], trace["omega"], t0=a, t1=b, nh=2)
        out.append((a, d["cos"], d["sin"], d["resid_rel"], du["cos"], du["sin"], du["amp"]))
    return out


def main():
    arms = [
        ("kc02_b100_T8", ArmCase("d1", D=16, kc=0.2, beta=100.0, transient=8.0, measure=4.0)),
        ("kc02_b400_T8", ArmCase("d2", D=16, kc=0.2, beta=400.0, transient=8.0, measure=4.0)),
        ("kc10_b100_T8", ArmCase("d3", D=16, kc=1.0, beta=100.0, transient=8.0, measure=4.0)),
    ]
    for tag, c in arms:
        r, tr = run_arm(c, threads=4)
        T = r["numerics"]["T_steps"]
        e1 = r["kpi"]["E1_mem_area"]
        mag = math.hypot(e1["C_I"], e1["C_Q"])
        print(f"\n=== {tag}: D={c.D} KC={c.kc} beta={c.beta} T={T:.1f} n_steps={r['numerics']['n_steps']}")
        print(f"  achieved KC={r['achieved']['KC_halfway']:.3f} beta={r['achieved']['beta_halfway']:.1f} "
              f"a0meas/a0={r['achieved']['a0_over_nominal']:.5f} U_resid={r['achieved']['U_resid_rel']:.3e}")
        for k, v in r["kpi"].items():
            print(f"  {k:16s} C_I={v['C_I']:+.4f} C_Q={v['C_Q']:+.4f} |C|={math.hypot(v['C_I'], v['C_Q']):.4f}")
        print(f"  ref same-mask area={r['reference']['potential_same_mask_area']:.4f} "
              f"halfway={r['reference']['potential_same_mask_halfway']:.4f}; E1|C|={mag:.4f}")
        print(f"  harmonic resid_rel E1={r['harmonic']['E1_mem']['resid_rel']:.3e} "
              f"U={r['harmonic']['U']['resid_rel']:.3e}; U cos/sin={r['harmonic']['U']['cos']:+.3e}/{r['harmonic']['U']['sin']:+.3e}")
        win = windowed(tr, int(round(c.transient * T)), 4, T)
        print("  windowed F_mem (cos, sin, |C| via area D, resid, U cos, U sin):")
        A = r["geometry"]["area_solid"]
        for (a, c_, s_, rr, uc, us, ua) in win:
            k = 1.0 / (r["numerics"]["a0"] * A)
            print(f"   t>={a:5d}: C_I={c_*k:+.3f} C_Q={s_*k:+.3f} |C|={math.hypot(c_, s_)*k:.3f} "
                  f"resid={rr:.2e}  U cos={uc:+.3e} sin={us:+.3e}")


if __name__ == "__main__":
    main()