#!/usr/bin/env python3
"""Scratch diagnostic 5: per-period harmonic windowing of the force trace, with
DC and 2*omega content, to see whether a periodic state is ever reached."""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/srv/tensor/agent/work/m20245/src")

from tensorlbm.oscillatory_cylinder_inertia import ArmCase, harmonic_decompose, run_arm

CS = 1.0 / math.sqrt(3.0)


def main():
    c = ArmCase("d5", D=12, kc=0.2, beta=200.0, u0=0.02, box=6.0, transient=8.0, measure=8.0)
    r, tr = run_arm(c, threads=4)
    n = r["numerics"]
    T = n["T_steps"]
    A = r["geometry"]["n_solid"]
    k1 = 1.0 / (n["a0"] * A)
    ref = r["reference"]
    print(f"D={c.D} box={c.box} T={T} n_steps={n['n_steps']} nu={n['nu']:.2e} tau={n['tau']:.4f} "
          f"wL/cs={n['omega']*n['nx']/CS:.2f} a0={n['a0']:.3e}")
    print(f"ref: same_mask_area={ref['potential_same_mask_area']:.4f} "
          f"halfway={ref['potential_same_mask_halfway']:.4f} unbounded=1.0")
    print(f"{'win':>4} {'t0':>6} {'dc*':>8} {'C_I':>8} {'C_Q':>8} {'|C|':>7} {'phi':>7} "
          f"{'2w_amp*':>8} {'resid':>6} {'Uamp':>9} {'Ucos':>9}")
    for k in range(0, 8):
        a = k * T
        b = a + T
        d = harmonic_decompose(tr["t"], tr["F_mem"], tr["omega"], t0=a, t1=b, nh=2)
        du = harmonic_decompose(tr["t"], tr["U"], tr["omega"], t0=a, t1=b, nh=1)
        h2 = d["harmonics"][1]
        print(f"{k:>4} {a:>6.0f} {d['dc']*k1:>+8.3f} {d['cos']*k1:>+8.3f} {d['sin']*k1:>+8.3f} "
              f"{math.hypot(d['cos'],d['sin'])*k1:>7.3f} {math.degrees(math.atan2(d['sin'],d['cos'])):>+7.1f} "
              f"{math.hypot(h2['cos'],h2['sin'])*k1:>8.3f} {d['resid_rel']:>6.3f} "
              f"{du['amp']:>9.3e} {du['cos']:>9.2e}")
    # raw series tail sanity
    F = np.asarray(tr["F_mem"])
    t = np.asarray(tr["t"])
    print(f"F range over run: min={F.min():+.4f} max={F.max():+.4f} head={F[:6].round(4)}")
    print(f"F tail(one period, every T/8): {F[-int(T):][::max(1,int(T)//8)].round(4)}")


if __name__ == "__main__":
    main()