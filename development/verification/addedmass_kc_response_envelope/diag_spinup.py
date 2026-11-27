#!/usr/bin/env python3
"""Scratch diagnostic 6: spin-up ladder.  Hypothesis: the drift of the fitted
phase per period is the slowest viscous mode of the box, decay time
(L/2pi)^2/nu, i.e. the required spin-up is N_spin ~ beta*(box/2pi)^2 periods.
Cheap geometry (D=12, box=4, beta=25) -> N_spin ~ 10."""
from __future__ import annotations

import math
import sys

import numpy as np

sys.path.insert(0, "/srv/tensor/agent/work/m20245/src")

from tensorlbm.oscillatory_cylinder_inertia import ArmCase, harmonic_decompose, run_arm


def main():
    D, box, kc, u0, beta = 12, 4.0, 0.2, 0.02, 25.0
    T = kc * D / u0
    nu = D * D / (beta * T)
    L = box * D
    n_spin = beta * (L / (2 * math.pi) / D) ** 2
    print(f"D={D} box={box} kc={kc} u0={u0} beta={beta}: T={T:.0f} nu={nu:.4f} "
          f"tau={(3*nu+0.5):.4f} L={L:.0f}; predicted N_spin ~ beta*(L/2piD)^2 = {n_spin:.1f}")
    print(f"{'spin':>5} {'n_steps':>8} {'C_I':>8} {'C_Q':>8} {'|C|':>7} {'phi':>7} "
          f"{'resid':>6} {'drift(deg/win)':>15} {'ref':>6} {'errI%':>7}")
    for spin in (2, 10, 25, 50):
        c = ArmCase(f"s{spin}", D=D, kc=kc, beta=beta, u0=u0, box=box,
                    transient=float(spin), measure=5.0)
        r, tr = run_arm(c, threads=4)
        n = r["numerics"]
        k = r["kpi"]["E1_mem_area"]
        ref = r["reference"]["potential_same_mask_area"]
        phis = []
        for j in range(5):
            a = n["measure_start_step"] + j * T
            d = harmonic_decompose(tr["t"], tr["F_mem"], tr["omega"], t0=a, t1=a + T, nh=1)
            phis.append(math.degrees(math.atan2(d["sin"], d["cos"])))
        dr = np.polyfit(range(5), phis, 1)[0] if len(phis) > 1 else 0.0
        print(f"{spin:>5} {n['n_steps']:>8} {k['C_I']:>+8.3f} {k['C_Q']:>+8.3f} "
              f"{math.hypot(k['C_I'],k['C_Q']):>7.3f} {math.degrees(math.atan2(k['C_Q'],k['C_I'])):>+7.1f} "
              f"{r['harmonic']['E1_mem']['resid_rel']:>6.3f} {dr:>+15.2f} {ref:>6.3f} "
              f"{100*(k['C_I']-ref)/ref:>+7.2f}", flush=True)


if __name__ == "__main__":
    main()