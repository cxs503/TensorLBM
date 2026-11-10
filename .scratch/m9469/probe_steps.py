"""Probe: how long must the dispersion run be before the fitted rate is asymptotic?

Writes one line per (Pe, N_y, steps, fit_start_frac) to stdout.  Used to pin the
``steps`` rule of the P74-N10 surface on physically defensible grounds
(t_end >> h^2/D) instead of picking whatever passes.
"""

import sys
import time

sys.path.insert(0, "/root/tp-agent/work/m9469/src")

import torch

torch.set_num_threads(1)

from tensorlbm.taylor_aris import DispersionConfig, simulate_channel_dispersion

U_REF = 0.09
PLAN = [(5.0, 9, (2000, 4000, 8000)), (20.0, 9, (2000, 4000, 8000))]

for pe, ny, step_list in PLAN:
    diffusivity = U_REF * ny / pe
    for steps in step_list:
        for frac in (0.5, 0.75):
            started = time.time()
            out = simulate_channel_dispersion(
                DispersionConfig(
                    nx=128,
                    ny=ny,
                    diffusivity=diffusivity,
                    u_mean=U_REF,
                    steps=steps,
                    method="mode",
                    fit_start_frac=frac,
                    dtype="float64",
                    report_every=max(1, steps // 200),
                )
            )
            ratio = out["d_eff"] / out["d_molecular"]
            closed = 1.0 + pe**2 / 210.0
            print(
                f"pe={pe:5g} ny={ny} steps={steps:6d} frac={frac} "
                f"D={diffusivity:.4f} ratio={ratio:.5f} closed={closed:.5f} "
                f"err={100 * (ratio / closed - 1):+.3f}% r2={out['r2']:.5f} "
                f"t={time.time() - started:.1f}s",
                flush=True,
            )