"""Probe: is the high-Pe ladder residual a resolution effect or renorm/meter bias?

Runs the same cells with and without the amplitude renormalisation of the mode
meter, plus one refined rung, so the P74-N10 ladder rule can be fixed on
evidence instead of preference.
"""

import sys
import time

sys.path.insert(0, "/root/tp-agent/work/m9469/src")

import torch

torch.set_num_threads(1)

from tensorlbm.taylor_aris import DispersionConfig, simulate_channel_dispersion

U_REF = 0.09
PLAN = [
    # (pe, ny, steps, renorm, fit_start_frac)
    (20.0, 33, 7326, True, 0.5),
    (20.0, 33, 7326, False, 0.5),
    (40.0, 17, 7530, True, 0.5),
    (40.0, 17, 7530, False, 0.5),
    (40.0, 65, 28900, True, 0.5),
]

for pe, ny, steps, renorm, frac in PLAN:
    diffusivity = U_REF * ny / pe
    started = time.time()
    out = simulate_channel_dispersion(
        DispersionConfig(
            nx=128,
            ny=ny,
            diffusivity=diffusivity,
            u_mean=U_REF,
            steps=steps,
            method="mode",
            renorm=renorm,
            fit_start_frac=frac,
            dtype="float64",
            report_every=max(1, steps // 200),
        )
    )
    ratio = out["d_eff"] / out["d_molecular"]
    closed = 1.0 + pe**2 / 210.0
    print(
        f"pe={pe:5g} ny={ny:3d} steps={steps:6d} renorm={renorm} D={diffusivity:.4f} "
        f"tau_d={3 * diffusivity + 0.5:.3f} ratio={ratio:.5f} closed={closed:.5f} "
        f"err={100 * (ratio / closed - 1):+.3f}% r2={out['r2']:.5f} "
        f"t={time.time() - started:.1f}s",
        flush=True,
    )