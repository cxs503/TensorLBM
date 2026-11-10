"""Probe: does the mode meter need a longer domain at high Pe_r?

The Taylor-Aris asymptote requires the longitudinal decay rate
``D_eff k^2`` to be much smaller than the transverse relaxation rate
``pi^2 D / h^2``.  With ``k = 2 pi / nx`` that is
``(D_eff/D) (2 pi h / nx)^2 << pi^2``, i.e. ``nx`` must grow like
``h sqrt(D_eff/D)``.  This probe holds (Pe_r, N_y, D, steps) fixed and only
lengthens the domain.
"""

import sys
import time

sys.path.insert(0, "/root/tp-agent/work/m9469/src")

import torch

torch.set_num_threads(1)

from tensorlbm.taylor_aris import DispersionConfig, simulate_channel_dispersion

U_REF = 0.09
PLAN = [
    (40.0, 17, 256, 7530),
    (40.0, 17, 512, 7530),
    (20.0, 33, 256, 7326),
    (40.0, 33, 512, 14653),
]

for pe, ny, nx, steps in PLAN:
    diffusivity = U_REF * ny / pe
    started = time.time()
    out = simulate_channel_dispersion(
        DispersionConfig(
            nx=nx,
            ny=ny,
            diffusivity=diffusivity,
            u_mean=U_REF,
            steps=steps,
            method="mode",
            dtype="float64",
            report_every=max(1, steps // 200),
        )
    )
    ratio = out["d_eff"] / out["d_molecular"]
    closed = 1.0 + pe**2 / 210.0
    k = 2.0 * 3.141592653589793 / nx
    h = 0.5 * ny
    criterion = (closed) * (k * h) ** 2 / 3.141592653589793**2
    print(
        f"pe={pe:5g} ny={ny:3d} nx={nx:4d} D={diffusivity:.4f} "
        f"ratio={ratio:.5f} closed={closed:.5f} err={100 * (ratio / closed - 1):+.3f}% "
        f"rate_ratio={criterion:.3f} r2={out['r2']:.5f} t={time.time() - started:.1f}s",
        flush=True,
    )