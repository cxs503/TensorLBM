#!/usr/bin/env python
"""W5-C pre-registration diagnostic: effective shear viscosity of each collision kernel.

Shear-wave (transverse wave) decay on a small periodic lattice, fp64, CPU.
Purpose: confirm the documented mapping nu = (tau - 1/2)/3 for each kernel so
the Taylor-Green analytic reference gamma_E = 6*nu*k^2 is the correct target.

All collide/stream/equilibrium/macroscopic operations are library calls from the
read-only bm_w5 worktree (main @ cf5709db3c). This script only orchestrates.

Mode: u = (A*sin(k*y), 0, 0), k = 2*pi/L, box (nz,nx,nx)=(T,L,T).
Linear theory: A(t) = A0 * exp(-nu_eff * k^2 * t).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

WORKTREE_SRC = "/nfs/wangxi/worktrees/bm_w5/src"
sys.path.insert(0, WORKTREE_SRC)

import tensorlbm  # noqa: E402

assert tensorlbm.__file__.startswith(WORKTREE_SRC), f"wrong tensorlbm copy: {tensorlbm.__file__}"

from tensorlbm.cascaded_collision import collide_cascaded_d3q27  # noqa: E402
from tensorlbm.cumulant import collide_cumulant_d3q27  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.d3q27 import (  # noqa: E402
    collide_mrt27,
    collide_rlbm27,
    collide_trt27,
    equilibrium27,
    macroscopic27,
    stream27_roll,
)
from tensorlbm.entropic_kbc import collide_kbc_d3q19  # noqa: E402
from tensorlbm.solver3d import stream3d  # noqa: E402

KERNELS = {
    "cumulant_d3q27": dict(q=27, op=lambda f, t: collide_cumulant_d3q27(f, t)),
    "cascaded_d3q27": dict(q=27, op=lambda f, t: collide_cascaded_d3q27(f, t)),
    "mrt27": dict(q=27, op=lambda f, t: collide_mrt27(f, t)),
    "kbc_d3q19": dict(q=19, op=lambda f, t: collide_kbc_d3q19(f, t)),
    "trt27": dict(q=27, op=lambda f, t: collide_trt27(f, t)),
    "rlbm27": dict(q=27, op=lambda f, t: collide_rlbm27(f, t)),
}


def shear_wave_case(
    kernel: str,
    tau: float,
    amp: float,
    length: int = 32,
    transverse: int = 4,
    steps: int = 200,
    fit_start: int = 20,
    dtype: str = "float64",
) -> dict:
    info = KERNELS[kernel]
    dt = torch.float64 if dtype == "float64" else torch.float32
    nz, ny, nx = transverse, length, transverse
    k = 2.0 * math.pi / length

    y = torch.arange(ny, dtype=dt)
    sin_ky = torch.sin(k * y)  # (ny,)
    ux_line = amp * sin_ky
    ux = ux_line.view(1, ny, 1).expand(nz, ny, nx).contiguous()
    uy = torch.zeros_like(ux)
    uz = torch.zeros_like(ux)
    rho = torch.ones_like(ux)

    if info["q"] == 19:
        f = equilibrium3d(rho, ux, uy, uz)
        stream_fn = stream3d
        macro_fn = macroscopic3d
    else:
        f = equilibrium27(rho, ux, uy, uz)
        stream_fn = stream27_roll
        macro_fn = macroscopic27

    sin2_mean = float((sin_ky * sin_ky).mean())
    amps: list[float] = []
    for step in range(steps + 1):
        if step > 0:
            f = info["op"](stream_fn(f), tau)
        _, uxm, _, _ = macro_fn(f)
        line = uxm.mean(dim=(0, 2))  # (ny,)
        amps.append(float((line * sin_ky).mean() / sin2_mean))

    t = torch.arange(len(amps), dtype=torch.float64)[fit_start:]
    ln_a = torch.log(torch.tensor(amps, dtype=torch.float64)[fit_start:])
    slope = float(((t - t.mean()) * (ln_a - ln_a.mean())).sum() / ((t - t.mean()) ** 2).sum())
    nu_eff = -slope / (k * k)
    nu_target = (tau - 0.5) / 3.0
    return {
        "kernel": kernel,
        "tau": tau,
        "amp": amp,
        "nu_target": nu_target,
        "nu_eff": nu_eff,
        "err_pct": (nu_eff - nu_target) / nu_target * 100.0,
        "length": length,
        "steps": steps,
        "fit_start": fit_start,
        "dtype": dtype,
    }


def main() -> None:
    torch.set_num_threads(16)
    out_path = Path(__file__).resolve().parent / "audit_viscosity.json"
    rows = []
    for name in KERNELS:
        for tau in (0.9, 1.1, 1.3):
            for amp in (1.0e-3, 0.05):
                r = shear_wave_case(name, tau, amp)
                rows.append(r)
                print(
                    f"{name:16s} tau={tau:.2f} amp={amp:7.4f}  "
                    f"nu_eff={r['nu_eff']:.8f} target={r['nu_target']:.8f} "
                    f"err={r['err_pct']:+.4f}%",
                    flush=True,
                )
    with open(out_path, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
