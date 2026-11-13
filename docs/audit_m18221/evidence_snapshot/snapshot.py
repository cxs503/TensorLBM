#!/usr/bin/env python
"""Final-state field snapshot for the report figures (density + velocity cloud).

Kept separate from ``xval_driver.py`` so that the driver sha256 recorded in
``artifacts/results.json`` stays valid for the production campaign.

Writes ``artifacts/snapshot.npz`` (rho, |u|, Ma) plus the run metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "benchmarks" / "eos_maxwell"))
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

import probe_lib as P  # noqa: E402
from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.multiphase import (  # noqa: E402
    collide_sc_single_component,
    make_psi_eos,
)
from tensorlbm.solver import stream  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T-r", dest="T_r", type=float, default=0.8266)
    ap.add_argument("--L", type=int, default=96)
    ap.add_argument("--R", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", default=str(HERE / "artifacts" / "snapshot.npz"))
    a = ap.parse_args()

    torch.set_num_threads(4)
    mw = P.maxwell_at_tr("pr", a.T_r, G=1.0)
    device = torch.device("cpu")
    rho0, r = P.droplet_field(a.L, a.R, mw["rho_liquid"], mw["rho_vapour"], device, 4.0)
    zero = torch.zeros_like(rho0)
    f = equilibrium(rho0, zero, zero)
    psi_fn = make_psi_eos("pr", mw["T"])

    t0 = time.time()
    for _ in range(a.steps):
        f = stream(collide_sc_single_component(f, G=1.0, tau=1.0, psi_fn=psi_fn,
                                              forcing="edm"))
    rho = f.sum(dim=0)
    rho_t = torch.clamp(rho, min=1e-12)
    ux = (f[1] - f[3] + f[5] - f[6] - f[7] + f[8]) / rho_t
    uy = (f[2] - f[4] + f[5] + f[6] - f[7] - f[8]) / rho_t
    umag = torch.sqrt(ux * ux + uy * uy)
    cs = float(np.sqrt(1.0 / 3.0))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        a.out,
        rho=rho.numpy(), umag=umag.numpy(), ux=ux.numpy(), uy=uy.numpy(),
        r=r.numpy(),
        meta=json.dumps(dict(
            T_r=a.T_r, T=mw["T"], L=a.L, R=a.R, steps=a.steps,
            rho_l_maxwell=mw["rho_liquid"], rho_v_maxwell=mw["rho_vapour"],
            ratio_maxwell=mw["ratio"], forcing="edm", tau=1.0, G=1.0,
            ma_max=float(umag.max()) / cs, wall_s=round(time.time() - t0, 1),
            finite=bool(torch.isfinite(rho).all()),
        )),
    )
    print(f"snapshot: Ma_max={float(umag.max())/cs:.4g} rho=[{float(rho.min()):.4g},"
          f"{float(rho.max()):.4g}] wall={time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()