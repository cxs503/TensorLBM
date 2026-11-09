#!/usr/bin/env python
"""Negative controls for the SC-LBM interface tension (node P143W2-N01)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

torch.set_num_threads(8)

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.sc_surface_tension import (
    build_stencil, droplet_rho, equivalent_radius, pressure_flux, radius_field,
    sigma_korteweg, step_sc,
)
from run import _jsonable

OUT = REPO / "artifacts" / "sc_surface_tension"


def kappa3_zero(L: int = 96, R: float = 20.0, steps: int = 6000,
                stencil: str = "iso4") -> dict:
    """kappa3 = 0 with a *local* stencil (the 4th-order isotropic one).

    The spectral gradient is the exact kappa3 = 0 limit but its Gibbs ringing
    destabilises a 12:1 density interface (measured: blow-up at step ~1000 at
    L=96), so the local 4th-order stencil is used instead: it has a3 = 0 exactly
    and is stable.
    """
    st = build_stencil(stencil)
    rho0 = droplet_rho(L, R)
    z = torch.zeros_like(rho0)
    f = equilibrium(rho0, z, z)
    for step in range(1, steps + 1):
        f = step_sc(f, stencil=st)
        if step % 1000 == 0 and not bool(torch.isfinite(f.sum(0)).all()):
            raise RuntimeError(f"spectral run diverged at {step}")
    rho = f.sum(0)
    r = radius_field(L)
    Req = equivalent_radius(rho)
    p = pressure_flux(f)
    inn, out = r < 0.5 * Req, r > 1.5 * Req
    _, ux, uy = macroscopic(f)
    return {
        "case": "kappa3_zero_spectral", "L": L, "R_init": R, "R_eq": Req,
        "dp_flux": float(p[inn].mean() - p[out].mean()),
        "rho_in": float(rho[inn].mean()), "rho_out": float(rho[out].mean()),
        "u_max": float(torch.sqrt(ux ** 2 + uy ** 2).max()),
        "sigma_K": sigma_korteweg(rho, L, st)["sigma_K"],
        "steps": steps, "gradient_mode": "stencil", "stencil": stencil,
    }


def g_eff_zero(L: int = 96, steps: int = 3000) -> dict:
    y = torch.arange(L, dtype=torch.float32).view(L, 1)
    x = torch.arange(L, dtype=torch.float32).view(1, L)
    r = torch.sqrt((x - 0.5 * (L - 1)) ** 2 + (y - 0.5 * (L - 1)) ** 2)
    rho0 = 1.2 * (1.0 + 0.01 * torch.exp(-(r / 20.0) ** 2))
    z = torch.zeros_like(rho0)
    f = equilibrium(rho0, z, z)
    st = build_stencil("d2q9")
    for step in range(1, steps + 1):
        f = step_sc(f, G_eff=0.0, stencil=st)
        if step % 1000 == 0 and not bool(torch.isfinite(f.sum(0)).all()):
            raise RuntimeError(f"G_eff=0 run diverged at {step}")
    rho = f.sum(0)
    r = radius_field(L)
    Req = 20.0
    p = pressure_flux(f)
    inn, out = r < 0.5 * Req, r > 1.5 * Req
    return {
        "case": "G_eff_zero", "L": L, "R_init": Req, "R_eq": equivalent_radius(rho),
        "dp_flux": float(p[inn].mean() - p[out].mean()),
        "rho_in": float(rho[inn].mean()), "rho_out": float(rho[out].mean()),
        "steps": steps, "gradient_mode": "stencil", "G_eff": 0.0,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (kappa3_zero, g_eff_zero):
        res = fn()
        (OUT / f"negative_{res['case']}.json").write_text(json.dumps(_jsonable(res), indent=1))
        print(f"[negative] {res['case']}: dp_flux={res['dp_flux']:.3e} rho={res['rho_in']:.3f}/{res['rho_out']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
