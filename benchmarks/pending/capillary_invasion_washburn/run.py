"""Formal Part-2 capillary invasion ladder (Washburn) — W4-B (FAIL record).

Protocol locked in NOTES.md amendment #2 (sha256 26e2f9f5...):
  solid seam column x=nx-1 (defect #9 workaround) + gas reservoir col 0
  (P_in = cs^2*drho) + water sink col nx-2 (P_0), coexistence-matched init,
  tau=(1.0,0.75) Design A, G=-2.5, drho = 3*1.8/Weff^2 (K=1 viscous sizing;
  sigma unmeasurable -> no capillary term).  Neutral walls (G_ads=0),
  theta = 83.33288817404761 deg from selftest_theta.

Physics kernels: library only.  This driver contains NO collide/stream/
equilibrium-of-its-own; grep check in the record README.
Original staging driver 2026-09-21, archived with path headers rewritten
only.  Requires one free CUDA device."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # <repo>/src
sys.path.insert(0, str(Path(__file__).parent))

from w4b_lib import (  # noqa: E402
    CS2,
    equilibrium,
    gas_front,
    init_from_fractions,
    mixture_fields,
    sc_mcmp_step,
    tube_solid_mask,
)

DEV = torch.device("cuda:6")
OUT = Path(__file__).parent

# Coexistence pair measured at tau=(1.0,0.75), G=-2.5 (Part-1 terminal state)
RW_W, RG_W = 0.7, 0.3001  # water-rich phase
RW_G, RG_G = 0.1593, 0.8407  # gas-rich phase

RUNGS = [
    dict(W=32, tw=31, ny=33, steps=40000, sample=100),
    dict(W=64, tw=63, ny=65, steps=40000, sample=100),
    dict(W=128, tw=127, ny=129, steps=40000, sample=100),
]
NX = 600
TAU_W, TAU_G, G12 = 1.0, 0.75, -2.5
X0, IW = 33.5, 3.0


def run_rung(rung):
    W, tw, ny = rung["W"], rung["tw"], rung["ny"]
    nsteps, samp = rung["steps"], rung["sample"]
    Weff = tw + 1
    drho = 3.0 * 1.8 / Weff**2
    dP = CS2 * drho

    solid = tube_solid_mask(ny, NX, tw, DEV)
    solid[:, -1] = True  # solid seam column buries periodic wrap (defect #9)

    xs = torch.arange(NX, dtype=torch.float32, device=DEV)
    alpha = 0.5 * (1.0 + torch.tanh((xs - X0) / IW)).unsqueeze(0).expand(ny, NX)
    rho_w, rho_g = init_from_fractions(alpha, RW_W, RG_W, RW_G, RG_G)
    zero = torch.zeros_like(rho_w)
    fw = equilibrium(rho_w, zero, zero)
    fg = equilibrium(rho_g, zero, zero)

    s_in = 1.0 + drho
    z = torch.zeros((ny, 1), device=DEV)
    feq_w_in = equilibrium(torch.full((ny, 1), RW_G * s_in, device=DEV), z, z)
    feq_g_in = equilibrium(torch.full((ny, 1), RG_G * s_in, device=DEV), z, z)
    feq_w_out = equilibrium(torch.full((ny, 1), RW_W, device=DEV), z, z)
    feq_g_out = equilibrium(torch.full((ny, 1), RG_W, device=DEV), z, z)

    series = []
    t0 = time.time()
    for step in range(1, nsteps + 1):
        fw, fg = sc_mcmp_step(fw, fg, G12, TAU_W, TAU_G, solid=solid)
        fw[:, :, 0:1] = feq_w_in  # gas reservoir  (P_in)
        fg[:, :, 0:1] = feq_g_in
        fw[:, :, -2:-1] = feq_w_out  # water sink     (P_0)
        fg[:, :, -2:-1] = feq_g_out
        if step % samp == 0 or step == nsteps:
            L = gas_front(fw, fg, solid)
            rw_, rg_, _, ux, uy = mixture_fields(fw, fg)
            fluid = ~solid
            umax = float(torch.sqrt(ux**2 + uy**2)[fluid].max().item())
            minf = float(min(fw[:, fluid].min().item(), fg[:, fluid].min().item()))
            series.append(
                dict(
                    step=step,
                    L=round(L, 2),
                    massw=round(float(rw_[fluid].sum().item()), 3),
                    massg=round(float(rg_[fluid].sum().item()), 3),
                    umax=round(umax, 6),
                    minf=round(minf, 6),
                )
            )
            if step % (samp * 50) == 0:
                print(f"W={W} step={step} L={L:.0f} umax={umax:.4f} minf={minf:.5f}", flush=True)
        if not (torch.isfinite(fw).all() and torch.isfinite(fg).all()):
            print(f"W={W} NONFINITE at step {step}", flush=True)
            break

    rw_, rg_, _, _, _ = mixture_fields(fw, fg)
    phi = (rg_ / (rw_ + rg_ + 1e-12)).cpu().numpy()
    np.savez_compressed(OUT / f"out_part2_W{W}_final.npz", phi=phi, solid=solid.cpu().numpy())

    cfg = dict(
        W=W,
        tw=tw,
        Weff=Weff,
        ny=ny,
        nx=NX,
        steps=nsteps,
        sample=samp,
        tau_water=TAU_W,
        tau_gas=TAU_G,
        G12=G12,
        drho=drho,
        dP=dP,
        x0=X0,
        interface_width=IW,
        L0=X0,
        wettability="neutral (G_ads=0 both components)",
        theta_measured_deg=83.33288817404761,
        nu_water=CS2 * (TAU_W - 0.5),
        nu_gas=CS2 * (TAU_G - 0.5),
    )
    out = dict(
        config=cfg,
        series=series,
        wall_time_s=round(time.time() - t0, 1),
        completed_steps=series[-1]["step"] if series else 0,
    )
    (OUT / f"out_part2_W{W}.json").write_text(json.dumps(out, indent=1))
    print(f"== W={W} done in {out['wall_time_s']}s, final L={series[-1]['L']}", flush=True)


def main():
    for rung in RUNGS:
        run_rung(rung)
    print("ALL RUNGS DONE", flush=True)


if __name__ == "__main__":
    main()
