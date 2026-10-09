#!/usr/bin/env python
"""W4-B Part 1 formal: two-phase Poiseuille ladder, H=64/128 (FAIL record).

tau=(1.0,0.75) nominal M=2, G=-2.5 full-swap, G_x=5e-6, tanh-3 interface.
Analytical = library's own porous_media._two_phase_poiseuille_analytical
(nominal nu; recalibration with the measured viscosity ratio was
pre-registered as FORBIDDEN and was not done).
Library kernels only (w4b_lib); grep iron rule: no hand-written physics
kernels in this driver. Original staging driver 2026-09-21, archived with
path headers rewritten only. Requires one free CUDA device."""

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))  # <repo>/src
sys.path.insert(0, str(Path(__file__).parent))

from w4b_lib import init_full_swap, mixture_fields, sc_mcmp_step, tanh_field_1d  # noqa: E402

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.porous_media import _two_phase_poiseuille_analytical  # noqa: E402

OUT = Path(__file__).parent
dev = torch.device("cuda:6")
TAU_W, TAU_G, G12, GX = 1.0, 0.75, -2.5, 5e-6
NX = 8


def run(ny, n_steps, tag):
    half = ny // 2
    alpha = tanh_field_1d(ny, NX, float(half), 3.0, dev)
    rw, rg = init_full_swap(alpha, 0.7, 0.3)
    z = torch.zeros((ny, NX), device=dev)
    fw, fg = equilibrium(rw, z, z), equilibrium(rg, z, z)
    wall = torch.zeros((ny, NX), dtype=torch.bool, device=dev)
    wall[0, :] = True
    wall[-1, :] = True
    nu_w, nu_g = (TAU_W - 0.5) / 3.0, (TAU_G - 0.5) / 3.0
    ana = _two_phase_poiseuille_analytical(ny, half, GX, nu_w, nu_g, nu_w / nu_g)
    ana_t = torch.tensor(ana, device=dev)
    conv = []
    prev = None
    for step in range(1, n_steps + 1):
        fw, fg = sc_mcmp_step(fw, fg, G12, TAU_W, TAU_G, GX, 0.0, wall)
        if step % 20000 == 0 or step == n_steps:
            rwt, rgt, rt, ux, uy = mixture_fields(fw, fg)
            umax = float(ux[1:-1, :].max())
            finite = bool(torch.isfinite(fw).all() and torch.isfinite(fg).all())
            conv.append({"step": step, "u_max": umax, "finite": finite})
            rel = abs(umax - prev) / prev if prev else None
            print(
                f"  {tag} step={step} u_max={umax:.6f} finite={finite} conv_rel={rel}", flush=True
            )
            prev = umax
            if not finite:
                break
    rwt, rgt, rt, ux, uy = mixture_fields(fw, fg)
    prof = ux[:, NX // 2].cpu().tolist()
    prof_t = ux[:, NX // 2]
    umax_s = float(prof_t.max())
    m_sim = prof_t.abs() > 0.2 * umax_s
    m_ana = ana_t.abs() > 0.2 * max(ana)

    def errs(mask):
        e = (prof_t[mask] - ana_t[mask]) / ana_t[mask]
        return {
            "n": int(mask.sum()),
            "max_rel_err_pct": float(e.abs().max()) * 100,
            "mean_rel_err_pct": float(e.abs().mean()) * 100,
        }

    spur = float(uy[1:-1, 1:-1].abs().max())
    return {
        "tag": tag,
        "ny": ny,
        "nx": NX,
        "H_eff": ny - 1,
        "half": half,
        "tau_w": TAU_W,
        "tau_g": TAU_G,
        "G_12": G12,
        "G_x": GX,
        "nominal_M": nu_w / nu_g,
        "nu_w": nu_w,
        "nu_g": nu_g,
        "n_steps": n_steps,
        "convergence": conv,
        "u_profile_sim": prof,
        "u_profile_analytical": ana,
        "u_max_sim": umax_s,
        "u_max_ana": max(ana),
        "u_max_ratio": umax_s / max(ana),
        "center_err_mask_simmax": errs(m_sim),
        "center_err_mask_anamax": errs(m_ana),
        "ratio_profile": [(prof[i] / ana[i]) if ana[i] > 1e-12 else None for i in range(ny)],
        "spurious_uy_max": spur,
        "Ma_max": umax_s / math.sqrt(1 / 3.0),
        "mass_water": float(rw.sum()),
        "mass_gas": float(rg.sum()),
        "mass_total": float(rw.sum() + rg.sum()),
        "rho_g_in_water_bulk": float(rg[: half - 8, 2:6].mean()),
        "rho_w_in_gas_bulk": float(rwt[half + 8 :, 2:6].mean()),
        "finite": conv[-1]["finite"],
    }


def main():
    results = {}
    for ny, n in ((65, 60000), (129, 180000)):
        tag = f"H{ny - 1}"
        print(f"Part 1 formal {tag}", flush=True)
        results[tag] = run(ny, n, tag)
        c = results[tag]["center_err_mask_simmax"]
        print(
            f"{tag}: max_err={c['max_rel_err_pct']:.2f}% mean={c['mean_rel_err_pct']:.2f}% "
            f"u_ratio={results[tag]['u_max_ratio']:.4f} spur={results[tag]['spurious_uy_max']:.4f}",
            flush=True,
        )
    with open(OUT / "out_part1.json", "w") as f:
        json.dump(results, f, indent=2)
    print("saved -> out_part1.json")


if __name__ == "__main__":
    main()
