"""CH4 static-droplet validation (DE-X-PF1-CH4, wave W1).

A single liquid droplet at rest in a periodic box.  The Cahn-Hilliard
fourth-order phase field relaxes to the equilibrium tanh profile while the
coupled D3Q19 fluid develops a (parasitic) spurious current.  We record, for
each density ratio:

* the maximum lattice speed (spurious flow),
* the interface thickness (0.1 -> 0.9 width of the radial phi profile),
* the liquid volume conservation,
* the Laplace pressure ``dp = c_s^2 (rho_in - rho_out)``.

Run locally (CPU) or on the SDAA card.  Small grid, few hundred steps.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorlbm.ch4_phasefield import (
    ch4_coupled_step,
    ch4_init,
)


def _radial_profile(phi: torch.Tensor, center: int, radius: int) -> list[tuple[int, float]]:
    """Average phi on shells of integer radius around the centre."""
    nz, ny, nx = phi.shape
    z, y, x = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij"
    )
    r = ((z - center) ** 2 + (y - center) ** 2 + (x - center) ** 2).sqrt()
    out = []
    for rr in range(max(0, radius - 6), min(max(nz, ny, nx), radius + 7)):
        mask = (r - rr).abs() < 0.5
        if int(mask.sum()) > 0:
            out.append((rr, float(phi[mask].mean())))
    return out


def static_droplet(
    n: int = 32,
    radius: float = 12.0,
    w: float = 3.0,
    A: float = 1e-5,
    kappa: float | None = None,
    M: float = 0.1,
    tau_g: float = 0.8,
    tau_f: float = 0.5,
    force_scale: float = 0.01,
    rho_g: float = 1.0,
    rho_l: float = 1000.0,
    steps: int = 500,
    seed: int = 0,
) -> dict:
    """Run the static-droplet test and return a JSON-safe summary."""
    torch.manual_seed(seed)
    if kappa is None:
        kappa = A * w * w
    phi = torch.zeros(n, n, n)
    z, y, x = torch.meshgrid(
        torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij"
    )
    r = ((z - n / 2) ** 2 + (y - n / 2) ** 2 + (x - n / 2) ** 2).sqrt()
    phi = torch.tanh((r - radius) / w)
    f, g = ch4_init(phi, A, kappa, M, rho_g, rho_l)
    vol0 = float(((phi + 1) / 2).sum())
    c = None
    umax_hist = []
    for s in range(steps):
        f, g, phi = ch4_coupled_step(
            f, g, phi, A, kappa, M, tau_g, tau_f, 0.0, force_scale, rho_g, rho_l
        )
        if s % 50 == 0 or s == steps - 1:
            rho_f = f.sum(0)
            if c is None:
                from tensorlbm.ch4_phasefield import _c_on

                c = _c_on(phi.device)
            mom = (f * c[:, 0].view(19, 1, 1, 1)).sum(0)
            u = mom / rho_f.clamp(min=1e-12)
            umax = float(u.abs().max())
            umax_hist.append({"step": s, "umax": umax})
            if not math.isfinite(umax):
                return {
                    "status": "nonfinite",
                    "step": s,
                    "rho_l": rho_l,
                    "umax_hist": umax_hist,
                }
    vol = float(((phi + 1) / 2).sum())
    prof = _radial_profile(phi, n // 2, int(radius))
    # interface thickness: 0.1 -> 0.9 width of the radial profile
    h = None
    r01 = r09 = None
    for i in range(1, len(prof)):
        (r_a, v_a), (r_b, v_b) = prof[i - 1], prof[i]
        if v_a <= 0.1 < v_b and r01 is None:
            r01 = r_a + (0.1 - v_a) / (v_b - v_a)
        if v_a <= 0.9 < v_b and r09 is None:
            r09 = r_a + (0.9 - v_a) / (v_b - v_a)
    if r01 is not None and r09 is not None:
        h = r09 - r01
    # Laplace pressure from the fluid density
    rho_f = f.sum(0)
    rho_in = float(rho_f[n // 2, n // 2, n // 2])
    rho_out = float(rho_f[0, n // 2, n // 2])
    dp = (1.0 / 3.0) * (rho_in - rho_out)
    return {
        "status": "ok",
        "n": n,
        "radius": radius,
        "w": w,
        "A": A,
        "kappa": kappa,
        "M": M,
        "tau_g": tau_g,
        "tau_f": tau_f,
        "force_scale": force_scale,
        "rho_g": rho_g,
        "rho_l": rho_l,
        "steps": steps,
        "umax": umax_hist[-1]["umax"],
        "umax_hist": umax_hist,
        "volume_drift": abs(vol / vol0 - 1),
        "interface_thickness_01_09": h,
        "laplace_dp": dp,
        "radial_profile": prof,
    }


def main() -> None:
    out = Path(__file__).with_name("ch4_static_results.json")
    results = []
    for rho_l in (10.0, 100.0, 1000.0):
        res = static_droplet(rho_l=rho_l)
        results.append(res)
        print(
            f"rho_l={rho_l}: status={res['status']} "
            f"umax={res.get('umax', float('nan')):.3e} "
            f"h={res.get('interface_thickness_01_09', float('nan'))} "
            f"dvol={res.get('volume_drift', float('nan')):.2e} "
            f"dp={res.get('laplace_dp', float('nan')):.3e}"
        )
    out.write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
