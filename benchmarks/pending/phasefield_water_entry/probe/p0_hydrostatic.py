"""P0 probe — still-water tank (hydrostatic) gate for a phase-field water-entry benchmark.

Question this probe answers
---------------------------
A phase-field water-entry case can only be attempted if the *undeformed*
configuration is already in hydrostatic balance: a flat air/water interface
must stay flat, the pressure field must carry the weight of the water column,
and the spurious velocity must stay small.  This script runs the library
conservative Allen-Cahn (CAC) two-phase model (``tensorlbm.cac_lbm``) in a
closed, walled tank with per-mass gravity and measures

* interface drift ``dy_iface(t)`` (phi = 0.5 crossing on the tank centreline),
* hydrostatic pressure slope ``dp/dy`` versus the imposed ``rho_l * g``,
* spurious current ``max|u|``,
* phase-field integral drift ``d(Int phi)/Int phi_0``,
* NaN-free requirement.

It is a prerequisite probe, NOT a benchmark: no external reference data is
compared here.  Nothing in this file may be quoted as a validated water-entry
result.

Run
---
    PYTHONPATH=src python benchmarks/pending/phasefield_water_entry/probe/p0_hydrostatic.py \
        --out /tmp/p0_hydrostatic.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorlbm import cac_lbm as m


def wall_mask(shape: tuple[int, int, int]) -> torch.Tensor:
    """All six faces of the tank are solid (1-cell thick)."""
    w = torch.zeros(shape, dtype=torch.bool)
    w[0, :, :] = w[-1, :, :] = True
    w[:, 0, :] = w[:, -1, :] = True
    w[:, :, 0] = w[:, :, -1] = True
    return w


def interface_height(phi: torch.Tensor) -> float:
    """phi = 0.5 crossing along y on the tank centreline (linear interpolation)."""
    col = phi[phi.shape[0] // 2, :, phi.shape[2] // 2]
    for y in range(col.numel() - 1):
        a, b = float(col[y]), float(col[y + 1])
        if (a - 0.5) * (b - 0.5) <= 0.0 and a != b:
            return y + (0.5 - a) / (b - a)
    return float("nan")


def hydrostatic_slope(p: torch.Tensor, phi: torch.Tensor, band: float = 0.9) -> float:
    """Least-squares dp/dy over the water bulk (phi > band), tank centreline."""
    col_p = p[p.shape[0] // 2, :, p.shape[2] // 2].double()
    col_phi = phi[p.shape[0] // 2, :, p.shape[2] // 2].double()
    ys = torch.arange(col_p.numel(), dtype=torch.float64)[col_phi > band]
    ps = col_p[col_phi > band]
    if ys.numel() < 3:
        return float("nan")
    y0 = ys.mean()
    p0 = ps.mean()
    return float(((ys - y0) * (ps - p0)).sum() / ((ys - y0) ** 2).sum())


def run_arm(
    *,
    rho_l: float,
    rho_g: float,
    nu: float,
    g: float,
    sigma: float,
    width: float,
    mobility: float,
    n_steps: int,
    sample_every: int,
    shape: tuple[int, int, int],
    label: str,
) -> dict:
    wall = wall_mask(shape)
    sim = m.CACSim(
        shape,
        rho_l=rho_l,
        rho_g=rho_g,
        mu_l=nu * rho_l,
        mu_g=nu * rho_g,
        sigma=sigma,
        width=width,
        mobility=mobility,
        p0=0.01,
        gravity=(0.0, -g, 0.0),
        gravity_mode="rho",
        axes="www",
        wall_mask=wall,
    )
    y0 = shape[1] / 2.0
    sim.initialize(m.init_phi_layers(shape, width=width, y0=y0))

    phi0 = float(sim.f.sum(dim=0)[~wall].double().sum())
    h0 = interface_height(sim.f.sum(dim=0))

    t0 = time.time()
    samples = []
    nan_seen = False
    for step in range(1, n_steps + 1):
        sim.step()
        if step % sample_every == 0 or step == n_steps:
            diag = m.diagnostics(sim)
            phi_bulk, rho_bulk, grad, mu_phi, F, u, p, u_star = m._macro_fp64(sim)
            h = interface_height(phi_bulk)
            dpdy = hydrostatic_slope(p, phi_bulk)
            nan_seen = nan_seen or bool(diag["nan_f"] or diag["nan_g"])
            samples.append(
                {
                    "step": step,
                    "interface_y": h,
                    "interface_drift": h - h0,
                    "dp_dy": dpdy,
                    "dp_dy_over_rho_g": dpdy / (rho_l * g),
                    "u_max": diag["u_max_fluid"],
                    "phi_integral_drift_rel": (diag["phi_integral_fluid"] - phi0) / phi0,
                    "phi_min": diag["phi_min_fluid"],
                    "phi_max": diag["phi_max_fluid"],
                }
            )

    return {
        "label": label,
        "shape": list(shape),
        "rho_l": rho_l,
        "rho_g": rho_g,
        "density_ratio": rho_l / rho_g,
        "nu": nu,
        "g": g,
        "sigma": sigma,
        "width": width,
        "mobility": mobility,
        "n_steps": n_steps,
        "interface_y_initial": h0,
        "nan_seen": nan_seen,
        "wall_time_s": time.time() - t0,
        "samples": samples,
        "final": samples[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/p0_hydrostatic.json"))
    ap.add_argument("--n-steps", type=int, default=4000)
    ap.add_argument("--sample-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    shape = (8, 96, 8)
    # nu_sigma from the verified Laplace archive (W=4, sigma=0.01, mobility 0.05)
    common = dict(
        nu=0.1,
        sigma=0.01,
        width=4.0,
        mobility=0.05,
        n_steps=args.n_steps,
        sample_every=args.sample_every,
        shape=shape,
    )
    arms = [
        dict(label="rho100_g1e-5", rho_l=1.0, rho_g=0.01, g=1e-5),
        dict(label="rho100_g1e-4", rho_l=1.0, rho_g=0.01, g=1e-4),
        dict(label="rho100_g1e-3", rho_l=1.0, rho_g=0.01, g=1e-3),
        dict(label="rho816_g1e-4", rho_l=1.0, rho_g=1.0 / 816.0, g=1e-4),
    ]

    out = {"probe": "p0_hydrostatic", "model": "cac_lbm.CACSim", "arms": []}
    for arm in arms:
        print(f"[p0] running {arm['label']} ...", flush=True)
        res = run_arm(**common, **arm)
        f = res["final"]
        print(
            f"[p0]   interface drift {f['interface_drift']:+.4e} cells | "
            f"dp/dy/(rho g) {f['dp_dy_over_rho_g']:.4f} | max|u| {f['u_max']:.3e} | "
            f"phi drift {f['phi_integral_drift_rel']:+.3e} | nan={res['nan_seen']} | "
            f"{res['wall_time_s']:.1f}s",
            flush=True,
        )
        out["arms"].append(res)

    out["notes"] = (
        "Prerequisite probe only. Hydrostatic equilibrium of the undeformed tank: "
        "flat interface + dp/dy = -rho_l g + bounded spurious current. "
        "No external reference data are compared; this is not a water-entry benchmark."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"[p0] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
