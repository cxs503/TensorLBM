"""Static-droplet verification of the Swift free-energy model (PF5-SWFE).

Runs a static droplet (heavy phase in a light bath, no body force) on D2Q9 and
reports:
* spurious current  ``u_max``  (should be round-off for a static droplet);
* volume drift      ``|V - V0| / V0``;
* Laplace pressure  ``p_in - p_out``  vs the analytic ``sigma / R`` (2-D cylinder).

Density ratios 10, 100, 1000 are tested.  A high density ratio that is unstable
is a *valid* (honest-FAIL) conclusion for the matrix-completion arm.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.swift_free_energy import (  # noqa: E402
    SwiftParams,
    swift_init_2d,
    swift_step_2d,
)
from tensorlbm.phasefield.free_energy_pressure import grad_lap as grad_lap_2d  # noqa: E402

C2 = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]],
    dtype=torch.float64,
)


def _macro(f: torch.Tensor, g: torch.Tensor, p: SwiftParams):
    psi = g.sum(0)
    rho = p.density(psi)
    ux = (f * C2[:, 0:1].view(9, 1, 1)).sum(0) / rho.clamp(min=1e-12)
    uy = (f * C2[:, 1:2].view(9, 1, 1)).sum(0) / rho.clamp(min=1e-12)
    return rho, psi, ux, uy


def run_case(
    ratio: float,
    *,
    n: int = 64,
    R: float = 16.0,
    kappa: float = 0.02,
    xi: float = 3.0,
    tau_f: float = 1.0,
    tau_g: float = 1.0,
    steps: int = 300,
    rho_g: float = 1.0,
) -> dict:
    p = SwiftParams(
        kappa=kappa, xi=xi, tau_f=tau_f, tau_g=tau_g, rho_g=rho_g, rho_l=ratio * rho_g
    )
    x = torch.arange(n, dtype=torch.float64).view(1, -1)
    y = torch.arange(n, dtype=torch.float64).view(-1, 1)
    r = torch.sqrt((x - n / 2) ** 2 + (y - n / 2) ** 2)
    psi0 = torch.tanh((R - r) / xi)
    f, g = swift_init_2d(psi0, p)
    vol0 = float(((psi0 + 1) / 2).sum())
    umax_hist = []
    vol_hist = []
    stable = True
    nan_step = None
    for s in range(steps):
        f, g = swift_step_2d(f, g, p)
        rho, psi, ux, uy = _macro(f, g, p)
        umax = float((ux**2 + uy**2).sqrt().max())
        vol = float(((psi + 1) / 2).sum())
        umax_hist.append(umax)
        vol_hist.append(vol)
        if not (math.isfinite(umax) and math.isfinite(vol)):
            stable = False
            nan_step = s
            break
    # Laplace pressure: measure from the bulk (where the Korteweg term has decayed)
    rho, psi, ux, uy = _macro(f, g, p)
    _, _, lap = grad_lap_2d(psi)
    mu = p.chemical_potential(psi, lap)
    p_field = p.free_energy_pressure(rho, psi, mu)
    p_in = float(p_field[psi > 0.9].mean()) if (psi > 0.9).any() else float("nan")
    p_out = float(p_field[psi < -0.9].mean()) if (psi < -0.9).any() else float("nan")
    dp = p_in - p_out
    dp_analytic = p.sigma / R
    return {
        "ratio": ratio,
        "stable": stable,
        "nan_step": nan_step,
        "umax": max(umax_hist) if umax_hist else float("nan"),
        "umax_final": umax_hist[-1] if umax_hist else float("nan"),
        "vol_drift": abs(vol_hist[-1] / vol0 - 1) if vol_hist else float("nan"),
        "p_in": p_in,
        "p_out": p_out,
        "dp": dp,
        "dp_analytic": dp_analytic,
        "dp_rel_err": abs(dp - dp_analytic) / dp_analytic if dp_analytic else float("nan"),
        "sigma": p.sigma,
        "a": p.a,
    }


def main() -> None:
    results = []
    for ratio in (10.0, 100.0, 1000.0):
        res = run_case(ratio)
        results.append(res)
        print(
            f"ratio={ratio:6.0f}  stable={res['stable']}  umax={res['umax']:.3e}  "
            f"vol_drift={res['vol_drift']:.2e}  dp={res['dp']:.3e}  "
            f"dp_analytic={res['dp_analytic']:.3e}  rel_err={res['dp_rel_err']:.2%}"
        )
    out = ROOT / "benchmarks" / "water_entry" / "single_drop" / "swift_static_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
