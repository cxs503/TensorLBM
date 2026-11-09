"""Differentially heated square cavity (de Vahl Davis benchmark) — P97-N05 shard worker.

Left wall hot, right wall cold, top/bottom adiabatic, D2Q9 + D2Q5 double
distribution function LBM with Guo forcing (``tensorlbm.thermal``).  The
reference values are de Vahl Davis (1983) for ``Ra = 10^4``, ``Pr = 0.71``:
``Nu = 2.243``, ``u_max = 0.1926`` (velocity normalised by ``alpha / L``).

Note on ``u_max``: the lattice velocity of the steady solution scales as
``alpha / H`` and is therefore *not* a resolution-independent number; the
dimensionless quantity to compare with the benchmark is
``u_max * H / alpha`` and that is what this driver reports as ``u_max_star``.

Usage::

    python scripts/p97_n05_cavity_case.py --nx 128 --ra 1e4 --steps 400000 \
        --out runs/p97_n05/cavity_ra1e4_n128.json --fields-out ...npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.thermal import (  # noqa: E402
    apply_temperature_boundaries,
    buoyancy_force,
    cavity_wall_mask,
    collide_bgk_force,
    nusselt_number,
    pre_streaming_bounce_back,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
    thermal_params,
)

NU_REF_1E4 = 2.243  # de Vahl Davis (1983), Ra=1e4, Pr=0.71
UMAX_STAR_REF_1E4 = 0.1926


def simulate_cavity(
    nx: int,
    ra: float = 1e4,
    pr: float = 0.71,
    tau: float = 0.6,
    steps: int = 200000,
    sample_every: int = 2000,
    spinup_frac: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Natural-convection cavity, long enough for the diffusive transient."""
    device = torch.device(device)
    ny = nx
    p = thermal_params(nx, ra, pr, tau)
    # ``thermal_params`` uses H = nx, but the isothermal walls are pinned *on*
    # the first/last columns, so the physical gap of the thermal problem is
    # nx - 1 cells.  Re-form Ra with the true gap (otherwise the realised
    # Rayleigh number is (1 - 1/nx)^3 lower than requested, -4.6 % at nx = 64)
    # and measure Nu over the same gap.
    H = float(nx - 1)
    nu, alpha, tau_T = p["nu"], p["alpha"], p["tau_T"]
    g_beta = ra * nu * alpha / H**3
    t_hot, t_cold = 1.0, 0.0
    dT = t_hot - t_cold

    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)
    T0 = torch.linspace(t_cold, t_hot, nx, device=device).unsqueeze(0).expand(ny, nx).contiguous()
    g = temperature_equilibrium(T0, u0, u0)
    wall = cavity_wall_mask(ny, nx, device)
    interior = ~wall

    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        F = buoyancy_force(rho, T, g_beta, t_ref=t_cold)
        g = temperature_collision(g, tau_T, ux, uy)
        g = temperature_stream(g)
        g = apply_temperature_boundaries(g, t_hot, t_cold)
        f_pre = f
        f = collide_bgk_force(f, tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
        if step % sample_every == 0:
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            nu_g2 = nusselt_number(T, H, dT, t_hot, t_cold, mode="grad2")["nu"]
            u_star = float(torch.sqrt(ux**2 + uy**2).max()) * H / alpha
            hist.append({"step": step, "nu": nu_g2, "u_max_star": u_star})
    elapsed = time.time() - t0

    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    nu_g2 = nusselt_number(T, H, dT, t_hot, t_cold, mode="grad2")
    nu_g1 = nusselt_number(T, H, dT, t_hot, t_cold, mode="grad1")
    steps_arr = np.array([h["step"] for h in hist], dtype=float)
    tail = steps_arr >= steps * spinup_frac
    nu_tail = np.array([h["nu"] for h in hist], dtype=float)[tail]
    us_tail = np.array([h["u_max_star"] for h in hist], dtype=float)[tail]
    steps_np = steps_arr[tail]
    resid = (
        float(np.max(np.abs(np.diff(nu_tail))) / max(abs(nu_tail[-1]), 1e-30))
        if nu_tail.size > 1
        else float("nan")
    )
    return {
        "nx": nx,
        "ny": ny,
        "ra": ra,
        "pr": pr,
        "tau": tau,
        "tau_T": tau_T,
        "nu_lattice": nu,
        "alpha": alpha,
        "H": H,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "nu": float(nu_tail[-1]) if nu_tail.size else float(nu_g2["nu"]),
        "nu_tail_mean": float(nu_tail.mean()) if nu_tail.size else float("nan"),
        "nu_tail_drift": resid,
        "nu_grad2": nu_g2["nu"],
        "nu_grad1": nu_g1["nu"],
        "nu_left": nu_g2["nu_left"],
        "nu_right": nu_g2["nu_right"],
        "u_max_star": float(us_tail.mean()) if us_tail.size else float("nan"),
        "u_max_star_last": float(us_tail[-1]) if us_tail.size else float("nan"),
        "steps_settled_from": float(steps_np[0]) if steps_np.size else float("nan"),
        "nu_ref": NU_REF_1E4 if abs(ra - 1e4) < 1.0 else float("nan"),
        "rel_err_vs_ref": (nu_g2["nu"] - NU_REF_1E4) / NU_REF_1E4 if abs(ra - 1e4) < 1.0 else float("nan"),
        "history": hist[:: max(1, len(hist) // 200)],
        "T_field": T.detach().cpu().numpy().astype(np.float32),
        "umag_field": torch.sqrt(ux**2 + uy**2).detach().cpu().numpy().astype(np.float32),
        "interior_mask": interior.detach().cpu().numpy(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ra", type=float, default=1e4)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=200000)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--spinup-frac", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields-out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    res = simulate_cavity(
        nx=args.nx,
        ra=args.ra,
        pr=args.pr,
        tau=args.tau,
        steps=args.steps,
        sample_every=args.sample_every,
        spinup_frac=args.spinup_frac,
    )
    T = res.pop("T_field")
    umag = res.pop("umag_field")
    res.pop("interior_mask", None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    if args.fields_out:
        Path(args.fields_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.fields_out, T=T, umag=umag)
    print(
        f"nx={args.nx} Ra={args.ra:.3g} Nu={res['nu']:.4f} "
        f"(ref {NU_REF_1E4}, {100 * res['rel_err_vs_ref']:+.2f}%) "
        f"u*={res['u_max_star']:.4f} drift={res['nu_tail_drift']:.2e} [{res['elapsed_s']}s]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())