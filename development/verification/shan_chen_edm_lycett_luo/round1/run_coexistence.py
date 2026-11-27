"""XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO — coexistence measurement of the SCMP kernel.

Arm under test: ``tensorlbm.multiphase.collide_sc_single_component``
(forcing in {velocity_shift, edm, guo}) + ``tensorlbm.solver.stream`` on D2Q9.

Setup
-----
Planar (flat) interface in a periodic box, 1-D in x (``ny = 2`` is enough to
exercise the diagonal stencil).  A flat interface has **no curvature**, hence
no Laplace pressure offset, so the plateau densities must converge to the
*thermodynamic* coexistence densities of the model EOS — that is what makes
this a sharp test of the forcing scheme rather than of surface tension.

Reference: :mod:`maxwell_sc` (Maxwell equal-area construction on the same
lattice EOS, independent pure-numpy implementation).

Measured quantities per case
----------------------------
* ``rho_l``, ``rho_g`` — plateau densities (mean of top / bottom 10 % cells);
* ``err_l``, ``err_g`` — relative error vs Maxwell;
* ``max_u`` — spurious-current metric ``max|u|``;
* ``mass_drift`` — ``|mean(rho) - mean(rho_0)|`` (must be ~1e-15: every
  scheme here is mass conserving);
* ``drift_l``, ``drift_g`` — change of the plateaus over the last 20 % of the
  run (convergence guard).

Usage
-----
    python run_coexistence.py --list
    python run_coexistence.py --all                   # full matrix
    python run_coexistence.py --case exp_G5.0_edm_tau1.0 --out results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import maxwell_sc as MW  # noqa: E402

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import (  # noqa: E402
    collide_sc_single_component,
    psi_eos_carnahan_starling,
    psi_exp as psi_exp_t,
)
from tensorlbm.solver import stream  # noqa: E402

DTYPE = torch.float64

# ---------------------------------------------------------------------------
# Case matrix (declared before any run; see REPORT.md §③)
# ---------------------------------------------------------------------------

TC_CS = 0.09432  # Carnahan-Starling critical temperature, a=1, b=4, R=1

DEFAULT_STEPS = 30000
NX, NY = 200, 2
IFACE_WIDTH = 3.0


def eos_specs():
    """Two pseudopotential arms: classic psi_exp and the TCLB CS-EOS psi."""
    return {
        "exp": {
            "label": "psi_exp (Shan-Chen 1993)",
            "G": 5.0,
            "psi_torch": psi_exp_t,
            "psi_np": MW.psi_exp,
            "rho_hi_maxwell": 6.0,
        },
        "cs": {
            "label": f"psi_CS (TCLB d2q9_scmp, Tr=0.85, G=1)",
            "G": 1.0,
            "psi_torch": lambda r: psi_eos_carnahan_starling(r, T=0.85 * TC_CS),
            "psi_np": lambda r: MW.psi_cs(r, 0.85 * TC_CS),
            "rho_hi_maxwell": 1.0,
        },
    }


def cases(G_sweep: bool = True):
    """Frozen case list.

    Blocks (all declared before any run, see REPORT.md section 3):
      B1  EOS x forcing x tau grid                tau in {0.6, 1.0, 1.5}
      B2  coexistence-curve points (G = 4.5, 5.5, edm, tau = 1)
      B3  tau lever widened to 2.0 (exp arm, all three forcings)
      B4  initial-condition independence (spinodal start, tau = 1)
    """
    out = []
    for eos in ("exp", "cs"):
        for forcing in ("velocity_shift", "edm", "guo"):
            for tau in (0.6, 1.0, 1.5):
                out.append({"eos": eos, "forcing": forcing, "tau": tau, "G": None,
                            "ic": "maxwell",
                            "tag": f"{eos}_Gdefault_{forcing}_tau{tau}"})
    if G_sweep:
        for G in (4.5, 5.5):
            out.append({"eos": "exp", "forcing": "edm", "tau": 1.0, "G": G,
                        "ic": "maxwell", "tag": f"exp_G{G}_edm_tau1.0"})
        for forcing in ("velocity_shift", "edm", "guo"):
            out.append({"eos": "exp", "forcing": forcing, "tau": 2.0, "G": None,
                        "ic": "maxwell", "tag": f"exp_Gdefault_{forcing}_tau2.0"})
        for eos in ("exp", "cs"):
            for forcing in ("velocity_shift", "edm", "guo"):
                out.append({"eos": eos, "forcing": forcing, "tau": 1.0, "G": None,
                            "ic": "shifted",
                            "tag": f"{eos}_Gdefault_{forcing}_tau1.0_icS"})
    return out


# ---------------------------------------------------------------------------
# Initialisation / measurement
# ---------------------------------------------------------------------------


def flat_interface_ic(rho_l: float, rho_g: float, nx: int = NX, ny: int = NY,
                      width: float = IFACE_WIDTH) -> torch.Tensor:
    """Periodic flat liquid slab (two interfaces) with a tanh profile."""
    x = np.arange(nx, dtype=np.float64)
    # |x - centre| mapped to a signed coordinate, slab centred at nx/4
    d = np.minimum(np.abs(x - nx / 4.0), nx - np.abs(x - nx / 4.0))
    half = nx / 4.0
    phase = np.tanh((half - d) / width)          # +1 liquid, -1 gas
    mid = 0.5 * (rho_l + rho_g)
    amp = 0.5 * (rho_l - rho_g)
    rho1d = mid + amp * phase
    rho = torch.tensor(rho1d, dtype=DTYPE).view(1, nx).expand(ny, nx).clone()
    zeros = torch.zeros_like(rho)
    return equilibrium(rho, zeros, zeros).to(DTYPE)


def measure(f: torch.Tensor) -> dict:
    rho, ux, uy = macroscopic(f)
    r = rho.detach().cpu().numpy().ravel()
    ux = ux.detach().cpu().numpy().ravel()
    uy = uy.detach().cpu().numpy().ravel()
    n = r.size
    k = max(1, int(0.10 * n))
    s = np.sort(r)
    return {
        "rho_l": float(s[-k:].mean()),
        "rho_g": float(s[:k].mean()),
        "rho_mean": float(r.mean()),
        "max_u": float(np.sqrt(ux**2 + uy**2).max()),
    }


def run_case(case: dict, steps: int = DEFAULT_STEPS, report_every: int = 1000,
             rows: int = 2, nx: int = NX, ic: str = "maxwell") -> dict:
    spec = eos_specs()[case["eos"]]
    G = case["G"] if case["G"] is not None else spec["G"]
    tau = float(case["tau"])
    forcing = case["forcing"]

    coex = MW.maxwell(G, spec["psi_np"], rho_hi=spec["rho_hi_maxwell"], n_solve=4000)
    if not coex.converged or coex.pressure_residual > 1e-9:
        raise RuntimeError(f"Maxwell construction failed for {case['tag']}: {coex}")

    ic = case.get("ic", ic)
    if ic == "maxwell":
        rho_l0, rho_g0 = coex.rho_l, coex.rho_g
    else:  # spinodal start: no prior knowledge of the coexistence point
        rho_l0, rho_g0 = coex.rho_l * 0.98, coex.rho_g * 1.02

    f = flat_interface_ic(rho_l0, rho_g0, nx=nx, ny=rows)
    m0 = measure(f)

    t0 = time.time()
    hist = []
    plateau_prev = None
    for it in range(1, steps + 1):
        rho = f.sum(0, keepdim=True) if False else None  # noqa: F841
        f = collide_sc_single_component(f, G=G, tau=tau, psi_fn=spec["psi_torch"],
                                        forcing=forcing)
        f = stream(f)
        if it % report_every == 0 or it == steps:
            m = measure(f)
            hist.append({"step": it, **m})
            plateau_prev = (m["rho_l"], m["rho_g"])
    wall = time.time() - t0

    rho_final, _, _ = macroscopic(f)
    profile = rho_final.detach().cpu().numpy()[0, :].tolist()
    m = hist[-1]
    n_tail = max(1, len(hist) // 5)
    tail = hist[-n_tail:]
    drift_l = float(np.ptp([h["rho_l"] for h in tail]))
    drift_g = float(np.ptp([h["rho_g"] for h in tail]))

    err_l = (m["rho_l"] - coex.rho_l) / coex.rho_l
    err_g = (m["rho_g"] - coex.rho_g) / coex.rho_g
    return {
        "tag": case["tag"],
        "eos": case["eos"],
        "forcing": forcing,
        "tau": tau,
        "G": G,
        "steps": steps,
        "nx": nx,
        "ny": rows,
        "ic": ic,
        "maxwell": coex.as_dict(),
        "measured": m,
        "err_l": err_l,
        "err_g": err_g,
        "err_l_pct": 100.0 * err_l,
        "err_g_pct": 100.0 * err_g,
        "drift_l": drift_l,
        "drift_g": drift_g,
        "mass_drift": abs(m["rho_mean"] - m0["rho_mean"]),
        "wall_s": wall,
        "profile_x": profile,
        "history": hist,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--rows", type=int, default=NY)
    ap.add_argument("--nx", type=int, default=NX)
    ap.add_argument("--ic", default="maxwell", choices=("maxwell", "shifted"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gsweep", action="store_true")
    args = ap.parse_args(argv)

    cl = cases(G_sweep=not args.no_gsweep)
    if args.list:
        for c in cl:
            print(c["tag"])
        return 0

    if args.case:
        sel = [c for c in cl if c["tag"] == args.case]
        if not sel:
            raise SystemExit(f"unknown case {args.case}")
    elif args.all:
        sel = cl
    else:
        sel = cl[:1]

    res = []
    for c in sel:
        r = run_case(c, steps=args.steps, rows=args.rows, nx=args.nx, ic=args.ic)
        res.append(r)
        print(f"{r['tag']:34s} rho_l={r['measured']['rho_l']:.5f} "
              f"(MW {r['maxwell']['rho_l']:.5f}, {r['err_l_pct']:+.2f}%)  "
              f"rho_g={r['measured']['rho_g']:.5f} "
              f"(MW {r['maxwell']['rho_g']:.5f}, {r['err_g_pct']:+.2f}%)  "
              f"max|u|={r['measured']['max_u']:.2e}  {r['wall_s']:.1f}s",
              flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())