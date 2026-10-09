#!/usr/bin/env python3
"""Measure SCMP coexistence densities for a given pseudopotential + G coupling.

Runs a small 2D box from a random-density spinodal field and reports the
liquid/vapour density peaks (histogram mode of the two clusters) plus the
density ratio.  Used to re-measure coexistence when swapping the
pseudopotential form (e.g. psi_exp -> Carnahan-Starling), since the discrete
coexistence densities depend on both psi(.) and the SC coupling G.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import (  # noqa: E402
    collide_sc_single_component,
    psi_carnahan_starling,
    psi_exp,
    psi_peng_robinson,
    psi_power,
)
from tensorlbm.solver import stream  # noqa: E402


def psi_sqrt(rho):
    return torch.sqrt(torch.clamp(rho, min=0.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi", default="exp",
                    choices=["exp", "sqrt", "cs", "pr", "power"])
    ap.add_argument("--g-coupling", type=float, default=5.0, dest="G")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--rho-mean", type=float, default=0.30)
    ap.add_argument("--rho-amp", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--forcing", default="velocity_shift")
    args = ap.parse_args()

    psi_map = {
        "exp": psi_exp,
        "sqrt": psi_sqrt,
        "cs": psi_carnahan_starling,
        "pr": psi_peng_robinson,
        "power": psi_power,
    }
    psi_fn = psi_map[args.psi]

    torch.manual_seed(args.seed)
    dev = torch.device("cpu")
    n = args.n
    rho = args.rho_mean + args.rho_amp * (torch.rand((n, n), device=dev) - 0.5) * 2.0
    rho = rho.clamp(min=1e-3)
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)

    def step(f):
        f = collide_sc_single_component(
            f, G=args.G, tau=args.tau, psi_fn=psi_fn, gy=0.0,
            forcing=args.forcing,
        )
        f = stream(f)
        return f

    for s in range(1, args.steps + 1):
        f = step(f)
        if s % 500 == 0:
            rho = f.sum(0)
            if not torch.isfinite(rho).all():
                print(f"NAN/DIVERGED at step {s}")
                return
    rho = f.sum(0).flatten().numpy()
    # histogram -> two peaks
    hist, edges = np.histogram(rho, bins=120, range=(0.0, max(1.0, rho.max())))
    centers = 0.5 * (edges[:-1] + edges[1:])
    mid = 0.5 * (rho.min() + rho.max())
    lo = centers < mid
    hi = ~lo
    rho_v = centers[lo][np.argmax(hist[lo])]
    rho_l = centers[hi][np.argmax(hist[hi])]
    print(f"psi={args.psi} G={args.G} tau={args.tau} n={n} steps={args.steps}")
    print(f"  rho_min={rho.min():.4f} rho_max={rho.max():.4f} rho_mean={rho.mean():.4f}")
    print(f"  rho_v(peak)={rho_v:.4f}  rho_l(peak)={rho_l:.4f}  ratio={rho_l/max(rho_v,1e-6):.2f}")


if __name__ == "__main__":
    main()