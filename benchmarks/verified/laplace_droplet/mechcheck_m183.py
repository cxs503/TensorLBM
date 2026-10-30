#!/usr/bin/env python
"""HONEST-NEGATIVE cross-check: naive one-sided SC pressure tensor vs Laplace sigma.

Motivation: the case measures sigma through the Young-Laplace law (Delta p vs R).
An independent route would be the *mechanical* (pressure-tensor) route

    sigma = \\int_0^inf [ P_rr(r) - P_tt(r) ] dr ,
    P_ab  = rho c_s^2 delta_ab + (G_eff/2) sum_i w_i  <psi psi>' e_ia e_ib ,

which is divergence-consistent only in the continuum limit.  Three discrete
variants of the psi psi' product are tried on the *stored steady fields*:

    v1  psi(x) psi(x+e_i)              (one-sided, "textbook" SC tensor)
    v2  psi_bar^2, psi_bar=(psi+psi')/2 (symmetrised)
    v3  psi(x+e_i)^2                    (one-sided, shifted)

All of them are known to suffer D2Q9 fourth-moment anisotropy across a diffuse
interface (W ~ 4 lu here).  This script records the outcome **whatever it is**;
it is a post-processing diagnostic, it does not modify the library or the case.
Output: artifacts/m183_laplace_droplet/mechroute_variants.json
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ART = REPO / "artifacts" / "m183_laplace_droplet"
RUNS = REPO / "runs" / "m183_laplace"


def load_case():
    spec = importlib.util.spec_from_file_location("laplace_bench_case", HERE / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASE = load_case()
G = CASE.G_EFF
EX = torch.tensor([0, 1, -1, 0, 0, 1, -1, 1, -1], dtype=torch.float32)
EY = torch.tensor([0, 0, 0, 1, -1, 1, 1, -1, -1], dtype=torch.float32)
W = torch.tensor([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                 dtype=torch.float32)


def sigma_mech(rho: torch.Tensor, variant: str, dr: float = 1.0) -> float:
    L = rho.shape[0]
    psi = CASE.psi_exp(rho.clamp(min=1e-8))
    Pxx, Pyy, Pxy = rho / 3.0, rho / 3.0, torch.zeros_like(rho)
    for i in range(9):
        ps = torch.roll(torch.roll(psi, -int(EX[i]), dims=1), -int(EY[i]), dims=0)
        if variant == "v1":
            coef = 0.5 * G * W[i] * psi * ps
        elif variant == "v2":
            pm = 0.5 * (psi + ps)
            coef = 0.5 * G * W[i] * pm * pm
        elif variant == "v3":
            coef = 0.5 * G * W[i] * ps * ps
        else:
            raise ValueError(variant)
        Pxx = Pxx + coef * EX[i] ** 2
        Pyy = Pyy + coef * EY[i] ** 2
        Pxy = Pxy + coef * EX[i] * EY[i]
    ys = torch.arange(L, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, ys, indexing="ij")
    dx, dy = xx - L / 2.0, yy - L / 2.0
    r = torch.sqrt(dx**2 + dy**2).clamp(min=1e-6)
    nx, ny = dx / r, dy / r
    Prr = Pxx * nx**2 + Pyy * ny**2 + 2.0 * Pxy * nx * ny
    Ptt = Pxx * ny**2 + Pyy * nx**2 - 2.0 * Pxy * nx * ny
    dP = Prr - Ptt
    edges = torch.arange(0.0, L / 2.0 + dr, dr)
    idx = torch.clamp((r / dr).long(), max=len(edges) - 2)
    nb = len(edges) - 1
    s = torch.zeros(nb, dtype=torch.float64).index_add_(0, idx.reshape(-1), dP.double().reshape(-1))
    c = torch.zeros(nb, dtype=torch.float64).index_add_(
        0, idx.reshape(-1), torch.ones(idx.numel(), dtype=torch.float64))
    return float((s / c.clamp(min=1)).sum().item() * dr)


def main() -> None:
    res = json.loads((ART / "sweep_result.json").read_text(encoding="utf-8"))
    lap = {float(p["R_init"]): p for p in res["dims"]["2d"]["per_radius"]}
    rows = []
    for R in sorted(lap):
        snap = RUNS / f"2d_R{int(R)}" / "snapshot.npy"
        if not snap.exists():
            continue
        rho = torch.from_numpy(np.load(snap))
        sigma_lap = lap[R]["dp"] * lap[R]["R_eq"]
        row = {"R_init": R, "R_eq": lap[R]["R_eq"], "sigma_laplace": sigma_lap}
        for v in ("v1", "v2", "v3"):
            s = sigma_mech(rho, v)
            row[v] = s
            row[f"{v}_rel_diff_pct"] = abs(s - sigma_lap) / sigma_lap * 100.0
        rows.append(row)
        print(row)
    payload = {
        "what": "naive discrete SC pressure-tensor route for sigma vs Laplace sigma",
        "formula": "sigma = int (P_rr - P_tt) dr, P_ab = rho c_s^2 d_ab + (G_eff/2) sum_i w_i (psi psi') e_ia e_ib",
        "G_eff": G,
        "variants": {"v1": "psi(x)psi(x+e_i)", "v2": "[(psi(x)+psi(x+e_i))/2]^2",
                     "v3": "psi(x+e_i)^2"},
        "rows": rows,
        "verdict": "FAIL - none of the three naive tensor variants reproduces the Laplace sigma "
                   "(46-126% deviation); reported as an honest negative cross-check, NOT used in "
                   "the accuracy verdict. Cause: the one-sided/naive SC pressure tensor is not "
                   "divergence-consistent on D2Q9 (fourth-moment anisotropy across a W~4 interface); "
                   "a divergence-consistent (gradient-corrected) tensor would be needed, which is "
                   "out of scope here (library must not be modified).",
    }
    (ART / "mechroute_variants.json").write_text(json.dumps(payload, indent=2) + "\n",
                                                  encoding="utf-8")
    print("saved ->", ART / "mechroute_variants.json")


if __name__ == "__main__":
    main()