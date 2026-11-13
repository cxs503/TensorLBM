"""CASE-MCMP-LAPLACE round 3 — closing virial re-adjudication of Route B.

Round 2 left the absolute-σ anchor VOID: the Shan(2006) two-point tensor
(``pairing="sym"``) does not conserve the normal stress of a static flat
interface under this library's force format (reverse gather + velocity shift),
measured flatness 46–64% (gate ≤5%).  Round 3 adds the empirically calibrated
closing tensor (``pairing="closed"``, see
``tensorlbm.scmc_pressure_tensor``) and re-measures, on the same slab protocol
as round 2 (tanh liquid slab, two periodic interfaces, coexistence densities
from the curvature-free planar run):

* ``flatness_sym`` / ``flatness_closed``  — normal-stress conservation residual;
* ``sigma_mech_sym`` / ``sigma_mech_closed`` — Kirkwood–Buff σ per interface;
* ``sigma_eos`` — EOS-profile σ (Route C, independent of the droplet);
* for the hero droplet (G12=-2.75, R_init=16, n=96): ``dp_eos``, ``radius``,
  ``sigma_laplace = dp_eos * radius``.

The adjudication question: does any independent σ agree with the droplet's
Δp·R within the 5% case tolerance?  Everything is written to
``artifacts/round3_closed_virial.json`` (zero hand-typed numbers).

Usage (one call per job, control-plane 120 s budget):
    python round3_closed_virial.py slab -2.5
    python round3_closed_virial.py slab -2.75
    python round3_closed_virial.py slab -3.0
    python round3_closed_virial.py droplet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from tensorlbm.d2q9 import C, W, equilibrium, macroscopic
from tensorlbm.mcmp_verification import (
    droplet_radius,
    estimate_coexistence,
    init_static_droplet,
    measure_laplace,
    sc_mcmp_pressure,
    simulate_static_droplet,
)
from tensorlbm.multiphase import collide_sc_two_component
from tensorlbm.scmc_pressure_tensor import pressure_consistency_report
from tensorlbm.solver import stream

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts" / "round3_closed_virial.json"
G12_HERO = -2.75
TOL = 0.05


def _load() -> dict:
    if ART.exists():
        return json.loads(ART.read_text())
    return {"case": "CASE-MCMP-LAPLACE", "round": 3, "tol_pct": TOL, "slabs": {}, "droplet": None}


def _save(doc: dict) -> None:
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.write_text(json.dumps(doc, indent=2))


def job_slab(G: float) -> None:
    torch.set_num_threads(1)
    t0 = time.time()
    coex = estimate_coexistence(G_12=G, n=96, steps=3000)
    r1l, r1g = coex["rho1_liq"], coex["rho1_gas"]
    r2l, r2g = coex["rho2_liq"], coex["rho2_gas"]
    NY, NX, HALF, W0, STEPS = 96, 8, 24.0, 2.0, 4000
    ys = torch.arange(NY, dtype=torch.float32).view(NY, 1)
    c = NY / 2.0
    s_in = (
        0.5 * (1.0 + torch.tanh((ys - (c - HALF)) / W0))
        * 0.5 * (1.0 - torch.tanh((ys - (c + HALF)) / W0))
    ).expand(NY, NX).clone()
    rho1 = r1g + s_in * (r1l - r1g)
    rho2 = r2g + s_in * (r2l - r2g)
    z = torch.zeros_like(rho1)
    f1 = equilibrium(rho1, z, z)
    f2 = equilibrium(rho2, z, z)
    umax_hist = []
    for step in range(1, STEPS + 1):
        f1, f2 = collide_sc_two_component(f1, f2, G_12=G, tau1=1.0, tau2=1.0)
        f1 = stream(f1)
        f2 = stream(f2)
        if step % 500 == 0:
            rho = (f1.sum(0) + f2.sum(0)).clamp(min=1e-12)
            cx = C[:, 0].view(9, 1, 1).float()
            cy = C[:, 1].view(9, 1, 1).float()
            ux = (f1 * cx).sum(0) + (f2 * cx).sum(0)
            uy = (f1 * cy).sum(0) + (f2 * cy).sum(0)
            umax_hist.append(float(((ux / rho) ** 2 + (uy / rho) ** 2).sqrt().max()))
    rep = pressure_consistency_report(f1, f2, G, n_interfaces=2)
    doc = _load()
    doc["slabs"][f"{G}"] = {
        "G_12": G,
        "ny": NY,
        "nx": NX,
        "slab_half": HALF,
        "w0": W0,
        "steps": STEPS,
        "coex": {k: coex[k] for k in ("rho1_liq", "rho1_gas", "rho2_liq", "rho2_gas", "flat_interface_dp")},
        "u_max": max(umax_hist[-2:]),
        **rep,
        "elapsed_s": round(time.time() - t0, 1),
    }
    _save(doc)
    print(json.dumps(doc["slabs"][f"{G}"], indent=2))


def job_droplet() -> None:
    torch.set_num_threads(1)
    t0 = time.time()
    G = G12_HERO
    run = simulate_static_droplet(G_12=G, n=96, radius_init=16.0, steps=12000)
    case = measure_laplace(run, dims=2)
    doc = _load()
    doc["droplet"] = {
        "G_12": G,
        "n": 96,
        "radius_init": 16.0,
        "radius": case["radius"],
        "dp_eos": case["dp"],
        "sigma_laplace": case["sigma_fit"] if case.get("sigma_fit") else case["dp"] * case["radius"],
        "sigma_fit_origin": case.get("sigma_fit"),
        "max_rel_err_radius": case.get("max_rel_err"),
        "u_max": run["u_max"],
        "steps_run": run["steps_run"],
        "converged": run["converged"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    # cross-route ratios with the hero slab (same G12)
    slab = doc["slabs"].get(f"{G}")
    if slab and doc["droplet"]["sigma_laplace"]:
        sl = doc["droplet"]["sigma_laplace"]
        doc["droplet"]["ratio_mech_closed_to_laplace"] = slab["sigma_mech_closed"] / sl
        doc["droplet"]["ratio_eos_to_laplace"] = slab["sigma_eos"] / sl
    _save(doc)
    print(json.dumps(doc["droplet"], indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    kind = sys.argv[1]
    if kind == "slab":
        if len(sys.argv) < 3:
            raise SystemExit("slab job needs a G12 argument")
        job_slab(float(sys.argv[2]))
    elif kind == "droplet":
        job_droplet()
    else:
        raise SystemExit(f"unknown job {kind!r}")
