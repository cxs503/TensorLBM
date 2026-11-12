#!/usr/bin/env python3
"""CASE-MCMP-LAPLACE round 2 — an *independent* surface tension for the
Shan-Chen two-component (SCMC) model, so that Young-Laplace is not tested
against a sigma fitted from the very same pressure jumps.

Motivation (defect of the round-1 executor)
-------------------------------------------
``run_case.py`` fits, for each coupling ``G12``, the single unknown of
``dp = sigma / R`` through the origin on the *measured* ``dp`` and then reports
``|dp_i - sigma_fit / R_i| / (sigma_fit / R_i)``.  With four radii and one free
parameter this leaves three residual degrees of freedom and does test that
``dp * R`` is radius independent -- but the *absolute* value of sigma is an
output of the same data, so the number is a consistency (linearity) test, not
an absolute-value test.  Any common multiplicative error of the measured
``dp * R`` would pass unchanged.

This module supplies the missing absolute anchor from a *separate* simulation
of a curvature-free flat interface, where sigma is the Kirkwood-Buff integral
of the mechanical stress deficit,

    sigma = integral ( P_NN - P_TT ) dn            (Landau & Lifshitz VI, §156)

taken over one interface, with the discrete SCMC pressure tensor

    P_ab = cs^2 (rho1+rho2) delta_ab
           + (G12/2) sum_i w_i [ rho1(x) rho2(x+c_i) + rho2(x) rho1(x+c_i) ] c_ia c_ib

(Shan 2008, Phys. Rev. E 77, 066702, eq. 12 in the two-point form; Sbragaglia
et al. 2007, Phys. Rev. E 75, 026702, for the discrete counterpart).  The
pairing direction (``fwd``/``bwd``/``sym``) is *not* assumed: all three are
measured and discriminated by the physical validity gate that a flat static
interface must satisfy, ``P_NN(y) = const`` (normal force balance), plus the
sign requirement ``sigma > 0``.  The isotropic EOS pressure
``cs^2(rho1+rho2) + G12 cs^2 rho1 rho2`` is included as the negative control:
it carries no anisotropic stress and therefore must give ``sigma = 0``; if it
did not, the gate would be blind.

The absolute anchor is then used to re-judge ``dp = sigma / R`` on static
droplets whose ``dp`` and ``R`` come from :mod:`tensorlbm.mcmp_verification`
(``sc_mcmp_pressure`` in a 4x4 box at the centre and at a corner; radius from
the ``rho_av`` crossing on four rays).

Usage::

    PYTHONPATH=src python3 benchmarks/case_pool/mcmp_laplace/independent_sigma.py \
        --out runs/m15952_laplace/independent_sigma.json [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.mcmp_verification import (
    estimate_coexistence,
    measure_laplace,
    simulate_planar_slab,
    simulate_static_droplet,
)
from tensorlbm.scmc_pressure_tensor import scmc_pressure_tensor

PAIRINGS = ("fwd", "bwd", "sym", "isotropic")


# ---------------------------------------------------------------------------
# Flat interface: independent sigma
# ---------------------------------------------------------------------------


def _profile(field: torch.Tensor) -> torch.Tensor:
    """x-average of a ``(ny, nx)`` field (the interface is uniform in x)."""
    return field.mean(dim=1)


def flat_sigma(
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    G12: float,
    pairing: str,
    *,
    n_interfaces: int = 2,
) -> dict[str, float]:
    """sigma = (1/n_int) * integral (P_NN - P_TT) dy of a flat static interface.

    The fields are the relaxed densities of a periodic slab with the interface
    normal along ``y``; velocities are set to zero when rebuilding the
    distributions because the pressure tensor used here depends on density
    only (``macroscopic(equilibrium(rho,0,0)) == (rho,0,0)`` exactly).

    Validity gates evaluated on the same profile (a flat static interface has
    no tangential stress source, so ``P_NN`` must be uniform):

    * ``flatness`` = ``max_y |P_NN(y) - <P_NN>| / |<P_NN>|``;
    * ``sigma_pos`` = ``sigma > 0`` (tension, not compression);
    * ``bulk_dp`` = ``p_iso(liq) - p_iso(gas)`` from the isotropic EOS
      pressure (must vanish for a flat interface at coexistence).
    """
    zero = torch.zeros_like(rho1)
    f1 = equilibrium(rho1, zero, zero)
    f2 = equilibrium(rho2, zero, zero)
    Pxx, Pyy, _ = scmc_pressure_tensor(f1, f2, G12, pairing)
    p_nn = _profile(Pyy)
    p_tt = _profile(Pxx)
    mean_nn = float(p_nn.mean())
    flatness = float((p_nn - mean_nn).abs().max()) / max(abs(mean_nn), 1e-30)
    sigma = float((p_nn - p_tt).sum()) / n_interfaces

    # isotropic EOS pressure in the two bulks (first/last rows are bulk: the
    # slab is centred and at least ~8 interface widths away from the boundary)
    p_iso = (1.0 / 3.0) * (rho1 + rho2) + G12 * (1.0 / 3.0) * rho1 * rho2
    p_iso_prof = _profile(p_iso)
    n = p_iso_prof.numel()
    k = max(4, n // 16)
    p_liq = float(p_iso_prof[:k].mean())
    p_gas = float(p_iso_prof[-k:].mean())
    return {
        "sigma": sigma,
        "flatness": flatness,
        "mean_normal": mean_nn,
        "p_iso_liq": p_liq,
        "p_iso_gas": p_gas,
        "bulk_dp": p_liq - p_gas,
        "sigma_pos": sigma > 0,
        "p_nn_min": float(p_nn.min()),
        "p_nn_max": float(p_nn.max()),
    }


def coexistence_for(G12: float, n: int, steps: int) -> dict[str, float]:
    """Planar coexistence densities (no Kelvin shift) via the library route."""
    co = estimate_coexistence(G_12=G12, n=n, geometry="planar", steps=steps)
    return {
        "rho1_in": float(co["rho1_liq"]),
        "rho2_in": float(co["rho2_liq"]),
        "rho1_out": float(co["rho1_gas"]),
        "rho2_out": float(co["rho2_gas"]),
        "flat_interface_dp": float(co.get("flat_interface_dp", float("nan"))),
    }


def flat_interface_study(
    G12: float,
    *,
    n: int,
    co: dict[str, float],
    steps: int,
    half_thickness: float,
) -> dict[str, object]:
    """Relax a flat slab at one resolution and score every pairing convention."""
    slab = simulate_planar_slab(
        G_12=G12,
        n=n,
        rho1_in=co["rho1_in"],
        rho2_in=co["rho2_in"],
        rho1_out=co["rho1_out"],
        rho2_out=co["rho2_out"],
        half_thickness=half_thickness,
        steps=steps,
    )
    out: dict[str, object] = {
        "G12": G12,
        "n": n,
        "converged": bool(slab["converged"]),
        "steps_run": int(slab["steps_run"]),
        "u_max": float(slab["u_max"]),
        "pairings": {},
    }
    for pairing in PAIRINGS:
        out["pairings"][pairing] = flat_sigma(
            slab["rho1"], slab["rho2"], G12, pairing
        )
    return out


# ---------------------------------------------------------------------------
# Droplet sweep: dp and R
# ---------------------------------------------------------------------------


def droplet_point(
    G12: float,
    *,
    n: int,
    radius_init: float,
    co: dict[str, float],
    steps: int,
) -> dict[str, float]:
    run = simulate_static_droplet(
        G_12=G12,
        n=n,
        radius_init=radius_init,
        rho1_in=co["rho1_in"],
        rho2_in=co["rho2_in"],
        rho1_out=co["rho1_out"],
        rho2_out=co["rho2_out"],
        steps=steps,
    )
    m = measure_laplace(run, box=4, dims=2)
    return {
        "radius_init": radius_init,
        "n": n,
        "radius": float(m["radius"]),
        "radius_spread": float(m["radius_spread"]),
        "delta_p": float(m["dp"]),
        "sigma_laplace": float(m["sigma"]),
        "u_max": float(run["u_max"]),
        "steps_run": int(run["steps_run"]),
        "converged": bool(run["converged"]),
        "guardrails": m["guardrails"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--G", type=float, nargs="+", default=[-2.5, -3.0])
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(1)
    t0 = time.time()
    G_list = args.G
    co_n = 96
    co_steps = 12000 if not args.quick else 6000
    flat_steps = 20000 if not args.quick else 8000
    drop_steps = 12000 if not args.quick else 4000
    radii = [12.0, 16.0, 20.0, 24.0]

    result: dict[str, object] = {
        "case": "CASE-MCMP-LAPLACE",
        "round": 2,
        "purpose": "absolute surface tension from a flat interface (Kirkwood-Buff) "
                   "used as an anchor independent of the droplet dp",
        "tau": 1.0,
        "dim": 2,
        "levels": [],
        "stray": {},
    }

    for G in G_list:
        co = coexistence_for(G, co_n, co_steps)
        level: dict[str, object] = {"G12": G, "coexistence": co, "flat": {}, "droplets": []}
        for n in (96, 128):
            half = 0.25 * n  # slab half thickness ~ n/4 -> two well separated interfaces
            level["flat"][str(n)] = flat_interface_study(
                G, n=n, co=co, steps=flat_steps, half_thickness=half
            )
            print(f"[{time.time()-t0:6.1f}s] flat G={G} n={n} done", flush=True)

        for n in (96, 128):
            for r in (radii if n == 96 else (16.0,)):
                if r > 0.25 * n:
                    continue
                pt = droplet_point(G, n=n, radius_init=r, co=co, steps=drop_steps)
                level["droplets"].append(pt)
                print(
                    f"[{time.time()-t0:6.1f}s] droplet G={G} n={n} r0={r} "
                    f"R={pt['radius']:.2f} dp={pt['delta_p']:.6f}",
                    flush=True,
                )
        result["levels"].append(level)
        if args.out is not None:  # incremental dump: partial data survives a kill
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=1, default=float))

    result["wall_seconds"] = time.time() - t0
    out = args.out
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, default=float))
        print(f"wrote {out}")
    print(json.dumps(_summary(result), indent=1, default=float))


def _summary(result: dict) -> dict:
    """Compact view: for each level, the decisive sigma comparison numbers."""
    rows = []
    for level in result["levels"]:
        G = level["G12"]
        flat96 = level["flat"]["96"]["pairings"]
        flat128 = level["flat"]["128"]["pairings"]
        # pick the pairing that satisfies the normal-force gate best at n=128
        best = min(
            (p for p in PAIRINGS if p != "isotropic"),
            key=lambda p: flat128[p]["flatness"],
        )
        sig_mech = {n: level["flat"][n]["pairings"][best]["sigma"] for n in ("96", "128")}
        for d in level["droplets"]:
            if d["n"] != 128:
                continue
            s = sig_mech["128"]
            ref = s / d["radius"]
            rows.append(
                {
                    "G12": G,
                    "pairing": best,
                    "R": d["radius"],
                    "dp_got": d["delta_p"],
                    "dp_ref": ref,
                    "rel_err_pct": 100.0 * (d["delta_p"] - ref) / ref,
                    "sigma_mech": s,
                    "sigma_laplace": d["sigma_laplace"],
                    "sigma_rel_diff_pct": 100.0
                    * (d["sigma_laplace"] - s)
                    / s,  # sign of 1/R convention: dp*R = sigma
                    "sigma_mech_96": sig_mech["96"],
                }
            )
        level["_best_pairing"] = best
        level["_flat_gate"] = {
            p: {
                "flatness_96": flat96[p]["flatness"],
                "flatness_128": flat128[p]["flatness"],
                "sigma_96": flat96[p]["sigma"],
                "sigma_128": flat128[p]["sigma"],
            }
            for p in PAIRINGS
        }
    return {"rows": rows, **_clean_levels(result)}


def _clean_levels(result: dict) -> dict:
    return {
        "gates": {
            f"G{lv['G12']}": lv.get("_flat_gate", {}) for lv in result["levels"]
        },
        "best_pairing": {f"G{lv['G12']}": lv.get("_best_pairing") for lv in result["levels"]},
    }


if __name__ == "__main__":
    main()