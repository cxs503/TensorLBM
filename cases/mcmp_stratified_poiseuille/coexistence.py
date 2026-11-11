"""Round-4 coexistence calibration for GAP-LBM-MCMP-POISEUILLE.

Why this module exists
----------------------
Rounds 1-3 initialised the stratified channel with *nominal* layer densities
(``rho_heavy = 0.7``, ``rho_light = 0.3``).  For the linear-pseudopotential
two-component Shan-Chen model those are **not** the coexisting densities of the
chosen coupling ``G_12``: the layer that is denser than its coexistence value
keeps condensing, the diffuse interface keeps thinning below the grid scale and
the parasitic current stays at ``O(1e-2)`` lattice units no matter how the pair
force is discretised (R3 §3.3, fail_stage "model/state identification").

This module measures the actual state the model *wants*:

* a flat interface in a fully periodic strip (no walls, no body force),
* quiescent initial condition with a wide tanh seed (width 4 cells),
* run to a steady state,
* then read off the two coexisting densities ``rho_A`` (majority, the phase
  that hosts component 1 in the lower half) and ``rho_B`` (minority), the
  equilibrium interface thickness (10-90 convention, and the equivalent tanh
  width ``w = (y10 - y90) / (2 atanh 0.8)``), the interface drift and the
  steadiness of the densities.

The calibration is the classical coexistence measurement of a two-component
Shan-Chen model (Shan & Chen 1994; Martys & Chen 1996) and feeds the acceptance
matrix of round 4 (``r4_matrix.py``).

Method selection (pre-registered in ``docs/MCMP_STRATIFIED_POISEUILLE_R4.md``
§2, written before the first run of this file):

    admissible(G_12) :=  w_10_90 >= 3.0 cells            (diffuse interface resolved)
                      and density contrast >= 0.5        (stratification sustained)
                      and steady                         (drift < 0.5 cell)
    select           :=  argmax_{G_12 in admissible} |G_12|  (strongest segregation
                          among the *resolved* models)

The 3-cell floor is the standard resolution guidance for diffuse-interface
lattice-Boltzmann models (Kruger et al., *The Lattice Boltzmann Method* (2017)
§9.3; Sbragaglia et al., J. Fluid Mech. 628 (2009) 299, multirange
pseudopotential).  The rule is fixed on interface *resolution*, not on any
acceptance error, so it cannot be tuned to the criterion it is used to pass.

Usage::

    PYTHONPATH=src python cases/mcmp_stratified_poiseuille/coexistence.py \
        --out artifacts/mcmp_stratified_poiseuille/r4/coexistence_scan.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch  # noqa: E402

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_two_component  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

#: 10-90 thickness -> tanh width conversion for alpha = 0.5 (1 - tanh((y-y0)/w)).
TANH_10_90 = 2.0 * math.atanh(0.8)

#: Pre-registered admissibility thresholds (see module docstring).
W_MIN_CELLS = 3.0
CONTRAST_MIN = 0.5
DRIFT_MAX_CELLS = 0.5

#: Pre-registered coupling grid (round 4, fixed before running).
G12_GRID = (-1.5, -1.75, -2.0, -2.25, -2.5, -2.75)

TAU_LOWER = 1.0
TAU_UPPER = 0.75


def _seed(ny, nx, half, width):
    """Quiescent flat-interface seed, component 1 in the lower half."""
    ys = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    a = 0.5 * (1.0 - torch.tanh((ys - float(half)) / float(width)))
    # wide nominal seed; the model relaxes to its own coexistence state
    rho1 = 0.7 * a + 0.3 * (1.0 - a)
    rho2 = 0.3 * a + 0.7 * (1.0 - a)
    z = torch.zeros((ny, nx))
    return equilibrium(rho1, z, z), equilibrium(rho2, z, z)


def _crossing(profile, level=0.5):
    """Interpolated row where a monotone-decreasing profile crosses ``level``."""
    a = profile.tolist() if hasattr(profile, "tolist") else list(profile)
    for j in range(len(a) - 1):
        lo, hi = a[j], a[j + 1]
        if (lo - level) * (hi - level) <= 0 and lo != hi:
            return j + (lo - level) / (lo - hi)
    return float("nan")


def run_flat_interface(
    g12: float,
    ny: int = 96,
    nx: int = 4,
    steps: int = 40000,
    seed_width: float = 4.0,
    tau1: float = TAU_LOWER,
    tau2: float = TAU_UPPER,
    sc_form: str = "edm",
    sample_every: int = 2000,
) -> dict:
    """Run a fully periodic flat interface and return the coexistence state.

    Returns ``rho_A``/``rho_B`` (majority/minority density of the two
    components, symmetrised over the two bulk halves), the 10-90 interface
    thickness, the tanh width, interface drift over the last quarter of the run
    and the density steadiness.  No walls: the strip is periodic in *both*
    directions, so the measurement is free of wetting and of the wall
    calibration that the channel acceptance needs.
    """
    half = ny // 2
    f1, f2 = _seed(ny, nx, half, seed_width)
    z = torch.zeros((ny, nx))
    mass0 = (float(macroscopic(f1)[0].sum()), float(macroscopic(f2)[0].sum()))

    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        f1, f2 = collide_sc_two_component(
            f1, f2, G_12=g12, tau1=tau1, tau2=tau2, gx=0.0,
            u_eq="viscous", force_scheme="guo", sc_form=sc_form,
        )
        f1, f2 = stream(f1), stream(f2)
        if step % sample_every and step != steps:
            continue
        r1, ux1, uy1 = macroscopic(f1)
        r2, ux2, uy2 = macroscopic(f2)
        rho_tot = torch.clamp(r1 + r2, min=1e-12)
        uy = (r1 * uy1 + r2 * uy2) / rho_tot
        alpha = ((r1 - r2) / torch.clamp(r1 + r2, min=1e-12)).mean(dim=1)
        y0 = _crossing(alpha, 0.0)
        if y0 != y0:  # no interface (components mixed or profile non-monotone)
            hist.append(dict(step=step, y_int=float("nan"),
                             rho1=float(r1.mean()), rho2=float(r2.mean()),
                             rho_tot_lower=float((r1 + r2).mean()),
                             rho_tot_upper=float((r1 + r2).mean()),
                             S=float(uy.mean(dim=1).abs().max()),
                             mass1=float(r1.sum()) / mass0[0] - 1.0,
                             mass2=float(r2.sum()) / mass0[1] - 1.0,
                             finite=bool(torch.isfinite(f1).all() and torch.isfinite(f2).all())))
            if not hist[-1]["finite"]:
                break
            continue
        # bulk windows: >= 8 cells away from the interface and from the
        # periodic image at the strip edges
        lo = slice(3, max(4, int(y0) - 8))
        hi = slice(min(ny - 4, int(y0) + 8), ny - 3)
        r1m, r2m = r1.mean(dim=1), r2.mean(dim=1)
        hist.append(dict(
            step=step,
            y_int=y0,
            alpha=alpha.tolist(),
            rho1=float(((r1m[lo].mean() + r2m[hi].mean()) / 2.0)),
            rho2=float(((r2m[lo].mean() + r1m[hi].mean()) / 2.0)),
            rho_tot_lower=float((r1m[lo] + r2m[lo]).mean()),
            rho_tot_upper=float((r1m[hi] + r2m[hi]).mean()),
            S=max(float(uy.mean(dim=1).abs().max()), 0.0),
            mass1=float(r1.sum()) / mass0[0] - 1.0,
            mass2=float(r2.sum()) / mass0[1] - 1.0,
            finite=bool(torch.isfinite(f1).all() and torch.isfinite(f2).all()),
        ))
        if not hist[-1]["finite"]:
            break

    last = hist[-1]
    # time window for steadiness: last quarter of the run
    tail = [h for h in hist if h["step"] >= 0.75 * last["step"]]
    if any(h["y_int"] != h["y_int"] for h in tail):
        drift = float("nan")
    else:
        drift = max(h["y_int"] for h in tail) - min(h["y_int"] for h in tail)
    drho = (max(h["rho1"] for h in tail) - min(h["rho1"] for h in tail),
            max(h["rho2"] for h in tail) - min(h["rho2"] for h in tail))

    # 10-90 thickness from the final normalised order parameter
    alpha = torch.tensor(last["alpha"]) if "alpha" in last else torch.zeros(ny)
    a_min, a_max = float(alpha.min()), float(alpha.max())
    norm = (alpha - a_min) / max(a_max - a_min, 1e-30)
    y10, y90 = _crossing(norm, 0.9), _crossing(norm, 0.1)
    w_10_90 = abs(y10 - y90) if (y10 == y10 and y90 == y90) else float("nan")
    w_tanh = w_10_90 / TANH_10_90 if w_10_90 == w_10_90 else float("nan")

    rho_A, rho_B = last["rho1"], last["rho2"]
    contrast = (rho_A - rho_B) / max(rho_A + rho_B, 1e-30)
    steady = bool(drift == drift and drift < DRIFT_MAX_CELLS and max(drho) < 1e-3 and last["finite"])
    admissible_ = admissible(dict(w_10_90=w_10_90, contrast=contrast,
                                 steady=steady, drift=drift))

    return dict(
        config=dict(g12=g12, ny=ny, nx=nx, steps=steps, seed_width=seed_width,
                    tau1=tau1, tau2=tau2, sc_form=sc_form,
                    boundary="periodic both directions (flat interface, no walls)"),
        rho_A=rho_A,
        rho_B=rho_B,
        rho_tot_lower=last["rho_tot_lower"],
        rho_tot_upper=last["rho_tot_upper"],
        contrast=contrast,
        y_int=last["y_int"],
        w_10_90=w_10_90,
        w_tanh=w_tanh,
        alpha_final=alpha.tolist(),
        drift=drift,
        drho=drho,
        steady=steady,
        admissible=admissible_,
        thresholds=dict(w_min_cells=W_MIN_CELLS, contrast_min=CONTRAST_MIN,
                        drift_max_cells=DRIFT_MAX_CELLS, tanh_10_90=TANH_10_90),
        wall_time_s=time.time() - t0,
        mass_drift=(last["mass1"], last["mass2"]),
        runtime=dict(python=sys.version.split()[0], torch=torch.__version__),
        history=[{k: v for k, v in h.items() if k != "alpha"} for h in hist],
    )


def w_max_cells(h_effective: float = 47.0) -> float:
    """Upper interface-thickness bound: the sharp-interface reference needs w << H."""
    return h_effective / 10.0


def admissible(rec, h_effective: float = 47.0) -> bool:
    """Full pre-registered admissibility predicate (resolution + separation of scales)."""
    w = rec.get("w_10_90", float("nan"))
    return bool(w == w and W_MIN_CELLS <= w <= w_max_cells(h_effective)
                and rec.get("contrast", 0.0) >= CONTRAST_MIN
                and rec.get("steady", False) is not False
                and abs(rec.get("drift", 1e9)) < DRIFT_MAX_CELLS)


def select_model(records, h_effective: float = 47.0) -> dict:
    """Apply the pre-registered selection rule to the calibration records."""
    adm = [r for r in records if admissible(r, h_effective)]
    if not adm:
        return dict(selected=None, g12=None,
                    reason="no admissible G12 (interface unresolved or stratified state not sustained)")
    best = max(adm, key=lambda r: abs(r["config"]["g12"]))
    return dict(selected=best["config"]["g12"],
                rho_A=best["rho_A"], rho_B=best["rho_B"],
                w_10_90=best["w_10_90"], w_tanh=best["w_tanh"],
                contrast=best["contrast"],
                reason=f"max |G12| among admissible ({W_MIN_CELLS:g} <= w_10_90 <= "
                       f"{w_max_cells(h_effective):g} cells, contrast >= {CONTRAST_MIN}, "
                       f"|drift| < {DRIFT_MAX_CELLS} cell)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ny", type=int, default=96)
    ap.add_argument("--nx", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--g12", type=float, nargs="*", default=list(G12_GRID))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    records = []
    for g in args.g12:
        rec = run_flat_interface(g12=g, ny=args.ny, nx=args.nx, steps=args.steps)
        records.append(rec)
        print("CALIB " + json.dumps(dict(
            g12=g, rho_A=round(rec["rho_A"], 5), rho_B=round(rec["rho_B"], 5),
            contrast=round(rec["contrast"], 4), w_10_90=round(rec["w_10_90"], 3),
            w_tanh=round(rec["w_tanh"], 3), drift=round(rec["drift"], 4),
            steady=rec["steady"], admissible=rec["admissible"],
            wall_time_s=round(rec["wall_time_s"], 1))))
    sel = select_model(records)
    out = dict(benchmark="mcmp_stratified_poiseuille", round="R4-calibration",
               anchor="GAP-LBM-MCMP-POISEUILLE",
               protocol=dict(g12_grid=list(args.g12), w_min_cells=W_MIN_CELLS,
                             contrast_min=CONTRAST_MIN, drift_max_cells=DRIFT_MAX_CELLS),
               records=records, selection=sel)
    print("SELECTED " + json.dumps(sel))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())