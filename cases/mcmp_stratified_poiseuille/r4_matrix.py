"""Round-4 acceptance matrix for GAP-LBM-MCMP-POISEUILLE.

Cells are fixed before running (see ``docs/MCMP_STRATIFIED_POISEUILLE_R4.md``
§2 for the pre-registration and the calibration that selected ``G_12``):

    sc_form in {"standard", "edm"}          # pair-force discretisation
    ny      in {49, 65}                     # two resolutions (criterion 4)
    kind    in {"driven", "static"}

Config (identical for every cell): the model selected by
``coexistence.py`` — ``G_12``, the coexisting densities ``rho_A``/``rho_B`` and
the equilibrium tanh width — instead of the round-1..3 nominal 0.7/0.3 state.
Everything else (halfway bounce-back wall ``H = ny-2``, ``mirror`` neutral
wetting, ``u_eq="viscous"``, Guo forcing, ``G_x = 5e-5`` for the driven cells)
is unchanged from round 3, so the comparison is a single-variable change
(initial state) plus the two force discretisations.

Writes one JSON record per cell to
``artifacts/mcmp_stratified_poiseuille/r4/cells/``.

Usage::

    PYTHONPATH=src python cases/mcmp_stratified_poiseuille/r4_matrix.py
    PYTHONPATH=src python cases/mcmp_stratified_poiseuille/r4_matrix.py --shard 0/4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

OUT_DIR = os.path.join("artifacts", "mcmp_stratified_poiseuille", "r4", "cells")
CALIB = os.path.join("artifacts", "mcmp_stratified_poiseuille", "r4",
                     "coexistence_scan.json")
G_X = 5e-5

NX = 4
TAU_LOWER = 1.0
TAU_UPPER = 0.75
STEPS = {49: (40000, 30000), 65: (60000, 40000)}   # (driven, static)

CELLS = [
    (f"{kind}_ny{ny}_{sc}", kind, ny, sc)
    for sc in ("standard", "edm")
    for ny in (49, 65)
    for kind in ("driven", "static")
]


def load_model(path: str = CALIB) -> dict:
    """Read the calibration file and return the selected model parameters."""
    if not os.path.exists(path):
        raise SystemExit(
            f"missing calibration file {path}; run coexistence.py first "
            "(the acceptance config must come from the measurement, not from a constant)"
        )
    cal = json.load(open(path))
    sel = cal.get("selection") or {}
    if sel.get("selected") is None:
        raise SystemExit(f"no admissible model in {path}: {sel.get('reason')}")
    return dict(g12=sel["selected"], rho_A=sel["rho_A"], rho_B=sel["rho_B"],
                width=sel["w_tanh"], w_10_90=sel["w_10_90"],
                contrast=sel["contrast"])


def run_cell(tag: str, kind: str, ny: int, sc_form: str, model: dict) -> dict:
    t0 = time.time()
    steps = STEPS[ny][0 if kind == "driven" else 1]
    if kind == "driven":
        from cases.mcmp_stratified_poiseuille.run_sweep import run_case

        rec = run_case(
            ny=ny, nx=NX, tau_lower=TAU_LOWER, tau_upper=TAU_UPPER, G_x=G_X,
            steps=steps, G_12=model["g12"], u_eq="viscous", force_scheme="guo",
            rho_heavy=model["rho_A"], rho_light=model["rho_B"],
            interface_width=model["width"], mode="two_phase", wall_mode="mirror",
            sc_form=sc_form,
        )
    else:
        from cases.mcmp_stratified_poiseuille.spurious_diagnostic import run

        rec = run(ny=ny, nx=NX, tau_lower=TAU_LOWER, tau_upper=TAU_UPPER,
                  gx=0.0, g12=model["g12"], steps=steps, u_eq="viscous",
                  force_scheme="guo", wall_mode="mirror", sc_form=sc_form,
                  rho_heavy=model["rho_A"], rho_light=model["rho_B"],
                  width=model["width"], sample_every=max(200, steps // 150))
    rec["cell"] = dict(tag=tag, kind=kind, ny=ny, sc_form=sc_form, steps=steps,
                       model=model, wall_time_s=time.time() - t0)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--calibration", default=CALIB)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args(argv)

    model = load_model(args.calibration)
    cells = CELLS
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cells = [c for k, c in enumerate(cells) if k % n == i]
    os.makedirs(args.out_dir, exist_ok=True)
    print("MODEL " + json.dumps(model))
    for tag, kind, ny, sc in cells:
        path = os.path.join(args.out_dir, f"{tag}.json")
        if os.path.exists(path):
            print("skip (exists)", path)
            continue
        rec = run_cell(tag, kind, ny, sc, model)
        with open(path, "w") as fh:
            json.dump(rec, fh, indent=1)
        if kind == "driven":
            print("CELL " + json.dumps(dict(tag=tag, Q_mix=rec.get("Q_mix"),
                                             y_int=rec.get("interface_crossing"),
                                             u_max=rec.get("u_max_mix"),
                                             phi=rec.get("phi_contrast"),
                                             steady=rec.get("steady"),
                                             wall_time_s=round(rec["cell"]["wall_time_s"], 1))))
        else:
            s = rec["series"][-1]
            print("CELL " + json.dumps(dict(tag=tag, S_ifc=s["S_ifc"], S_all=s["S_all"],
                                             y_int=s["y_int"], phi=s["phi_contrast"],
                                             wall_time_s=round(rec["cell"]["wall_time_s"], 1))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())