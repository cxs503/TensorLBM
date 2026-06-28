"""Head-to-head: real-DG SUBOFF vs legacy (staircased BGK) drag, vs reference.

Runs both solvers at the same config on GPU and compares the resistance
coefficient Cd = |F_x| / (0.5 * rho * u_in^2 * wetted_area) against the
laminar flat-plate Blasius reference Cf = 1.328/sqrt(Re) (form-factor scaled).

    PYTHONPATH=src python examples/dg_lbm_suboff_bench_compare.py
"""
from __future__ import annotations

import json
import math
import pathlib

import torch

from tensorlbm import DGLBMSuboffConfig, run_dg_lbm_suboff_flow
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import (
    _appendage_factor,
    _laminar_friction_coefficient,
    _voxel_wetted_area,
)


def wetted_area(hull_type, nx, ny, nz, length, device="cpu"):
    mask, _ = build_suboff_mask(hull_type=hull_type, nx=nx, ny=ny, nz=nz,
                                cx=nx * 0.35, cy=ny * 0.5, cz=nz * 0.5,
                                length=length, device=device)
    return float(_voxel_wetted_area(mask, 1.0)), mask


def cd_from_drag(drag_lu, u_in, area):
    return abs(drag_lu) / (0.5 * 1.0 * u_in ** 2 * max(area, 1e-12))


def run_one(use_real_dg, cfg_dict, out, label):
    cfg = DGLBMSuboffConfig(use_real_dg=use_real_dg, **cfg_dict)
    rd = run_dg_lbm_suboff_flow(cfg)
    m = json.loads((pathlib.Path(out) / "dg_lbm_suboff" / cfg_dict["run_name"] / "run_metadata.json").read_text())
    drag = m["drag_force_lu"]
    print(f"  [{label}] drag_force_lu = {drag:.4f}")
    return drag


def main():
    out = pathlib.Path("/tmp/dg_bench_compare")
    out.mkdir(exist_ok=True)
    common = dict(
        nx=128, ny=64, nz=64, u_in=0.06, re=100.0, hull_length=80.0,
        hull_type=SuboffHullType.BARE_HULL.value, dg_band=4.0,
        n_steps=250, output_interval=50, output_root=out, overwrite=True,
        device="cuda", dg_substeps=10,
    )

    area, _ = wetted_area("bare_hull", common["nx"], common["ny"], common["nz"], common["hull_length"])
    u_in = common["u_in"]
    re = common["re"]
    cf_blasius = _laminar_friction_coefficient(re)
    ff = _appendage_factor(SuboffHullType.BARE_HULL)
    cd_ref = cf_blasius * ff
    print(f"wetted_area_lu2 = {area:.1f}  Re = {re}")
    print(f"reference: Cf_Blasius = {cf_blasius:.4f}  form-factor(1+k) = {ff:.2f}  Cd_ref = {cd_ref:.4f}\n")

    print("Running LEGACY (staircased BGK)...")
    drag_leg = run_one(False, {**common, "run_name": "legacy"}, out, "legacy")
    print("Running REAL DG hybrid...")
    drag_dg = run_one(True, {**common, "run_name": "real_dg"}, out, "real-dg")

    cd_leg = cd_from_drag(drag_leg, u_in, area)
    cd_dg = cd_from_drag(drag_dg, u_in, area)
    err_leg = abs(cd_leg - cd_ref) / cd_ref * 100
    err_dg = abs(cd_dg - cd_ref) / cd_ref * 100

    print("\n=== Cd comparison (Re=100, bare_hull) ===")
    print(f"  reference (Blasius×ff) : {cd_ref:.4f}")
    print(f"  legacy  staircased BGK : {cd_leg:.4f}   err {err_leg:5.1f}%")
    print(f"  real-DG hybrid         : {cd_dg:.4f}   err {err_dg:5.1f}%")
    print(f"  real-DG vs legacy Δ    : {abs(cd_dg - cd_leg)/cd_leg*100:.1f}%")


if __name__ == "__main__":
    main()
