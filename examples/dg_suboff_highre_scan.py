"""SUBOFF high-Re stability + drag scan: DG-LBM vs standard LBM.

The earlier production runs targeted Re = 2M–10M.  At high Re the lattice tau
-> 0.5, so the DG band's tau_dg = tau_lbm - 1/2 -> 0 and the DG method-of-lines
goes unstable.  This scan finds the DG stability ceiling on the real SUBOFF
geometry and checks that standard LBM (+ LES) stays stable into high Re, with
the drag trend.

    PYTHONPATH=src python examples/dg_suboff_highre_scan.py
"""
from __future__ import annotations

import json
import math
import pathlib
import tempfile

import torch

from tensorlbm import DGLBMSuboffConfig, run_dg_lbm_suboff_flow


def scan(re, use_real_dg, n_steps=120):
    with tempfile.TemporaryDirectory() as d:
        cfg = DGLBMSuboffConfig(
            nx=96, ny=48, nz=48, u_in=0.06, re=float(re), hull_length=48.0,
            hull_type="bare_hull", dg_band=3.0, n_steps=n_steps, output_interval=60,
            output_root=pathlib.Path(d), run_name=f"r{re}_dg{int(use_real_dg)}",
            overwrite=True, device="cuda",
            use_real_dg=use_real_dg, dg_substeps=10,
            dynamic_smag=True,        # LES for high-Re stability
        )
        tau = cfg.tau
        try:
            rd = run_dg_lbm_suboff_flow(cfg)
            m = json.loads((rd / "run_metadata.json").read_text())
            diag = m.get("diagnostics", [])
            ms = diag[-1].get("max_speed", float("nan")) if diag else float("nan")
            drag = m.get("drag_force_lu", float("nan"))
            ok = math.isfinite(ms) and ms < 5.0
            return tau, ms, drag, ok
        except Exception as e:
            return tau, float("nan"), float("nan"), False


if __name__ == "__main__":
    print("SUBOFF high-Re scan (bare_hull, 96x48x48, dynamic-Smagorinsky, channel BC)\n")
    print(f"{'Re':>9} {'tau':>7} {'tau_dg':>7}  {'std: max|u| / drag / ok':>26}  {'DG: max|u| / drag / ok':>26}")
    for re in (1e2, 3e2, 1e3, 1e4, 1e5, 1e6):
        t_std, ms_std, d_std, ok_std = scan(re, use_real_dg=False)
        t_dg, ms_dg, d_dg, ok_dg = scan(re, use_real_dg=True)
        tau_dg = t_dg - 0.5
        def fmt(ms, d, ok): return f"{ms:7.4f} / {d:8.3f} / {'OK' if ok else 'UNSTABLE'}"
        print(f"{re:>9.0e} {t_std:>7.4f} {tau_dg:>7.4f}  {fmt(ms_std,d_std,ok_std):>26}  {fmt(ms_dg,d_dg,ok_dg):>26}")
