#!/usr/bin/env python
"""Measurement-only Gallium PF grid/time convergence diagnostic.

Runs the existing single isothermal-interface Stefan model unchanged.  Refinement
keeps the reported Gau--Viskanta Fo and Ra invariant: dt is lattice-fixed, so
steps scale as nx**2 and |gy| scales as nx**-3.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
from benchmark_gallium_pf import (  # noqa: E402
    GALLIUM_SOLID_TO_LIQUID_CONDUCTIVITY_RATIO,
    run_gallium_pf,
    stefan_nondimensional_diagnostic,
)
from benchmark_gallium_melting import _GV_FO, _GV_FLIQ  # noqa: E402

BASE_NX, BASE_NY, BASE_GY = 40, 56, -0.001875
TAU, TAU_T = 0.506, 0.8
CP, LATENT, T_HOT, T_COLD, T_MELT = 1.0, 18.52, 1.0, 0.0, 0.148


def grid_time_mapping(*, nx: int, ny: int, target_fo=(0.205, 0.410)) -> dict:
    """Return exact lattice mapping; this helper has no model-side effects."""
    if ny * BASE_NX != nx * BASE_NY:
        raise ValueError("grid must preserve the 40:56 cavity aspect ratio")
    alpha = (TAU_T - 0.5) / 3.0
    factor = nx / BASE_NX
    steps = tuple(round(fo * nx * nx / alpha) for fo in target_fo)
    gy = BASE_GY / factor**3
    nu = (TAU - 0.5) / 3.0
    ra = abs(gy) * 0.1 * (T_HOT - T_COLD) * nx**3 / (nu * alpha)
    return {"nx": nx, "ny": ny, "factor": factor, "alpha": alpha,
            "Fo_factor": alpha / nx**2, "target_Fo": tuple(target_fo),
            "steps": steps, "gy": gy, "Ra": ra}


def metrics(result: dict, target_fo: tuple[float, ...]) -> dict:
    history = result["history"]
    fo = np.array([x["Fo"] for x in history])
    fl = np.array([x["f_liq"] for x in history])
    front = np.array([(x["s_top"] + x["s_mid"] + x["s_bot"]) / 3 for x in history])
    ref = np.interp(np.array(target_fo), _GV_FO, _GV_FLIQ)
    measure = np.interp(np.array(target_fo), fo, fl)
    front_measure = np.interp(np.array(target_fo), fo, front)
    mape = float(np.mean(np.abs(measure - ref) / ref) * 100)
    # Sensible+latent total in model lattice units; fixed phase/temperature
    # wall values make it a reporting diagnostic, not a conservation assertion.
    energy = float((CP * result["T_field"] + LATENT * (result["phi_field"] + 1.0) / 2.0).sum())
    return {"Fo": list(target_fo), "f_liq": measure.tolist(), "f_ref": ref.tolist(),
            "front_lattice": front_measure.tolist(), "front_over_width": (front_measure / result["nx"]).tolist(),
            "MAPE_percent": mape, "energy_lattice_total": energy,
            "final": {k: float(result[k]) for k in ("Fo", "f_liq", "s_top", "s_mid", "s_bot", "deformation", "u_max")}}


def run_case(mapping: dict, device: str) -> dict:
    final_step = mapping["steps"][-1]
    result = run_gallium_pf(nx=mapping["nx"], ny=mapping["ny"], tau=TAU, tau_T=TAU_T,
                            T_hot=T_HOT, T_cold=T_COLD, T_melt=T_MELT, cp=CP,
                            L_latent=LATENT, beta=0.1, gy=mapping["gy"],
                            solid_conductivity_ratio=GALLIUM_SOLID_TO_LIQUID_CONDUCTIVITY_RATIO,
                            steps=final_step, device=device, log_every=mapping["steps"][0], quiet=True)
    return {"mapping": mapping, "metrics": metrics(result, mapping["target_Fo"]),
            "nondimensional": stefan_nondimensional_diagnostic(nx=mapping["nx"], tau_T=TAU_T,
                steps=final_step, cp=CP, latent_heat=LATENT, T_hot=T_HOT, T_melt=T_MELT)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", action="append", type=int, default=[])
    p.add_argument("--device", default="sdaa:0")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    grids = args.grid or [40, 80]
    data = {"physical_mapping": {"cavity_width_m": 0.0635, "cavity_height_m": 0.0889,
            "T_hot_C": 38.3, "T_cold_C": 28.3, "T_melt_C": 29.78,
            "Pr_target": 0.020, "Ra_target": 60000.0, "Ste_target": 0.046,
            "k_solid_W_mK": 40.6, "k_liquid_W_mK": 28.0,
            "k_s_over_k_l": GALLIUM_SOLID_TO_LIQUID_CONDUCTIVITY_RATIO},
            "lattice_mapping": {"tau": TAU, "tau_T": TAU_T, "cp": CP, "L": LATENT,
            "T_hot": T_HOT, "T_cold": T_COLD, "T_melt": T_MELT,
            "nu": (TAU - .5) / 3, "alpha": (TAU_T - .5) / 3}, "cases": []}
    for nx in grids:
        case = run_case(grid_time_mapping(nx=nx, ny=nx * BASE_NY // BASE_NX), args.device)
        data["cases"].append(case)
        print(json.dumps(case, indent=2), flush=True)
    if len(data["cases"]) >= 2:
        early = [x["metrics"]["f_liq"][0] - x["metrics"]["f_ref"][0] for x in data["cases"]]
        data["early_overshoot"] = {"absolute": early, "drops_with_resolution": bool(abs(early[-1]) < abs(early[0]))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
