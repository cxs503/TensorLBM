"""LBM grid-convergence + continuity study for P66-W2-N04 (background run)."""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.ice_fullship.lbm_porous import (  # noqa: E402
    PorousLBM, PorousLBMConfig, residual_decades,
)

OUT = os.path.join("runs", "p66_w2_n04")
os.makedirs(OUT, exist_ok=True)


def gci(f1, f2, f3, r=1.5):
    """Roache GCI with three grids (fine, medium, coarse) and measured order."""
    e21 = abs(f2 - f1)
    e32 = abs(f3 - f2)
    if e21 == 0.0 or e32 == 0.0:
        return dict(p=float("nan"), gci_fine=float("nan"), f_extrap=f1)
    ratio = e32 / e21
    p = abs(np.log(ratio) / np.log(r)) if ratio > 0 else float("nan")
    f_extrap = f1 + (f1 - f2) / (r ** p - 1.0)
    Fs = 1.25
    gci_fine = Fs * abs((f2 - f1) / f1) / (r ** p - 1.0)
    return dict(p=float(p), gci_fine=float(gci_fine), f_extrap=float(f_extrap),
                e21=float(e21), e32=float(e32))


def run_one(n_hull: int, ice_layer: int, steps: int, check: int = 100):
    cfg = PorousLBMConfig(n_hull=n_hull, ice_layer_cells=ice_layer)
    solver = PorousLBM(cfg)
    t0 = time.time()
    out = solver.run(steps, check_every=check)
    out["grid"] = dict(n_hull=n_hull, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                       cells=cfg.nx * cfg.ny * cfg.nz, tau=cfg.tau)
    out["wall_time"] = time.time() - t0
    out["residual_decades"] = residual_decades(out["residual_history"])
    out["sink_power"] = solver.sink_power
    out["outlet_scale"] = solver._outlet_scale
    out["pre_correction_imbalance"] = solver.pre_correction_imbalance()
    return out


def main():
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    results = {"steps": steps, "runs": {}}
    for n_hull in (32, 48, 64):
        for ice_layer in (0, 2):
            key = f"h{n_hull}_ice{ice_layer}"
            print("running", key, flush=True)
            res = run_one(n_hull, ice_layer, steps)
            results["runs"][key] = res
            print(key, "R_x", res["force"]["R_x_total"],
                  "residual", res["final_residual"],
                  "decades", res["residual_decades"],
                  "flux", res["max_flux_imbalance"], flush=True)
            with open(os.path.join(OUT, "lbm_study.json"), "w") as fh:
                json.dump(results, fh, indent=2)

    # ---- GCI on the total and on the ice-induced (porous) resistance increment
    def series(keyfmt):
        return [results["runs"][keyfmt % n]["force"]["R_x_total"] for n in (64, 48, 32)]

    tot = series("h%d_ice2")
    ow = series("h%d_ice0")
    inc = [a - b for a, b in zip(tot, ow)]
    results["R_total_fine_med_coarse"] = tot
    results["R_openwater_fine_med_coarse"] = ow
    results["R_ice_hydro_fine_med_coarse"] = inc
    results["gci_total"] = gci(*tot)
    results["gci_ice_increment"] = gci(*inc)
    results["gci_openwater"] = gci(*ow)
    # finest-grid force decomposition
    f = results["runs"]["h64_ice2"]["force"]
    results["force_split_finest"] = f
    with open(os.path.join(OUT, "lbm_study.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps({k: results[k] for k in ("gci_total", "gci_ice_increment",
                                             "gci_openwater")}, indent=2), flush=True)


if __name__ == "__main__":
    main()