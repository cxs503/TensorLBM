#!/usr/bin/env python3
"""Chunked local runner for the XVAL-R3 study (cluster overflow fallback).

Usage:  python _scratch/r3_parts.py closure
        python _scratch/r3_parts.py growth -2 -1
        python _scratch/r3_parts.py fields
        python _scratch/r3_parts.py refine 512 -1
Accumulates results in xval/.../artifacts/r3_parts.json so the summary JSON can be
assembled later without re-running anything.
"""
import json
import sys
import time
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

HERE = Path("/srv/tensor/agent/work/m21247")
XV = HERE / "xval" / "XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT"
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(XV))

spec = spec_from_file_location("r3", XV / "xval_exact_darcy_r3.py")
r3 = module_from_spec(spec)
spec.loader.exec_module(r3)

OUT = XV / "artifacts" / "r3_parts.json"
data = json.loads(OUT.read_text()) if OUT.exists() else {"closure": [], "growth": [],
                                                         "refine": [], "fields": {}}


def save():
    OUT.write_text(json.dumps(data))


def main(argv):
    torch.set_num_threads(8)
    anchor = r3.load_anchor()
    part = argv[1]
    k = r3.K_MODE
    if part == "closure":
        for R in (-1.0, -0.5, 0.5, 1.0):
            row = {"R": R}
            for elliptic in ("oss_frozen", "exact_varcoef"):
                sim = r3.SpectralSaffmanTaylor2D(r3.NY, r3.NX, aspect=r3.ASPECT, pe=r3.PE,
                                                 viscosity_ratio=R, dt=r3.DT,
                                                 mode_index=r3.MODE, elliptic=elliptic)
                sim.set_concentration(sim.single_mode_ic(x0=0.5 * r3.ASPECT * r3.PE,
                                                         width=r3.W_FRONT, amplitude=1e-3,
                                                         n_mode=r3.MODE))
                sim._update_flow()
                row[elliptic] = {"residual_l2": sim.elliptic_residual()["residual_l2"],
                                 "divergence_error": sim.divergence_error(),
                                 "gmres_iter": sim.solver_stats["gmres_iter"],
                                 "gmres_residual": sim.solver_stats["gmres_residual"]}
            data["closure"] = [c for c in data["closure"] if c["R"] != R] + [row]
            save()
            print(f"closure R={R} done", flush=True)
    elif part == "growth":
        for R in [float(a) for a in argv[2:]]:
            theory = anchor.exact_linear_sigma(k, R, r3.W_FRONT)["sigma"].real
            exact = r3.run_case(R, "exact_varcoef")
            frozen = r3.run_case(R, "oss_frozen")
            denom = max(abs(theory), 1e-30)
            row = {"R": R, "sigma_theory": theory,
                   "sigma_exact": exact["sigma"], "sigma_frozen": frozen["sigma"],
                   "rel_exact": abs(exact["sigma"] - theory) / denom,
                   "rel_frozen": abs(frozen["sigma"] - theory) / denom,
                   "sign_exact_ok": bool((exact["sigma"] > 0) == (theory > 0)),
                   "sign_frozen_ok": bool((frozen["sigma"] > 0) == (theory > 0)),
                   "exact": {kk: vv for kk, vv in exact.items() if kk not in ("ts", "amps")},
                   "frozen": {kk: vv for kk, vv in frozen.items() if kk not in ("ts", "amps")},
                   "fit_exact": {"ts": exact["ts"], "amps": exact["amps"]},
                   "fit_frozen": {"ts": frozen["ts"], "amps": frozen["amps"]}}
            data["growth"] = [g for g in data["growth"] if g["R"] != R] + [row]
            save()
            print(f"growth R={R:+.1f} sigma_exact={exact['sigma']:+.6e} theory={theory:+.6e} "
                  f"rel={100*row['rel_exact']:.2f}%", flush=True)
    elif part == "fields":
        f = r3.run_case(-1.0, "exact_varcoef", keep_fields=True)
        ff = r3.run_case(-1.0, "oss_frozen", keep_fields=True)
        data["fields"] = {kk: f[kk] for kk in ("x", "y", "field_c", "field_psi")}
        data["fields_frozen"] = {kk: ff[kk] for kk in ("x", "y", "field_c")}
        save()
        print("fields done", flush=True)
    elif part == "refine":
        nx = int(argv[2])
        for R in [float(a) for a in argv[3:]]:
            theory = anchor.exact_linear_sigma(k, R, r3.W_FRONT)["sigma"].real
            res = r3.run_case(R, "exact_varcoef", nx=nx)
            row = {"R": R, "nx": nx, "dx": r3.ASPECT * r3.PE / nx,
                   "cells_per_front_width": r3.W_FRONT / (r3.ASPECT * r3.PE / nx),
                   "sigma": res["sigma"], "sigma_theory": theory,
                   "rel": abs(res["sigma"] - theory) / max(abs(theory), 1e-30),
                   "wall_s": res["wall_s"], "darcy_residual_max": res["darcy_residual_max"]}
            data["refine"] = [r for r in data["refine"]
                              if not (r["R"] == R and r["nx"] == nx)] + [row]
            save()
            print(f"refine R={R:+.1f} nx={nx} sigma={res['sigma']:+.6e} rel={100*row['rel']:.2f}% "
                  f"({res['wall_s']:.1f}s)", flush=True)
    print("saved", OUT)


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv)
    print(f"elapsed {time.time()-t0:.1f}s")