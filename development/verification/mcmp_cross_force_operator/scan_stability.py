#!/usr/bin/env python3
"""Parameter scan for the physics arm: find stable, phase-separating settings.

Scans the library's two-component Shan-Chen controls (G_12, tau, u_eq,
force_scheme) and reports, per combination, whether the run survives, the
maximum velocity, the density range, and the analytic-invariant residuals used
by the xval (EOS uniformity, Boltzmann relation, operator error on the measured
interface).  Used to fix the settings of ``run_xval.py``; not part of the
acceptance path.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arms  # noqa: E402


def main() -> None:
    torch.set_num_threads(int(os.environ.get("NTHREADS", "8")))
    t0 = time.time()
    out = []
    for G, tau, ueq, fs in itertools.product((-0.6, -0.9, -1.2, -1.5), (0.8, 1.0),
                                             ("self", "viscous"), ("shift", "guo")):
        r = arms.simulate_mcmp(n=48, G=G, tau=tau, steps=1500, report_every=1500,
                               u_eq=ueq, force_scheme=fs)
        if r["status"] != "ok":
            out.append({"G": G, "tau": tau, "u_eq": ueq, "force_scheme": fs,
                        "status": "diverged", "step": r["step"]})
            continue
        d = arms._diagnose_equilibrium(r, psi=arms.psi_linear)
        out.append({"G": G, "tau": tau, "u_eq": ueq, "force_scheme": fs, "status": "ok",
                    "max_u": d["max_u"], "rho1_min": d["rho1_min"], "rho1_max": d["rho1_max"],
                    "p_eos_uniformity_rel": d["p_eos_uniformity_rel"],
                    "boltzmann_residual_rel": d["boltzmann_relation_residual_rel"],
                    "operator_error_rel": d["operator_error_on_interface_rel"]})
        print(json.dumps(out[-1], ensure_ascii=False), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "artifacts", "scan_stability.json"), "w") as fh:
        json.dump({"cases": out, "elapsed_s": time.time() - t0}, fh, indent=1)
    print("elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"),
                exist_ok=True)
    main()