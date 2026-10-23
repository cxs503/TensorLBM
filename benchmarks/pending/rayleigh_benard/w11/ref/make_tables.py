#!/usr/bin/env python
"""Wave-11 Track C: frozen gamma_ref table + eigenfunction seed exports.

For each grid level ny in {64,128,256} (nx = 2*ny, construction "half":
velocity walls at half-planes 0.5 / ny-1.5, thermal anti-symmetric Dirichlet
at the same planes, layer depth d = ny - 2, all measured by probes P1/P0.7):

  k = 2*pi/nx,  a = k*d
  Ra_c(a)  : continuum critical Ra at the domain's wavenumber (own solver)
  arms     : {0.85, 0.95, 1.0, 1.05, 1.15, 1.3} x Ra_c(a)
  gamma_ref(a, Ra, Pr) : continuum leading growth rate, per t* = d^2/alpha
  eigenfunctions : Theta, Phi, dPhi/dz sampled at node rows z_j=(j-0.5)/d,
             fluid rows only (walls excluded; the BC overwrites wall rows).

Outputs (ref/):
  ref_table.json          - machine-frozen reference table
  eig_ny{ny}.npz          - seeds: arm{frac}/Theta_node, Phi_node, Phider_node
"""

import json
import math
from pathlib import Path

import numpy as np
import rb_eig

PR = ((0.855 - 0.5) / 3.0) / ((1.0 - 0.5) / 3.0)  # = 0.71 exactly
N_SP = 64
ARMS = (0.85, 0.95, 1.0, 1.05, 1.15, 1.3)
T_HOT = 1.0
T_COLD = 0.0


def sample_rows(mode, ny):
    """Sample leading eigenmode on node rows; fluid rows 1..ny-2 at
    z_j = (j-0.5)/d, d = ny-2 (wall planes at z=0,1)."""
    d = ny - 2
    j = np.arange(ny)
    z = (j - 0.5) / d
    t = 2.0 * z - 1.0
    Th = rb_eig.barycentric_eval(mode["t"], mode["Theta"], t)
    Ph = rb_eig.barycentric_eval(mode["t"], mode["Phi"], t)
    Phd = rb_eig.barycentric_eval(mode["t"], mode["dPhi"], t)
    return {"z_rows": z, "Theta": Th, "Phi": Ph, "Phider": Phd}


def main():
    pdir = Path("/nfs/wangxi/runs/bm_widen_w11_20261006/c_rayleigh_benard/ref")
    tab = {
        "pr": PR,
        "N_spectral": N_SP,
        "arms": list(ARMS),
        "construction": "half",
        "tau": 0.855,
        "tau_T": 1.0,
        "alpha_lattice": (1.0 - 0.5) / 3.0,
        "nu_lattice": (0.855 - 0.5) / 3.0,
        "levels": [],
    }
    for ny in (64, 128, 256):
        d = ny - 2
        nx = 2 * ny
        k = 2.0 * math.pi / nx
        a = k * d
        rac = rb_eig.ra_critical(a, PR, lambda aa, rr, pp: rb_eig.sigma_spectral(aa, rr, pp, N_SP))
        lev = {
            "ny": ny,
            "nx": nx,
            "d": d,
            "k": k,
            "a": a,
            "Ra_c_at_a": rac,
            "steps_per_tstar": d * d / ((1.0 - 0.5) / 3.0),
            "arms": [],
        }
        npz = {}
        npz[f"ny{ny}/a"] = a
        npz[f"ny{ny}/Ra_c_at_a"] = rac
        npz[f"ny{ny}/z_rows"] = (np.arange(ny) - 0.5) / d
        for frac in ARMS:
            Ra = frac * rac
            sp = rb_eig.SpectralRB(a, N_SP)
            mode = sp.leading_mode(Ra, PR)
            smp = sample_rows(mode, ny)
            lev["arms"].append(
                {
                    "frac": frac,
                    "Ra": Ra,
                    "gamma_ref": mode["sigma"],
                    "sigma_solver_check": rb_eig.sigma_spectral(a, Ra, PR, N_SP),
                }
            )
            tag = f"arm{frac}"
            npz[f"ny{ny}/{tag}/Theta_node"] = smp["Theta"]
            npz[f"ny{ny}/{tag}/Phi_node"] = smp["Phi"]
            npz[f"ny{ny}/{tag}/Phider_node"] = smp["Phider"]
            # analytic eigenfunction identity checks (seed sanity)
            npz[f"ny{ny}/{tag}/Theta_wallrow0_antisym"] = -smp["Theta"][1]
            npz[f"ny{ny}/{tag}/sigma_mode"] = mode["sigma"]
        tab["levels"].append(lev)
        np.savez(pdir / f"eig_ny{ny}.npz", **npz)
        print(
            f"ny={ny} a={a:.8f} Ra_c(a)={rac:.6f} "
            + " ".join(f"{x['frac']:.2f}:{x['gamma_ref']:+.6f}" for x in lev["arms"])
        )
    (pdir / "ref_table.json").write_text(json.dumps(tab, indent=1))
    print("REF_TABLE_DONE")


if __name__ == "__main__":
    main()
