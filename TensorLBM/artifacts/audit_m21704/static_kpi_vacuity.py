"""Audit probe (m21704): is the ``layered_static`` spurious-current KPI informative?

Claim under test
----------------
m21640 reports ``layered_static`` spurious current = 0.0 % of the driven peak and
counts that as a passed KPI gate ("spurious current of the static interface
<= 15 %").

Hypothesis (adversarial): the case is run with ``nx = 1`` and an initial state
that is *exactly* uniform in x, therefore the x-moment sum of the Shan-Chen
force cancels pairwise (``w_i dx_i = -w_j dx_j`` and ``psi(x+c_i) = psi(x+c_j)``
whenever the field is x-uniform), ``F_x == 0`` identically, ``u_x == 0`` for all
time -- the gate cannot fail and carries no information about spurious currents.

This probe measures:
  1. max|u_x| of the shipped ``layered_static`` configuration (nx = 1, no noise);
  2. max|u_x| of the same physics at nx = 8 with the shipped initial state
     (still x-uniform -> still expected to be exactly 0);
  3. max|u_x| of the same physics at nx = 8 with a 1e-3 density perturbation
     (x-dependent -> the symmetry that zeroes F_x is broken).

If (1) and (2) are exactly 0 while (3) is not, the reported 0.0 % is a
*structural* zero, not a physical spurious-current measurement.

Run:  PYTHONPATH=src python artifacts/audit_m21704/static_kpi_vacuity.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from tensorlbm.verification.layered_poiseuille.two_component_sc import (
    TwoComponentSCConfig, TwoComponentSCSolver)
from tensorlbm.verification.layered_poiseuille.analytic import solve_layered

BASE = dict(ny=18, tau_a=0.8, tau_b=1.0, G=2.5, g_x=0.0, g_y=0.0, steps=1500,
            init_rho_a=2.12, init_rho_b=2.36, init_layer_frac=0.5,
            interface_width=2.0, g_ramp_steps=500, u_eq="viscous", wall_mode="mirror")


def driven_peak(cfg: TwoComponentSCConfig) -> float:
    sol = solve_layered(cfg.height, [0.5 * cfg.height], [cfg.nu_a, cfg.nu_b], 5e-5)
    return float(sol.u_peak())


def run(nx: int, noise: float):
    cfg = TwoComponentSCConfig(nx=nx, noise=noise, **BASE)
    prof = TwoComponentSCSolver(cfg).run()["final_profile"]
    ux = np.asarray(prof["ux_total"])
    return float(np.max(np.abs(ux))), float(np.max(np.abs(ux)) / driven_peak(cfg))


if __name__ == "__main__":
    out = {}
    for tag, nx, noise in (("nx1_shipped", 1, 0.0),
                           ("nx8_uniform", 8, 0.0),
                           ("nx8_perturbed_1e-3", 8, 1e-3)):
        peak, ratio = run(nx, noise)
        out[tag] = {"nx": nx, "noise": noise, "max_abs_ux": peak, "ratio_to_driven_peak": ratio}
        print(f"{tag:22s} nx={nx} noise={noise:g}  max|ux|={peak:.3e}  ratio={ratio:.3%}")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "static_kpi_vacuity.json"), "w") as fh:
        json.dump(out, fh, indent=1)