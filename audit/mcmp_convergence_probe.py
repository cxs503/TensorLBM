"""Audit probe (m15892): does the m15468 PASS survive time-convergence?

m15468 reported U_spurious = 3.98e-04 (<1e-3 PASS) for baseline_bgk_lattice at
n=96, G12=2.5, 1500 steps -- but with converged=false.  This probe re-runs the
same configuration with a tighter stop_tol (1e-4) and a much longer step budget
so the *plateau* value of max|u| can be read off the returned history.

Usage:  python mcmp_convergence_probe.py <method> <out.json>
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/root/tp-agent/work/m15468/src")
from tensorlbm.verification.spurious_current import METHODS, run_static_droplet  # noqa

method = sys.argv[1]
out_path = sys.argv[2]
cfg = METHODS[method]
t0 = time.time()
run = run_static_droplet(
    G12=2.5, n=96, radius=24.0, width=4.0,
    steps=40000, collision=cfg["collision"], force=cfg["force"],
    forcing=cfg.get("forcing", "guo"), stop_tol=1e-4, check_every=100,
    keep_fields=False,
)
hist = np.asarray(run["history"], dtype=float)
rec = {
    "method": method,
    "config": {"G12": 2.5, "n": 96, "radius": 24.0, "width": 4.0,
               "stop_tol": 1e-4, "check_every": 100},
    "converged": run["converged"],
    "steps_run": run["steps_run"],
    "u_max_final": run["u_max"],
    "u_rms_final": run["u_rms"],
    "rho1_centre": run["rho1_centre"],
    "rho1_far": run["rho1_far"],
    "mass1": run["mass1"], "mass2": run["mass2"],
    "rho_tot_drift": run["rho_tot_drift"],
    "history": hist.tolist(),
    "wall_s": time.time() - t0,
    "reading_at_1500": float(hist[hist[:, 0] == 1500][0, 1]) if (hist[:, 0] == 1500).any() else None,
    "peak_u_max": float(hist[:, 1].max()),
    "passes_1e3_at_plateau": bool(run["u_max"] < 1e-3),
    "passes_1e3_at_1500": bool(float(hist[hist[:, 0] == 1500][0, 1]) < 1e-3) if (hist[:, 0] == 1500).any() else None,
}
with open(out_path, "w") as f:
    json.dump(rec, f, indent=2)
print(json.dumps({k: v for k, v in rec.items() if k != "history"}, indent=2))
