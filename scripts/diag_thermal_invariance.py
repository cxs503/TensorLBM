"""Post-fix dimensional-invariance evidence.

Runs three converged-ish cases with the FIXED code:
  A. RB  (64x32, Ra=1e4, dT=2, 12000 steps)  -- compare with dT=1 baseline
  B. DHC (48,    Ra=1e4, dT=1, 12000 steps)  -- compare with pre-fix baseline history
  C. DHC (48,    Ra=1e4, dT=2, 12000 steps)  -- must equal B
"""
from __future__ import annotations

import json
import time

from tensorlbm.thermal_verification import simulate_dhc_2d, simulate_rayleigh_benard_2d

out = {"cases": {}}

t0 = time.time()
rb = simulate_rayleigh_benard_2d(
    nx=64, ny=32, ra=1e4, pr=0.71, tau=0.6, t_hot=2.0, t_cold=0.0,
    steps=12000, report_every=6000, device="cpu",
)
out["cases"]["rb_dT2_12000"] = {
    "nu": rb.nu, "nu_bot": rb.nu_bot, "nu_top": rb.nu_top, "u_max": rb.u_max,
    "g_beta": rb.g_beta, "ra_effective": rb.ra_effective, "dT": rb.dT,
    "energy_imbalance": rb.energy_imbalance, "elapsed_s": rb.elapsed_s,
}
print("A done", json.dumps(out["cases"]["rb_dT2_12000"]), flush=True)

for tag, thot in (("dhc_dT1_12000", 1.0), ("dhc_dT2_12000", 2.0)):
    d = simulate_dhc_2d(
        nx=48, ra=1e4, pr=0.71, tau=0.6, t_hot=thot, t_cold=0.0,
        steps=12000, report_every=6000, device="cpu",
    )
    out["cases"][tag] = {
        "nu_grad2": d.nu_grad2, "nu_left_grad2": d.nu_left_grad2,
        "nu_right_grad2": d.nu_right_grad2, "nu_grad1": d.nu_grad1,
        "nu_halfway": d.nu_halfway, "u_max": d.u_max,
        "g_beta": d.g_beta, "ra_effective": d.ra_effective, "dT": d.dT,
        "last_resid_u": d.last_resid_u, "last_resid_T": d.last_resid_T,
        "elapsed_s": d.elapsed_s,
    }
    print(tag, "done", json.dumps(out["cases"][tag]), flush=True)

out["total_elapsed_s"] = time.time() - t0
with open("artifacts/thermal_rb_debt/post_fix_invariance.json", "w") as fh:
    json.dump(out, fh, indent=2, default=float)
print("ALL DONE", out["total_elapsed_s"], flush=True)