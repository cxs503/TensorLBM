"""Pilot: smooth-channel convergence of Cf vs log law (criterion (a) pre-check).

Cheap ladder over integration length at one coarse grid; prints Cf, U+, Re_h,
momentum-balance residual and the log-fit intercept so the production-step
count can be chosen before the campaign is launched.
"""
import json
import time

from tensorlbm.underice_drag import UnderIceChannelConfig, run_underice_channel_native

GRID = dict(nx=32, ny=32, nz=66, re_tau=180, u_tau=0.02, sample_interval=20, threads=64)
LADDER = [(4000, 1000), (12000, 3000), (24000, 6000), (40000, 8000)]

out = []
for steps, warm in LADDER:
    cfg = UnderIceChannelConfig(steps=steps, warmup=warm, **GRID)
    t = time.time()
    r = run_underice_channel_native(cfg)
    dt = time.time() - t
    row = dict(steps=steps, warmup=warm, wall=round(dt, 1), cf=r["cf"],
               u_plus=r["u_plus_bulk"], re_h=r["re_h"],
               cf_log=r["cf_log_law"], cf_dean=r["dean_cf"],
               mom_res=r["momentum_balance_residual"],
               drift=r.get("stationarity_drift"),
               log_fit_b=(r["log_fit"] or {}).get("b"),
               log_fit_kappa=(r["log_fit"] or {}).get("kappa"),
               u_rms_over_u=r["u_rms"] / r["u_bulk"])
    out.append(row)
    print(json.dumps(row), flush=True)
print(json.dumps(out, indent=1))