"""Pilot 2: sane forcing (u_tau = 0.002) -> is the smooth channel Cf right,
and does the corrected (nonequilibrium) wall-force audit close on rho g h_eff?"""
import json
import time

from tensorlbm.underice_drag import UnderIceChannelConfig, run_underice_channel_native

GRID = dict(nx=32, ny=32, nz=66, re_tau=180, sample_interval=20, threads=64)
for steps, warm in [(30000, 8000), (60000, 20000)]:
    cfg = UnderIceChannelConfig(steps=steps, warmup=warm, u_tau=0.002, **GRID)
    t = time.time()
    r = run_underice_channel_native(cfg)
    row = dict(steps=steps, wall=round(time.time() - t, 1), cf=r["cf"],
               u_plus=r["u_plus_bulk"], re_h=r["re_h"], mach=r["max_mach"],
               tau_balance=r["tau_total"], tau_me=r["tau_me"],
               tau_skin=r["tau_skin"], tau_form=r["tau_form"],
               mom_res=r["momentum_balance_residual"],
               closure=r["closure_residual"], drift=r["stationarity_drift"],
               cf_log=r["cf_log_law"], cf_dean=r["dean_cf"],
               log_fit=r["log_fit"])
    print(json.dumps(row), flush=True)