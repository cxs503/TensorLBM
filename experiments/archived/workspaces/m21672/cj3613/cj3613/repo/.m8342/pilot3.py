"""Pilot 3: does the smooth channel converge, and which stress audit is right?

Compares three independent estimates of the wall stress on the *smooth* channel
where the answer is exact (tau = rho g h_eff):
  tau_flux  x-z momentum flux at the wall row   (textbook definition)
  tau_me    nonequilibrium link momentum exchange
  tau_skin  2 nu u_x on the z-normal faces
"""
import json
import time

from tensorlbm.underice_drag import UnderIceChannelConfig, run_underice_channel_native

GRID = dict(nx=32, ny=32, nz=66, re_tau=180, u_tau=0.002, sample_interval=20, threads=64)
for steps in [100000, 200000]:
    cfg = UnderIceChannelConfig(steps=steps, warmup=steps // 3, **GRID)
    t = time.time()
    r = run_underice_channel_native(cfg)
    print(json.dumps(dict(
        steps=steps, wall=round(time.time() - t, 1), cf=r["cf"], u_plus=r["u_plus_bulk"],
        re_h=round(r["re_h"]), mach=round(r["max_mach"], 4),
        tau_bal=r["tau_total"], tau_flux=r["tau_flux"], tau_me=r["tau_me"],
        tau_skin=r["tau_skin"], tau_form=r["tau_form"],
        flux_res=(r["tau_flux"] - r["tau_total"]) / r["tau_total"],
        me_res=(r["tau_me"] - r["tau_total"]) / r["tau_total"],
        skin_res=(r["tau_skin"] - r["tau_total"]) / r["tau_total"],
        drift=round(r["stationarity_drift"], 5), cf_log=r["cf_log_law"],
        cf_dean=round(r["dean_cf"], 5), log_fit=r["log_fit"])), flush=True)