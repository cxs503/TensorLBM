"""Pilot 4: does resolving the near-wall layer (64 cells/half) sustain turbulence?

Same forcing (u_tau = 0.002, Re_tau = 180) but nz = 130, nx = ny = 48.
"""
import json
import time

from tensorlbm.underice_drag import UnderIceChannelConfig, run_underice_channel_native

for steps, nz, nxy in [(120000, 130, 48)]:
    cfg = UnderIceChannelConfig(nx=nxy, ny=nxy, nz=nz, re_tau=180, u_tau=0.002,
                               steps=steps, warmup=steps // 3, sample_interval=20, threads=64)
    t = time.time()
    r = run_underice_channel_native(cfg)
    print(json.dumps(dict(
        steps=steps, nz=nz, wall=round(time.time() - t, 1), cf=r["cf"], u_plus=r["u_plus_bulk"],
        re_h=round(r["re_h"]), mach=round(r["max_mach"], 4), h_eff=r["h_eff"], tau=r["tau"],
        tau_bal=r["tau_total"], tau_flux=r["tau_flux"], tau_me=r["tau_me"],
        tau_skin=r["tau_skin"], tau_form=r["tau_form"],
        flux_res=(r["tau_flux"] - r["tau_total"]) / r["tau_total"],
        skin_res=(r["tau_skin"] - r["tau_total"]) / r["tau_total"],
        drift=round(r["stationarity_drift"], 5), cf_log=r["cf_log_law"],
        cf_dean=round(r["dean_cf"], 5), log_fit=r["log_fit"])), flush=True)