"""Pilot 5: parabolic (exact steady) seed -> stationary state, audit check."""
import json, time
from tensorlbm.underice_drag import UnderIceChannelConfig, run_underice_channel_native
cfg = UnderIceChannelConfig(nx=32, ny=32, nz=66, re_tau=180, u_tau=0.002,
                            steps=20000, warmup=5000, sample_interval=20, threads=64,
                            init_profile="parabolic")
t = time.time(); r = run_underice_channel_native(cfg)
print(json.dumps(dict(wall=round(time.time()-t,1), cf=r["cf"], u_plus=round(r["u_plus_bulk"],2),
    re_h=round(r["re_h"]), cf_laminar_exact=12.0/r["re_h"], tau_bal=r["tau_total"],
    tau_flux=r["tau_flux"], tau_me=r["tau_me"], tau_skin=r["tau_skin"], tau_form=r["tau_form"],
    flux_res=(r["tau_flux"]-r["tau_total"])/r["tau_total"],
    me_res=(r["tau_me"]-r["tau_total"])/r["tau_total"],
    skin_res=(r["tau_skin"]-r["tau_total"])/r["tau_total"],
    drift=r["stationarity_drift"], mom_res=r["momentum_balance_residual"])), flush=True)
