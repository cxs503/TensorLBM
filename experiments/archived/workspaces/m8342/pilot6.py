import json
from tensorlbm.underice_drag import UnderIceChannelConfig, run_underice_channel_native
base = dict(nx=32, ny=32, nz=66, re_tau=180, u_tau=0.002, sample_interval=20, threads=64,
            init_profile="parabolic")
for kw in [dict(steps=1500, init_scale=0.5, noise_amplitude=0.03),
           dict(steps=1500, init_scale=0.5, noise_amplitude=0.0),
           dict(steps=1500, init_scale=0.25, noise_amplitude=0.0),
           dict(steps=1500, init_scale=1.0, noise_amplitude=0.0),
           dict(steps=6000, init_scale=0.5, noise_amplitude=0.0)]:
    cfg = UnderIceChannelConfig(warmup=kw["steps"]//3, **{**base, **kw})
    r = run_underice_channel_native(cfg)
    print(json.dumps(dict(kw, u_plus=round(r["u_plus_bulk"],2), mach=round(r["max_mach"],4),
                          tau_flux=r["tau_flux"], tau_bal=r["tau_total"],
                          tau_skin=r["tau_skin"], drift=round(r["stationarity_drift"],4))), flush=True)
