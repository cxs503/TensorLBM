import sys; sys.path.insert(0,'src')
import numpy as np, json
from tensorlbm.elbm_channel_native import native_channel_run
r = native_channel_run(nx=8,ny=18,nz=8,steps=3000,warmup=1000,sample_every=20,re_tau=30.0,u_tau=0.02,n_iter=4,noise_amplitude=0.05,log_every=500)
np.savez("artifacts/elbm/raw/_smoketest.npz", **{f"prof_{k}":v for k,v in r.profiles.items()}, history=r.history,
         scalars=np.array([r.scalars[k] for k in ("samples","min_population","max_abs_u","alpha_mean","alpha_min","alpha_max","entropy_drop_max","finite","nu_nominal_internal","mean_rho")]))
json.dump({"steps":3000}, open("artifacts/elbm/raw/_smoketest_meta.json","w"))
