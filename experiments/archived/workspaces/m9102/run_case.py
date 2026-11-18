import sys, os, json, time; sys.path.insert(0,'src')
import numpy as np
from tensorlbm.elbm_channel_native import native_channel_run
label = os.environ["LABEL"]; re_tau = float(os.environ["RETAU"]); u_tau=float(os.environ.get("UTAU","0.008"))
nx,ny,nz = int(os.environ.get("NX","96")), int(os.environ.get("NY","98")), int(os.environ.get("NZ","48"))
steps, warmup = int(os.environ.get("STEPS","80000")), int(os.environ.get("WARMUP","25000"))
t0=time.time()
r = native_channel_run(nx=nx,ny=ny,nz=nz,steps=steps,warmup=warmup,sample_every=int(os.environ.get("SAMPLE","20")),
                       re_tau=re_tau,u_tau=u_tau,n_iter=int(os.environ.get("NITER","6")),log_every=2000)
el=time.time()-t0
np.savez(f"artifacts/elbm/raw/{label}.npz", **{f"prof_{k}":v for k,v in r.profiles.items()}, history=r.history,
         scalars=np.array([r.scalars[k] for k in ("samples","min_population","max_abs_u","alpha_mean","alpha_min","alpha_max","entropy_drop_max","finite","nu_nominal_internal","mean_rho")]))
meta = dict(label=label, re_tau=re_tau, u_tau=u_tau, nx=nx, ny=ny, nz=nz, steps=steps, warmup=warmup,
            wall_clock_s=el, mcell_s=nx*ny*nz*steps/el/1e6, **r.scalars)
json.dump(meta, open(f"artifacts/elbm/raw/{label}_meta.json","w"), indent=1, default=float)
print(json.dumps(meta, indent=1, default=float), flush=True)
