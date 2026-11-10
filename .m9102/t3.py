import sys, time; sys.path.insert(0,'src')
import torch, numpy as np
torch.set_num_threads(128)
from tensorlbm.elbm_channel import *
cfg = ElbmChannelConfig(nx=48, ny=66, nz=32, steps=60, warmup=20, sample_every=10, log_every=20)
print({k:cfg.summary()[k] for k in ("half_height","body_force","nu","tau","delta_plus_wall_normal")})
t=time.time(); r = run_elbm_channel(cfg, progress=True); dt=time.time()-t
cells = cfg.nx*cfg.ny*cfg.nz
print("wall", round(dt,2), "s for", cfg.steps, "steps;", round(cells*cfg.steps/dt/1e6,2), "MLUPS")
print({k:v for k,v in r.scalars.items() if k not in ()})
print("u_mean center", r.profiles["u_mean"][len(r.profiles["u_mean"])//2])
