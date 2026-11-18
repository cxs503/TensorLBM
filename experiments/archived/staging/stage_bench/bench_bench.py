import sys, time, torch
sys.path.insert(0,'src')
from tensorlbm.boundary.wall_resolved_channel3d import WallResolvedChannel3DConfig, run_wall_resolved_channel3d
print("cuda", torch.cuda.is_available(), flush=True)
for model,cs in (("cumulant",0.1),("natural_kbc",0.0)):
  for threads in (32,128):
    torch.set_num_threads(threads)
    cfg=WallResolvedChannel3DConfig(nx=128,ny=66,nz=64,re_tau=178.12,u_tau=0.01,steps=25,warmup_steps=5,
       sample_interval=10,report_interval=1000,checkpoint_interval=1000,collision_model=model,
       smagorinsky_cs=cs, compile_natural_kbc=False,
       device="cpu",output="/tmp/b.json",checkpoint="/tmp/b.ckpt")
    t=time.time(); run_wall_resolved_channel3d(cfg); dt=time.time()-t
    cells=cfg.nx*cfg.ny*cfg.nz
    print(f"RESULT {model} threads={threads} cells={cells} elapsed={dt:.1f}s Mcs/s={cells*cfg.steps/dt/1e6:.2f}", flush=True)
