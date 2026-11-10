import sys, time
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import power_law, run_channel, l2_relative_error, channel_geometry
from tensorlbm.nonnewtonian_frontier.reference import power_law_velocity
torch.set_num_threads(4)
n, nu_wall, UM = 0.7, 0.1, 1e-3
for h in (32, 64, 128):
    H=h/2; gw=UM*(n+1)/(n*H); K=nu_wall*gw**(1-n); a=K*gw**n/H
    cl=power_law(K,n)
    y=channel_geometry(h)["y_center"]
    ui=power_law_velocity(y,H,a,K,n); ui[0]=0; ui[-1]=0
    for nst in (40000,):
        t0=time.time()
        out=run_channel(h,a,cl,nx=4,n_steps=nst,tol=0.0,check_every=1000,u_init=ui)
        e=l2_relative_error(out["u"][1:-1].double(), power_law_velocity(y[1:-1],H,a,K,n))
        print(f"h={h} steps={nst} L2={e*100:.5f}% resid={out['residual']:.1e} t={time.time()-t0:.0f}s", flush=True)
