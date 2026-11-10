import sys, time
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import power_law, channel_reference, l2_relative_error, run_channel
from tensorlbm.nonnewtonian_frontier.reference import power_law_velocity
torch.set_num_threads(8)
U, nu_wall = 0.02, 0.1
for n in (0.7,):
    for h in (16, 32, 48):
        H=h/2; gw=U*(n+1)/(n*H); K=nu_wall*gw**(1-n); a=K*gw**n/H
        cl=power_law(K,n)
        y=torch.arange(h+1,dtype=torch.float64)-H
        ui=power_law_velocity(y,H,a,K,n); ui[0]=0; ui[-1]=0
        t0=time.time()
        out=run_channel(h,a,cl,nx=4,n_steps=40000,tol=1e-6,check_every=500,u_init=ui)
        e=l2_relative_error(out["u"][1:-1].double(), power_law_velocity(y[1:-1],H,a,K,n))
        print(f"n={n} h={h} steps={out['steps']} conv={out['converged']} resid={out['residual']:.1e} L2={e*100:.3f}% t={time.time()-t0:.0f}s", flush=True)
