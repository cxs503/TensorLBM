import sys, time
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.nonnewtonian_frontier import newtonian, channel_reference, l2_relative_error, run_channel
torch.set_num_threads(8)
h=32; H=16.0; nu=0.1; U=0.02; a=2*nu*U/H**2
cl=newtonian(nu)
y=torch.arange(h+2,dtype=torch.float64)-0.5-H
ref=channel_reference(cl,a,H,y=y)
prev=0.0
for nst in (4000, 8000, 16000, 32000):
    out=run_channel(h,a,cl,nx=4,n_steps=nst,tol=1e-20,check_every=nst,u_init=ref["u"].clone())
    e=l2_relative_error(out["u"][1:-1].double(), ref["u"][1:-1])
    print(f"steps={nst} L2={e*100:.4f}% resid={out['residual']:.2e}", flush=True)
