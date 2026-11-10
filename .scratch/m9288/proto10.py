import sys
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import stream
from tensorlbm.powerlaw import collide_powerlaw_bgk_forced
from tensorlbm.nonnewtonian_frontier.channel import bounce_back_halfway_2d, channel_geometry
from tensorlbm.nonnewtonian_frontier import newtonian, channel_reference, l2_relative_error

torch.set_num_threads(4)
h=32; H=16.0; nu=0.1; U=0.02; a=2*nu*U/H**2
geo=channel_geometry(h, nx=4); ny,nx=geo["ny"],geo["nx"]
solid, fluid = geo["solid"], geo["fluid"]
y=geo["y_center"]
ref=channel_reference(newtonian(nu),a,H,y=y)["u"]
rho0=torch.ones((ny,nx)); zero=torch.zeros((ny,nx))
us=ref.clone().float(); us[0]=0; us[-1]=0
f=equilibrium(rho0, us.unsqueeze(1).expand(ny,nx).contiguous(), zero)
tau=torch.full((ny,nx), 3*nu+0.5)
for step in range(1,32001):
    f_pre=f
    f=collide_powerlaw_bgk_forced(f, a, 1.0, 1.0, tau_field=tau)
    f=bounce_back_halfway_2d(f, f_pre, solid)
    f=stream(f)
    if step % 4000 == 0:
        r,ux,uy=macroscopic(f)
        prof=ux.mean(dim=1)+0.5*a
        err=l2_relative_error(((prof[1:-1])).double(), ref[1:-1])
        print(f"step={step} mass={float(f.sum()):.10f} rho_mean={float(f.mean()):.8f} ux_mean={float(ux.mean()):.6e} "
              f"u_max={float(prof[1:-1].max()):.8f} L2={err*100:.4f}% xvar={float(ux.std(dim=1).max()):.2e}", flush=True)
