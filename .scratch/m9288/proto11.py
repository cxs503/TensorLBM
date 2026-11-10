import sys
sys.path.insert(0, "/root/tp-agent/work/m9288/src")
import torch
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import stream
from tensorlbm.powerlaw import collide_powerlaw_bgk_forced
from tensorlbm.nonnewtonian_frontier.channel import bounce_back_halfway_2d, channel_geometry
torch.set_num_threads(4)
for label, a, u0 in (("a=0, u0=0.01", 0.0, 0.01), ("a=1e-4, u0=0", 1e-4, 0.0)):
    h=16; geo=channel_geometry(h, nx=4); ny,nx=geo["ny"],geo["nx"]
    solid=geo["solid"]
    rho0=torch.ones((ny,nx)); zero=torch.zeros((ny,nx))
    uu=torch.full((ny,nx), u0); uu[0]=0; uu[-1]=0
    f=equilibrium(rho0, uu, zero)
    tau=torch.full((ny,nx), 0.8)
    m0=float(f.sum())
    for step in range(1,4001):
        f_pre=f
        f=collide_powerlaw_bgk_forced(f, a, 1.0, 1.0, tau_field=tau)
        f=bounce_back_halfway_2d(f, f_pre, solid)
        f=stream(f)
        if step % 1000 == 0:
            r,ux,uy=macroscopic(f)
            print(f"{label} step={step} mass_rel={(float(f.sum())-m0)/m0:+.3e} rho_min={float(r.min()):.6f} rho_max={float(r.max()):.6f} ux_mean={float(ux.mean()):.6e}", flush=True)
