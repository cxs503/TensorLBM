import sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks" / "xval_per_step_hook_mask_blend")); sys.path.insert(0, str(ROOT / "src"))
import run_xval as R
import tensorlbm.step_hooks as SH
torch.set_num_threads(4); DT = torch.float64

def run(ny, nx, tau, eps, arm, u_wall=0.02):
    nu=(tau-0.5)/3.0; y_b=0.5+eps; y_t=ny-1.5; H=y_t-y_b
    steps=R.steps_for_couette(H,nu)
    solid=torch.zeros(ny,nx,dtype=DT); solid[0,:]=1.0
    part=torch.zeros(ny,nx,dtype=DT); part[1,:]=float(eps)
    top=torch.zeros(ny,nx,dtype=DT); top[-1,:]=1.0
    f0=R.rest_field(ny,nx); tables=SH.tables_for(f0)
    if arm=="post":  partial=SH.partial_cell_layer("p",part,tau,(u_wall,0.0),reflect="post")
    elif arm=="pre": partial=SH.partial_cell_layer("p",part,tau,(u_wall,0.0),reflect="pre")
    elif arm=="naive": partial=SH.BoundaryLayer(name="p",mask=part,kind="moving",velocity=(u_wall,0.0))
    elif arm=="stair": partial=SH.moving_bounce_layer("p",part,tau if False else (u_wall,0.0))
    layers=[SH.stationary_bounce_layer("s0",solid),partial,SH.stationary_bounce_layer("t",top)]
    op=SH.masked_step(tau=tau,layers=layers,tables=tables)
    with torch.no_grad():
        st,_=SH.solve_with_hooks({"f":f0.clone()},steps,operator=op)
    rho,ux,_=R.macroscopic(st["f"]); u=ux.numpy()[:,0]
    ys=np.arange(2,ny-1,dtype=float); uu=u[2:ny-1]
    ref=u_wall*(y_t-ys)/H
    A=np.polyfit(ys,uu,1)
    u_at_yb=A[1]+A[0]*y_b
    return dict(l2=100*np.linalg.norm(uu-ref)/np.linalg.norm(ref), u_at_yb=u_at_yb/u_wall, prof=uu)

ny,nx,tau=24,8,0.8
for eps in (0.25,0.5,0.75,1.0):
    for arm in ("post","pre","naive"):
        r=run(ny,nx,tau,eps,arm)
        print(f"eps={eps:4.2f} arm={arm:6s} L2={r['l2']:7.3f}%  u@y_b/U={r['u_at_yb']:+.4f}")
print("\n-- staircase reference (moving bounce at row1, y_b=1.5) --")
ny,nx,tau=24,8,0.8; nu=(tau-0.5)/3.; H=(ny-1.5)-1.5
solid=torch.zeros(ny,nx,dtype=DT); solid[0,:]=1.0
r1=torch.zeros(ny,nx,dtype=DT); r1[1,:]=1.0
top=torch.zeros(ny,nx,dtype=DT); top[-1,:]=1.0
f0=R.rest_field(ny,nx); tables=SH.tables_for(f0)
layers=[SH.stationary_bounce_layer("s0",solid),SH.moving_bounce_layer("lid",r1,(0.02,0.0)),SH.stationary_bounce_layer("t",top)]
op=SH.masked_step(tau=tau,layers=layers,tables=tables)
with torch.no_grad(): st,_=SH.solve_with_hooks({"f":f0.clone()},R.steps_for_couette(H,nu),operator=op)
rho,ux,_=R.macroscopic(st["f"]); u=ux.numpy()[:,0]
ys=np.arange(2,ny-1,dtype=float); ref=0.02*((ny-1.5)-ys)/H
print(f"L2={100*np.linalg.norm(u[2:ny-1]-ref)/np.linalg.norm(ref):.4f}%  u@1.5/U={np.polyfit(ys,u[2:ny-1],1)[1]+np.polyfit(ys,u[2:ny-1],1)[0]*1.5:+.4f}")