import sys, math, json, itertools
sys.path.insert(0,'src')
import torch
from tensorlbm.phasefield.free_energy_pressure import (
    FreeEnergyBinaryParams, initial_condition, collide_and_stream,
    macroscopic, mean_pressure, periodic_distance, tanh_profile)
torch.set_grad_enabled(False); dt=torch.float64

def trial(A,xi,n=64,R=12.,steps=200,Gamma=0.05,tau_phi=1.0,tau_f=1.0):
    kappa=A*xi*xi/2.0
    p=FreeEnergyBinaryParams(A=A,kappa=kappa,Gamma=Gamma,tau_f=tau_f,tau_phi=tau_phi)
    gr=torch.arange(n,dtype=dt); c=n/2.
    dx=periodic_distance(gr.view(1,-1).expand(n,n).contiguous(),c,n)
    dy=periodic_distance(gr.view(-1,1).expand(n,n).contiguous(),c,n)
    r=torch.sqrt(dx*dx+dy*dy); phi0=-tanh_profile(r,R,p.xi)
    f,g=initial_condition(phi0,p)
    umax0=None; res=dict(A=A,xi=p.xi,kappa=round(kappa,4),sigma=round(p.sigma,5),
        M=round(p.mobility,5),Gamma=Gamma,tau_phi=tau_phi,stable=True)
    for i in range(steps):
        f,g=collide_and_stream(f,g,p)
        if (i+1)%25==0:
            rho,ux,uy,phi=macroscopic(f,g)
            ok=bool(torch.isfinite(phi).all() and torch.isfinite(f).all())
            u=float((ux*ux+uy*uy).sqrt().max()); pm=float(phi.abs().max())
            if not ok: res.update(stable=False, blew_at=i+1); return res
            if umax0 is None: umax0=u
            res["u_end"]=round(u,6); res["phi_end"]=round(pm,4)
            if u>0.05: res.update(stable=False, blew_at=i+1, u_end=round(u,6)); return res
    return res

out=[]
for (A,xi) in [(0.1,2.0),(0.1,2.5),(0.1,3.0),(0.05,3.0),(0.02,3.0),(0.02,4.0),(0.1,1.5),(0.2,2.0)]:
    r=trial(A,xi); out.append(r); print(json.dumps(r),flush=True)
print("=== mobility scan at A=0.1, xi=3 ===")
for (G,tp) in [(0.05,1.0),(0.02,1.0),(0.05,0.8),(0.01,1.0)]:
    r=trial(0.1,3.0,Gamma=G,tau_phi=tp); out.append(r); print(json.dumps(r),flush=True)
json.dump(out,open('_scratch/scan_stability.json','w'),indent=1)
