import sys, math, time
sys.path.insert(0,'src')
import torch
from tensorlbm.phasefield.free_energy_pressure import (
    FreeEnergyBinaryParams, initial_condition, collide_and_stream,
    macroscopic, mean_pressure, periodic_distance, tanh_profile)

torch.set_grad_enabled(False); dtype=torch.float64

def run(n,R,xi,steps,A=0.1,Gamma=0.05,tau_phi=1.0):
    kappa=A*xi*xi/2.0
    p=FreeEnergyBinaryParams(A=A,kappa=kappa,Gamma=Gamma,tau_f=1.0,tau_phi=tau_phi)
    grid=torch.arange(n,dtype=dtype); c=n/2.0
    dx=periodic_distance(grid.view(1,-1).expand(n,n).contiguous(),c,n)
    dy=periodic_distance(grid.view(-1,1).expand(n,n).contiguous(),c,n)
    r=torch.sqrt(dx*dx+dy*dy); phi0=-tanh_profile(r,R,p.xi)
    f,g=initial_condition(phi0,p); t0=time.time(); hist=[]
    for i in range(steps):
        f,g=collide_and_stream(f,g,p)
        if (i+1)%100==0:
            rho,ux,uy,phi=macroscopic(f,g); hist.append((i+1,round(float(phi.abs().max()),4),round(float((ux*ux+uy*uy).sqrt().max()),6)))
            if not torch.isfinite(phi).all(): return dict(FAIL=i+1,hist=hist)
    rho,ux,uy,phi=macroscopic(f,g); pr=mean_pressure(f)
    R_eq=math.sqrt(float((phi>0).to(dtype).sum())/math.pi)
    p_in=float(pr[r<=R-3*p.xi].mean()); p_out=float(pr[r>=R+3*p.xi].mean())
    dp=p_in-p_out; sig=dp*R_eq
    return dict(n=n,R=R,xi=p.xi,M=p.mobility,sigma=round(p.sigma,6),R_eq=round(R_eq,3),
      sig_cap=round(sig,6),err_pct=round(100*(sig-p.sigma)/p.sigma,2),
      umax=float((ux*ux+uy*uy).sqrt().max()),hist=hist[-3:],wall=round(time.time()-t0,1))

print(run(64,14.,3.,steps=400),flush=True)
print(run(64,20.,4.,steps=400,Gamma=0.02),flush=True)
