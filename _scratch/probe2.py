import sys, math, time
sys.path.insert(0,'src')
import torch
from tensorlbm.phasefield.free_energy_pressure import (
    FreeEnergyBinaryParams, initial_condition, collide_and_stream,
    macroscopic, mean_pressure, periodic_distance, tanh_profile, pressure_tensor_measured)

torch.set_grad_enabled(False); dtype=torch.float64

def run(n,R,xi,steps,A=0.1):
    kappa=A*xi*xi/2.0
    p=FreeEnergyBinaryParams(A=A,kappa=kappa,Gamma=0.2,tau_f=1.0,tau_phi=1.5)
    grid=torch.arange(n,dtype=dtype); c=n/2.0
    dx=periodic_distance(grid.view(1,-1).expand(n,n).contiguous(),c,n)
    dy=periodic_distance(grid.view(-1,1).expand(n,n).contiguous(),c,n)
    r=torch.sqrt(dx*dx+dy*dy)
    phi0=-tanh_profile(r,R,p.xi)   # +1 inside
    f,g=initial_condition(phi0,p)
    t0=time.time()
    for _ in range(steps): f,g=collide_and_stream(f,g,p)
    rho,ux,uy,phi=macroscopic(f,g); pr=mean_pressure(f)
    def area(lev): return float((phi>lev).to(dtype).sum())
    R_eq=math.sqrt(area(0.0)/math.pi)
    p_in=float(pr[r<=R-3*p.xi].mean()); p_out=float(pr[r>=R+3*p.xi].mean())
    dp=p_in-p_out; sig=dp*R_eq
    return dict(n=n,R=R,xi=p.xi,sigma=round(p.sigma,6),R_eq=round(R_eq,3),
      dp=dp,sig_cap=sig,err_pct=round(100*(sig-p.sigma)/p.sigma,2),
      umax=float((ux*ux+uy*uy).sqrt().max()), phi_mean=round(float(phi.mean()),4),
      Rfit=round(R_eq,3), wall=round(time.time()-t0,1))

for args in [(64,10.,2.),(64,14.,3.),(64,20.,4.)]:
    for st in (400,800):
        print(st, run(*args,steps=st),flush=True)
