import sys, math, time
sys.path.insert(0,'src')
import torch
from tensorlbm.phasefield.free_energy_pressure import (
    FreeEnergyBinaryParams, initial_condition, collide_and_stream,
    macroscopic, mean_pressure, periodic_distance, tanh_profile)

torch.set_grad_enabled(False)
dtype=torch.float64

def run(n,R,xi,sigma_target=None,steps=3000, A=None):
    # choose kappa from xi and A: xi = sqrt(2k/A); sigma = (2/3)sqrt(2kA)
    if A is None: A=0.1
    kappa = A*xi*xi/2.0
    p = FreeEnergyBinaryParams(A=A,kappa=kappa,Gamma=0.2,tau_f=1.0,tau_phi=1.5)
    grid=torch.arange(n,dtype=dtype); c=n/2.0
    dx=periodic_distance(grid.view(1,-1).expand(n,n).contiguous(),c,n)
    dy=periodic_distance(grid.view(-1,1).expand(n,n).contiguous(),c,n)
    r=torch.sqrt(dx*dx+dy*dy)
    phi0=tanh_profile(r,R,p.xi)
    f,g=initial_condition(phi0,p)
    t0=time.time()
    for i in range(steps):
        f,g=collide_and_stream(f,g,p)
    rho,ux,uy,phi=macroscopic(f,g)
    pr=mean_pressure(f)
    shell = (r<=R-3*p.xi); out = (r>=R+3*p.xi)
    p_in=float(pr[shell].mean()); p_out=float(pr[out].mean())
    dp=p_in-p_out
    # measured radius from phi=0 contour area
    V=float((phi>0).to(dtype).sum())
    R_eq=math.sqrt(V/math.pi)
    # interface position better: integrate (1+phi)/2? use phi>0.5 etc. keep simple
    sig_cap = dp*R_eq
    return dict(n=n,R=R,xi=p.xi,kappa=p.kappa,sigma=p.sigma,R_eq=R_eq,dp=dp,
                sig_cap=sig_cap,err_pct=100*(sig_cap-p.sigma)/p.sigma,
                p_in=p_in,p_out=p_out,umax=float((ux*ux+uy*uy).sqrt().max()),
                phi_mean=float(phi.mean()), rho_in=float(rho[shell].mean()),
                rho_out=float(rho[out].mean()), wall=time.time()-t0)

import itertools
for (n,R,xi) in [(64,10.,2.),(64,14.,3.),(96,14.,3.)]:
    print(run(n,R,xi,steps=600),flush=True)
    print(run(n,R,xi,steps=2500 if n<=96 else 1500))
