import sys, math
sys.path.insert(0,'src')
import torch
from tensorlbm.phasefield.free_energy_pressure import *

torch.set_grad_enabled(False); dtype=torch.float64
def go(n,R,xi,steps=60,Gamma=0.05,tau_phi=1.0,A=0.1):
    kappa=A*xi*xi/2
    p=FreeEnergyBinaryParams(A=A,kappa=kappa,Gamma=Gamma,tau_f=1.0,tau_phi=tau_phi)
    grid=torch.arange(n,dtype=dtype); c=n/2.
    dx=periodic_distance(grid.view(1,-1).expand(n,n).contiguous(),c,n)
    dy=periodic_distance(grid.view(-1,1).expand(n,n).contiguous(),c,n)
    r=torch.sqrt(dx*dx+dy*dy); phi0=-tanh_profile(r,R,p.xi)
    gx,gy,lap=grad_lap(phi0); mu=chemical_potential(phi0,lap,A,kappa)
    print("xi",p.xi,"mu range",float(mu.min()),float(mu.max()),"lap range",float(lap.min()),float(lap.max()),flush=True)
    f,g=initial_condition(phi0,p)
    print("g range",float(g.min()),float(g.max()),"f range",float(f.min()),float(f.max()),flush=True)
    for i in range(steps):
        f,g=collide_and_stream(f,g,p)
        if (i+1)%10==0:
            rho,ux,uy,phi=macroscopic(f,g)
            print(i+1,"phi",round(float(phi.abs().max()),3),"u",round(float((ux*ux+uy*uy).sqrt().max()),5),
                  "rho",round(float(rho.min()),3),round(float(rho.max()),3),"f",round(float(f.abs().max()),3),"g",round(float(g.abs().max()),3),flush=True)
go(64,14.,3.,steps=40)
