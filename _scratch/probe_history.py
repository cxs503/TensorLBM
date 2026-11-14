import sys, math, json
sys.path.insert(0,'src')
import torch
from tensorlbm.phasefield.free_energy_pressure import (
    FreeEnergyBinaryParams, initial_condition, collide_and_stream,
    macroscopic, mean_pressure, periodic_distance, tanh_profile, pressure_tensor_measured)
torch.set_grad_enabled(False); dt=torch.float64

def case(n=64,R=12.,A=0.1,xi=2.0,steps=4000,Gamma=0.05,tau_phi=1.0,every=100):
    kappa=A*xi*xi/2.; p=FreeEnergyBinaryParams(A=A,kappa=kappa,Gamma=Gamma,tau_f=1.0,tau_phi=tau_phi)
    gr=torch.arange(n,dtype=dt); c=n/2.
    dx=periodic_distance(gr.view(1,-1).expand(n,n).contiguous(),c,n)
    dy=periodic_distance(gr.view(-1,1).expand(n,n).contiguous(),c,n)
    r=torch.sqrt(dx*dx+dy*dy); phi0=-tanh_profile(r,R,p.xi)
    f,g=initial_condition(phi0,p)
    hist=[]
    for i in range(steps):
        f,g=collide_and_stream(f,g,p)
        if (i+1)%every==0:
            rho,ux,uy,phi=macroscopic(f,g)
            if not torch.isfinite(phi).all(): hist.append((i+1,"BLOWUP")); break
            pr=mean_pressure(f); Pxx,Pyy,Pxy=pressure_tensor_measured(f)
            rin=(r<=R-3*p.xi); rout=(r>=R+3*p.xi) & (r<=R+0.45*n)
            pin=float(pr[rin].mean()); pout=float(pr[rout].mean())
            u=float((ux*ux+uy*uy).sqrt().max())
            # radial pressure anisotropy integral
            rn=torch.where(r>0,r,torch.ones_like(r))
            er_x=torch.where(r>0,dx/rn,torch.zeros_like(dx)); er_y=torch.where(r>0,dy/rn,torch.zeros_like(dy))
            Prr=Pxx*er_x**2+Pyy*er_y**2+2*Pxy*er_x*er_y
            Ptt=Pxx*er_x**2+Pyy*er_y**2-2*Pxy*er_x*er_y
            iso=0.5*(Pxx+Pyy); Ptt=2*iso-Prr
            anis=(Prr-Ptt)
            # solid-angle average of anis over radial bins
            Rb=torch.arange(0.0,n/2,0.25)
            prof=[]
            for rb in Rb:
                m=(r>=rb)&(r<rb+0.25)
                prof.append(float(anis[m].mean()) if int(m.sum())>0 else 0.0)
            prof=torch.tensor(prof,dtype=dt)
            sig_rad=float(prof.sum()*0.25)
            dp_tens=float((prof/(Rb+0.125)).sum()*0.25)
            hist.append(dict(step=i+1,dp=pin-pout,dp_tensor=dp_tens,pin=pin,pout=pout,
                             sigma_cap=(pin-pout)*R,sig_rad=sig_rad,umax=u,
                             sig_analytic=p.sigma,ratio=(pin-pout)*R/p.sigma,ratio_t=dp_tens*R/p.sigma))
    return dict(n=n,R=R,A=A,xi=p.xi,kappa=kappa,sigma=p.sigma,hist=hist)

res=case()
print(json.dumps({k:v for k,v in res.items() if k!='hist'},indent=1),flush=True)
for h in res['hist']: print(json.dumps(h),flush=True)
json.dump(res,open('_scratch/probe_history.json','w'),indent=1)
