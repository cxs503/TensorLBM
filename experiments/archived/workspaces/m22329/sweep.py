import sys, torch, math
sys.path.insert(0,'src')
torch.set_num_threads(6)
from tensorlbm.d3q19 import C as C3D, W as W3D
from tensorlbm.phasefield.ac_csf_force_chain import (chemical_potential_isotropic, surface_tension_force,
    pressure_force, second_moment_neq, viscous_force, _stream, _guo_source,
    density_from_phase_field, relaxation_time_from_phase_field)
from tensorlbm.allen_cahn_lbm import free_energy_parameters, step_allen_cahn_field
def droplet(n,R,W):
    a=torch.arange(n,dtype=torch.float64); z,y,x=torch.meshgrid(a,a,a,indexing='ij')
    r=torch.sqrt((z-n/2+0.5)**2+(y-n/2+0.5)**2+(x-n/2+0.5)**2)
    return torch.tanh((R-r)/W)
def eq_dev(rho_dev,ux,uy,uz,rho0=1.0):
    c=C3D.double(); w=W3D.double().view(19,1,1,1)
    cu=c[:,0].view(19,1,1,1)*ux.unsqueeze(0)+c[:,1].view(19,1,1,1)*uy.unsqueeze(0)+c[:,2].view(19,1,1,1)*uz.unsqueeze(0)
    usq=ux*ux+uy*uy+uz*uz
    return w*(rho_dev.unsqueeze(0)+rho0*(3*cu+4.5*cu*cu-1.5*usq.unsqueeze(0)))
def run(variant,ratio=10.0,n=32,R=10.,W=3.,steps=300,sigma=1e-2,terms=('st',),nu=0.05,mob=5e-3,verbose=True):
    rhoh=1.0; rhol=1.0/ratio; p=free_energy_parameters(sigma,W)
    phi=droplet(n,R,W); rho=density_from_phase_field(phi,rhoh,rhol); tau=relaxation_time_from_phase_field(phi,nu,nu)
    f=eq_dev(torch.zeros_like(phi),torch.zeros_like(phi),torch.zeros_like(phi),torch.zeros_like(phi))
    c=C3D.double()
    for s in range(steps):
        rho_dev=f.sum(0); mom=[(f*c[:,a].view(19,1,1,1)).sum(0) for a in range(3)]
        if variant=='A':   us=[mom[a] for a in range(3)]; rs=torch.ones_like(phi)
        else:              rs=rho.clamp(min=1e-12); us=[mom[a]/rs for a in range(3)]
        mu=chemical_potential_isotropic(phi,p.beta,p.kappa); st=surface_tension_force(phi,mu)
        F=[st[i] for i in range(3)]
        if 'p' in terms:
            fp=pressure_force(phi,rho_dev,rhoh,rhol); F=[F[i]+fp[i] for i in range(3)]
        if variant=='C': feq=eq_dev(rho_dev,us[0],us[1],us[2],rho0=None) # placeholder
        else: feq=eq_dev(rho_dev,us[0],us[1],us[2])
        if 'v' in terms:
            fv=viscous_force(phi,second_moment_neq(f-feq),tau,rhoh,rhol); F=[F[i]+fv[i] for i in range(3)]
        ue=[us[a]+0.5*F[a]/rs for a in range(3)]
        feq=eq_dev(rho_dev,ue[0],ue[1],ue[2])
        fpost=f-(f-feq)/tau.unsqueeze(0)+_guo_source(ue[0],ue[1],ue[2],F[0],F[1],F[2],tau)
        f=_stream(fpost)
        phi=step_allen_cahn_field(phi,beta=p.beta,kappa=p.kappa,mobility=mob,u=(ue[0],ue[1],ue[2]),boundary='periodic')
        if not (torch.isfinite(f).all() and torch.isfinite(phi).all()):
            print(f"{variant} {terms} ratio={ratio} NONFINITE at {s+1}"); return
        if verbose and (s+1)%100==0:
            ins=phi>0.9; out=phi<-0.9
            dp=(float(rho_dev[ins].mean())-float(rho_dev[out].mean()))/3 if bool(ins.any()) else float('nan')
            print(f"{variant} {terms} s={s+1} maxu={max(float(x.abs().max()) for x in ue):.3e} dp={dp:.3e} ref={sigma/R:.3e} err={abs(dp-sigma/R)/(sigma/R):.2%}",flush=True)

for ratio in (10.0,100.0):
    for t in (('st',),('st','p','v')):
        run('A',ratio=ratio,terms=t,steps=800,nu=0.1)
