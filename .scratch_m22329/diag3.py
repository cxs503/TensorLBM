import sys, torch
sys.path.insert(0,'src')
torch.set_num_threads(6)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C3D, W as W3D
from tensorlbm.phasefield.free_energy import force_mu_grad_phi
from tensorlbm.phasefield.operators import laplacian_3d
from tensorlbm.allen_cahn_lbm import free_energy_parameters, step_allen_cahn_field
n,R,W=32,8.0,4.0
z,y,x=torch.meshgrid(torch.arange(n),torch.arange(n),torch.arange(n),indexing='ij')
r=torch.sqrt((x-n/2+0.5)**2+(y-n/2+0.5)**2+(z-n/2+0.5)**2)
phi0=torch.tanh((R-r)/W).double()
p=free_energy_parameters(1e-2,W)
rhoh,rhol=1.0,0.1
nu=lambda ph:0.02+0.08*(1+ph)/2
f0=equilibrium3d(rhol+(rhoh-rhol)*(1+phi0)/2,torch.zeros_like(phi0),torch.zeros_like(phi0),torch.zeros_like(phi0))
c=C3D.double(); w=W3D.double().view(19,1,1,1)
def run(tag, frozen=False, rhoeq_pop=False, shift_ac=True, steps=30):
    phi=phi0.clone(); f=f0.clone()
    for s in range(steps):
        mu=p.beta*phi*(phi*phi-1)-p.kappa*laplacian_3d(phi,boundary='periodic')
        sx,sy,sz=force_mu_grad_phi(phi,mu,boundary='periodic')
        rho_pf=rhol+(rhoh-rhol)*(1+phi)/2; tau=3.0*nu(phi)+0.5
        rho_f,ux,uy,uz=macroscopic3d(f)
        rn=rho_pf.clamp(min=1e-12)
        ux2=ux+0.5*sx/rn; uy2=uy+0.5*sy/rn; uz2=uz+0.5*sz/rn
        feq=equilibrium3d(rho_f if rhoeq_pop else rho_pf,ux2,uy2,uz2)
        cu=c[:,0].view(19,1,1,1)*ux2.unsqueeze(0)+c[:,1].view(19,1,1,1)*uy2.unsqueeze(0)+c[:,2].view(19,1,1,1)*uz2.unsqueeze(0)
        cdf=c[:,0].view(19,1,1,1)*sx.unsqueeze(0)+c[:,1].view(19,1,1,1)*sy.unsqueeze(0)+c[:,2].view(19,1,1,1)*sz.unsqueeze(0)
        udf=ux2*sx+uy2*sy+uz2*sz
        fpost=f-(f-feq)/tau.unsqueeze(0)+w*(1.0-0.5/tau).unsqueeze(0)*(3.0*(cdf-udf.unsqueeze(0))+9.0*cu*cdf)
        f=torch.stack([torch.roll(fpost[q],shifts=(int(c[q,2]),int(c[q,1]),int(c[q,0])),dims=(0,1,2)) for q in range(19)])
        uac=(ux2,uy2,uz2) if shift_ac else (ux,uy,uz)
        if not frozen:
            phi=step_allen_cahn_field(phi,beta=p.beta,kappa=p.kappa,mobility=5e-3,u=uac,boundary='periodic')
        rf,ax,ay,az=macroscopic3d(f)
        if s<4 or s%10==9:
            print(f"{tag} s={s+1:3d} maxu={max(float(ux2.abs().max()),float(uy2.abs().max()),float(uz2.abs().max())):.3e} "
                  f"maxu_raw={max(float(ux.abs().max()),float(uy.abs().max()),float(uz.abs().max())):.3e} "
                  f"sum_mom={float((rf*ax).sum()):.3e} rhof=[{float(rf.min()):.4f},{float(rf.max()):.4f}] phic=[{float(phi[n//2,n//2,n//2]):.4f}]",flush=True)
        if not torch.isfinite(f).all(): print(tag,"NONFINITE",s+1); return
run('FROZEN    ',frozen=True,steps=30)
run('V0-coupled',frozen=False,steps=20)
run('V1-rho=pop',frozen=False,rhoeq_pop=True,steps=20)
run('V2-AC-u_raw',frozen=False,shift_ac=False,steps=20)
