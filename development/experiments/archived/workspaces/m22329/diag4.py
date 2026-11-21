import sys, torch
sys.path.insert(0,'src')
torch.set_num_threads(6)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C3D, W as W3D
from tensorlbm.phasefield.free_energy import force_mu_grad_phi
from tensorlbm.phasefield.operators import laplacian_3d
from tensorlbm.allen_cahn_lbm import free_energy_parameters
def droplet(n,R,W):
    z,y,x=torch.meshgrid(torch.arange(n),torch.arange(n),torch.arange(n),indexing='ij')
    r=torch.sqrt((x-n/2+0.5)**2+(y-n/2+0.5)**2+(z-n/2+0.5)**2)
    return torch.tanh((R-r)/W),r
def run(tag,n,R,W,ratio,rhoeq_pop,steps=60,rho_norm='pf'):
    p=free_energy_parameters(1e-2,W); phi,r=droplet(n,R,W)
    rhoh,rhol=1.0,1.0/ratio
    nu=lambda ph:0.02+0.08*(1+ph)/2
    f=equilibrium3d(rhol+(rhoh-rhol)*(1+phi)/2,torch.zeros_like(phi),torch.zeros_like(phi),torch.zeros_like(phi))
    c=C3D.double(); w=W3D.double().view(19,1,1,1)
    mu=p.beta*phi*(phi*phi-1)-p.kappa*laplacian_3d(phi,boundary='periodic'); sx,sy,sz=force_mu_grad_phi(phi,mu,boundary='periodic'); 
    F0=max(float(sx.abs().max()),float(sy.abs().max()),float(sz.abs().max()))
    print(f"--- {tag}: n={n} R={R} W={W} R/W={R/W} ratio={ratio} max|F0|={F0:.3e} sigma/R={1e-2/R:.3e}")
    for s in range(steps):
        mu=p.beta*phi*(phi*phi-1)-p.kappa*laplacian_3d(phi,boundary='periodic'); sx,sy,sz=force_mu_grad_phi(phi,mu,boundary='periodic')
        rho_pf=rhol+(rhoh-rhol)*(1+phi)/2; tau=3.0*nu(phi)+0.5
        rho_f,ux,uy,uz=macroscopic3d(f)
        rn=(rho_pf if rho_norm=='pf' else rho_f).clamp(min=1e-12)
        u2x=ux+0.5*sx/rn; u2y=uy+0.5*sy/rn; u2z=uz+0.5*sz/rn
        feq=equilibrium3d(rho_f if rhoeq_pop else rho_pf,u2x,u2y,u2z)
        cu=c[:,0].view(19,1,1,1)*u2x.unsqueeze(0)+c[:,1].view(19,1,1,1)*u2y.unsqueeze(0)+c[:,2].view(19,1,1,1)*u2z.unsqueeze(0)
        cdf=c[:,0].view(19,1,1,1)*sx.unsqueeze(0)+c[:,1].view(19,1,1,1)*sy.unsqueeze(0)+c[:,2].view(19,1,1,1)*sz.unsqueeze(0)
        udf=u2x*sx+u2y*sy+u2z*sz
        fpost=f-(f-feq)/tau.unsqueeze(0)+w*(1.0-0.5/tau).unsqueeze(0)*(3.0*(cdf-udf.unsqueeze(0))+9.0*cu*cdf)
        f=torch.stack([torch.roll(fpost[q],shifts=(int(c[q,2]),int(c[q,1]),int(c[q,0])),dims=(0,1,2)) for q in range(19)])
        rf,ax,ay,az=macroscopic3d(f)
        if s in (0,1,2,4,9,19,39,59):
            print(f"  s={s+1:3d} maxu={max(float(u2x.abs().max()),float(u2y.abs().max()),float(u2z.abs().max())):.3e} "
                  f"rhof=[{float(rf.min()):.4f},{float(rf.max()):.4f}] pin={float(rf[(phi>0.9)].mean()) if bool((phi>0.9).any()) else float('nan'):.5f}/{float(rf[phi<-0.9].mean()):.5f} "
                  f"dp={(float(rf[phi>0.9].mean())-float(rf[phi<-0.9].mean()))/3:.3e}",flush=True)
        if not torch.isfinite(f).all(): print("  NONFINITE",s+1); return
run('R1-V0',48,14.0,3.5,1.0,False)
run('R1-V1',48,14.0,3.5,1.0,True)
run('R10-V1',48,14.0,3.5,10.0,True)
