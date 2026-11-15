import sys, torch, math
sys.path.insert(0,'src')
torch.set_num_threads(6)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d, C as C3D, W as W3D
from tensorlbm.phasefield.free_energy import force_mu_grad_phi
from tensorlbm.phasefield.operators import laplacian_3d
from tensorlbm.allen_cahn_lbm import free_energy_parameters, step_allen_cahn_field

def droplet(n,R,W):
    z,y,x = torch.meshgrid(torch.arange(n),torch.arange(n),torch.arange(n),indexing='ij')
    r = torch.sqrt((x-n/2+0.5)**2+(y-n/2+0.5)**2+(z-n/2+0.5)**2)
    return torch.tanh((R-r)/W).double()

def run(n=32,R=8,W=4,ratio=10.0,steps=40,tag='V0'):
    p=free_energy_parameters(1e-2,W)
    phi=droplet(n,R,W)
    rhoh=1.0; rhol=rhoh/ratio
    rho_lc = rhol+(rhoh-rhol)*(1+phi)/2
    f=equilibrium3d(rho_lc,torch.zeros_like(phi),torch.zeros_like(phi),torch.zeros_like(phi))
    nu=lambda ph: 0.02+(0.1-0.02)*(1+ph)/2
    for s in range(steps):
        mu = p.beta*phi*(phi*phi-1)-p.kappa*laplacian_3d(phi,boundary='periodic')
        sx,sy,sz = force_mu_grad_phi(phi,mu,boundary='periodic')
        rho_pf = rhol+(rhoh-rhol)*(1+phi)/2
        tau = 3.0*nu(phi)+0.5
        rho_f,ux,uy,uz = macroscopic3d(f)
        if tag=='V0':
            rhoeq=rho_pf; rnorm=rho_pf
        else:
            rhoeq=rho_f; rnorm=rho_pf
        rn=rnorm.clamp(min=1e-12)
        ux2=ux+0.5*sx/rn; uy2=uy+0.5*sy/rn; uz2=uz+0.5*sz/rn
        feq=equilibrium3d(rhoeq,ux2,uy2,uz2)
        c=C3D.double(); w=W3D.double().view(19,1,1,1)
        cu=c[:,0].view(19,1,1,1)*ux2.unsqueeze(0)+c[:,1].view(19,1,1,1)*uy2.unsqueeze(0)+c[:,2].view(19,1,1,1)*uz2.unsqueeze(0)
        cdf=c[:,0].view(19,1,1,1)*sx.unsqueeze(0)+c[:,1].view(19,1,1,1)*sy.unsqueeze(0)+c[:,2].view(19,1,1,1)*sz.unsqueeze(0)
        udf=ux2*sx+uy2*sy+uz2*sz
        guo=w*(1.0-0.5/tau).unsqueeze(0)*(3.0*(cdf-udf.unsqueeze(0))+9.0*cu*cdf)
        fpost=f-(f-feq)/tau.unsqueeze(0)+guo
        # stream
        fnew=torch.empty_like(fpost)
        for q in range(19):
            fnew[q]=torch.roll(fpost[q],shifts=(int(c[q,2]),int(c[q,1]),int(c[q,0])),dims=(0,1,2))
        phi_new=step_allen_cahn_field(phi,beta=p.beta,kappa=p.kappa,mobility=5e-3,u=(ux2,uy2,uz2),boundary='periodic')
        f,phi=fnew,phi_new
        if s<6 or s%10==9:
            rf=micro=None
            print(f"{tag} s={s+1:3d} max|u|={max(float(ux2.abs().max()),float(uy2.abs().max()),float(uz2.abs().max())):.3e} "
                  f"rho_pf=[{float(rho_pf.min()):.3e},{float(rho_pf.max()):.3e}] rho_f=[{float(f.sum(0).min()):.3e},{float(f.sum(0).max()):.3e}] "
                  f"max|F|={max(float(sx.abs().max()),float(sy.abs().max()),float(sz.abs().max())):.3e} phi=[{float(phi.min()):.3f},{float(phi.max()):.3f}]", flush=True)
        if not torch.isfinite(f).all() or not torch.isfinite(phi).all():
            print(tag,"NONFINITE at step",s+1); return
run(tag='V0',steps=20)
