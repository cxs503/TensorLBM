import sys,json,base64,io,math
sys.path.insert(0,'src')
import torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from tensorlbm.phasefield.free_energy_pressure import grad_lap, pressure_tensor_analytic, periodic_distance
from tensorlbm.phasefield.laplace_droplet_calibration import (droplet_phi0, xi_analytic, sigma_analytic,
    polar_pressure_components, radial_anisotropy_profile, tension_integrals)
def rg(N):
    c=0.5*(N-1); g=torch.arange(N,dtype=torch.float64)
    dx=periodic_distance(g.view(1,-1).expand(N,N).contiguous(),c,float(N))
    dy=periodic_distance(g.view(-1,1).expand(N,N).contiguous(),c,float(N))
    return torch.sqrt(dx*dx+dy*dy),dx,dy
rows=[]
for (R,A,k) in [(20.,0.1,0.2),(26.,0.05,0.225),(36.,0.05,0.225)]:
    xi=xi_analytic(A,k); N=128 if R<30 else 160
    phi=droplet_phi0(N,R,xi); rad,dx,dy=rg(N)
    gx,gy,lap=grad_lap(phi); rho=torch.ones_like(phi)
    Pxx,Pyy,Pxy=pressure_tensor_analytic(rho,phi,gx,gy,lap,A,k)
    Prr,Ptt=polar_pressure_components(Pxx,Pyy,Pxy,dx,dy)
    rb,an=radial_anisotropy_profile(Prr,Ptt,rad,bin_width=0.25,r_max=0.5*N-1)
    out=tension_integrals(rb,an,r_lo=R-4*xi,r_hi=R+4*xi,bin_width=0.25)
    s=sigma_analytic(A,k); rows.append(dict(R=R,xi=xi,ratio=out["sigma_integral"]/s,err=out["sigma_integral"]/s-1,
        Rsig=out["R_surface_of_tension"],dp_ratio=out["dp_tensor"]/(s/R),A=A,k=k))
# profile + kernel figure
phi=droplet_phi0(128,20.,2.); rad,dx,dy=rg(128); gx,gy,lap=grad_lap(phi); rho=torch.ones_like(phi)
Pxx,Pyy,Pxy=pressure_tensor_analytic(rho,phi,gx,gy,lap,0.1,0.2); Prr,Ptt=polar_pressure_components(Pxx,Pyy,Pxy,dx,dy)
rb,an=radial_anisotropy_profile(Prr,Ptt,rad,bin_width=0.25,r_max=63.)
rc,pm=__import__('tensorlbm.phasefield.laplace_droplet_calibration',fromlist=['x']).radial_average(phi,rad,bin_width=0.25,r_max=63.)[:2]
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
ax[0].plot(rc.tolist(),pm.tolist(),label=r'$\bar\phi(r)$ measured-bin mean')
ax[0].axhline(0,color='k',lw=.6); ax[0].set_xlabel('r [lu]'); ax[0].set_ylabel(r'$\phi$'); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[0].set_title('synthetic tanh droplet, R=20, xi=2 (N=128)')
ax[1].plot(rb.tolist(),an.tolist(),label=r'$\overline{P_{rr}-P_{tt}}$ (analytic tensor)')
ax[1].set_xlabel('r [lu]'); ax[1].set_ylabel('anisotropy [lu]'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
ax[1].set_title(r'$\int(P_{rr}-P_{tt})dr = \sigma$  (one interface)')
fig.tight_layout(); fig.savefig('_scratch/fig_chain.png',dpi=120)
json.dump(rows,open('_scratch/analytic_chain.json','w'),indent=1)
print(json.dumps(rows,indent=1))
