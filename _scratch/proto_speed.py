import sys, time
import numpy as np, torch
sys.path.insert(0,"/srv/tensor/agent/work/m21247/src")
sys.path.insert(0,"/srv/tensor/agent/work/m21247/_scratch")
from tensorlbm.spectral_saffman_taylor import SpectralSaffmanTaylor2D
from proto_varcoef import apply_A, apply_Pinv
torch.set_num_threads(8)

def gmres(sim, b, x0=None, tol=1e-8, restart=20, maxit=60):
    nb=float(torch.linalg.vector_norm(b)); x=torch.zeros_like(b) if x0 is None else x0.clone()
    r=b-apply_A(sim,x); beta=float(torch.linalg.vector_norm(r)); res=beta/nb; it=0
    while it<maxit and res>tol:
        V=[r/beta]; H=np.zeros((restart+1,restart))
        m=0
        for j in range(restart):
            z=apply_Pinv(sim,V[j]); w=apply_A(sim,z)
            for i in range(j+1):
                H[i,j]=float(torch.sum(w*V[i])); w=w-H[i,j]*V[i]
            H[j+1,j]=float(torch.linalg.vector_norm(w))
            if H[j+1,j]<1e-14: m=j+1; break
            V.append(w/H[j+1,j]); m=j+1
        y=np.linalg.lstsq(H[:m+1,:m],np.r_[beta,np.zeros(m)],rcond=None)[0]
        combo=torch.zeros_like(b)
        for j in range(m): combo=combo+float(y[j])*V[j]
        x=x+apply_Pinv(sim,combo); it+=m; r=b-apply_A(sim,x); res=float(torch.linalg.vector_norm(r))/nb
    return x,it,res

sim=SpectralSaffmanTaylor2D(128,256,aspect=2.0,pe=500.0,viscosity_ratio=-1.0,dt=0.1,mode_index=4)
sim.set_concentration(sim.single_mode_ic(x0=500.0,width=4.0,amplitude=1e-3,n_mode=4))
b=-sim.viscosity_ratio*sim.c_y.real
for tol,rest in [(1e-10,40),(1e-8,20),(1e-6,15)]:
    t0=time.time(); psi,it,res=gmres(sim,b,tol=tol,restart=rest); print(f"cold tol={tol} rest={rest}: it={it} res={res:.1e} t={time.time()-t0:.3f}s")
    t0=time.time(); psi2,it2,res2=gmres(sim,b,x0=psi,tol=tol,restart=rest); print(f"  warm: it={it2} res={res2:.1e} t={time.time()-t0:.3f}s")
# picard test
def picard(sim,b,n=50):
    x=torch.zeros_like(b)
    for k in range(n):
        x=apply_Pinv(sim,b-sim.viscosity_ratio*(sim.c_x.real*torch.fft.ifft2(torch.fft.fft2(x)*sim._km*1j).real+sim.c_y.real*torch.fft.ifft2(torch.fft.fft2(x)*sim._kn*1j).real))
    r=b-apply_A(sim,x); return float(torch.linalg.vector_norm(r))/float(torch.linalg.vector_norm(b))
t0=time.time(); print(f"picard 50 iters res={picard(sim,b,50):.2e} t={time.time()-t0:.3f}s")
