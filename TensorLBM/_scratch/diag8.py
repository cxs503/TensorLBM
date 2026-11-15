import sys; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
n=48
yy,xx=torch.meshgrid(torch.arange(n,dtype=torch.float64),torch.arange(n,dtype=torch.float64),indexing="ij")
rho=(0.1529+1.7761*0.5*(1.0+torch.tanh((yy-n/2)/6.0)))
psi=mp.psi_exp(rho)
dpsi=(torch.roll(psi,-1,0)-torch.roll(psi,1,0))/2
i=torch.argmax(dpsi.abs()); iy,ix=int(i//n),int(i%n)
print("max |dpsi| at y=",iy, " value",float(dpsi[iy,0]))
print("rho col:", [round(float(rho[j,0]),3) for j in range(n)])
print("dpsi col:", ["%.3f"%float(dpsi[j,0]) for j in range(n)])
