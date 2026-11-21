import sys; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
import tensorlbm.multiphase as mp
from tensorlbm.d2q9 import equilibrium
from sc_proto import sc_force_fixed
n=48
yy,xx=torch.meshgrid(torch.arange(n,dtype=torch.float64),torch.arange(n,dtype=torch.float64),indexing="ij")
rho=(0.1529+1.7761*0.5*(1.0+torch.tanh((yy-n/2)/6.0)))
psi=mp.psi_exp(rho)
sx,sy=mp._sc_neighbor_sum(psi)
dpsi=(torch.roll(psi,-1,0)-torch.roll(psi,1,0))/2
print("max|psi'|=",float(dpsi.abs().max())," max|sy|=",float(sy.abs().max()), " ratio sy/(psi'/3)=",float((sy[10:38]/(dpsi[10:38]/3)).min()),float((sy[10:38]/(dpsi[10:38]/3)).max()))
fx,fy=sc_force_fixed(rho,5.0)
i=torch.argmax(fy.abs()); iy,ix=int(i//n),int(i%n)
print("Fmax at y=",iy," fy=",float(fy[iy,ix])," psi=",float(psi[iy,ix])," sy=",float(sy[iy,ix]), " rhs 5*psi*sy=",float(5*psi[iy,ix]*sy[iy,ix]))
print("col profile rho:", [round(float(rho[j,0]),3) for j in range(0,48,3)])
print("col profile fy :", ["%.3f"%float(fy[j,0]) for j in range(0,48,3)])
