import sys; sys.path.insert(0,"src"); sys.path.insert(0,"_scratch")
import torch; torch.set_default_dtype(torch.float64)
from tensorlbm.d2q9 import C, equilibrium, stream
import tensorlbm.multiphase as mp
import refport as rp
from sc_proto import sc_force_fixed
N=64;R=15.0;RHOL=2.1;RHOV=0.15;G=4.7
yy,xx=torch.meshgrid(torch.arange(N,dtype=torch.float64),torch.arange(N,dtype=torch.float64),indexing="ij")
ins=((yy-N/2.)**2+(xx-N/2.)**2)<=R*R
rho=torch.where(ins,torch.tensor(RHOL,dtype=torch.float64),torch.tensor(RHOV,dtype=torch.float64))
f0=equilibrium(rho,torch.zeros_like(rho),torch.zeros_like(rho))
# library arm
r=f0.sum(0); fx,fy=sc_force_fixed(r,G)
fL=mp.collide_guo(f0,fx,fy,1.0)
outL=torch.empty_like(fL)
for i in range(9):
    cx,cy=int(C[i,0]),int(C[i,1])
    outL[i]=fL[i] if (cx==0 and cy==0) else torch.roll(fL[i],shifts=(cy,cx),dims=(0,1))
# port arm
Fx,Fy=rp.sc_forces(r,-G)
outP=rp.push(f0,Fx,Fy,1.0)
print("force max diff", float((fx-Fx).abs().max()), float((fy-Fy).abs().max()))
print("after 1 step: max|diff| per dir", [round(float((outL[i]-outP[i]).abs().max()),12) for i in range(9)])
print("rho after1 L/P:", float(outL.sum(0).min()), float(outP.sum(0).min()), float(outL.sum(0).max()), float(outP.sum(0).max()))
