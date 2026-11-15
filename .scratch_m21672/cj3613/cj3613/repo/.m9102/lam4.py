import sys; sys.path.insert(0,'src')
import numpy as np
from tensorlbm.elbm_channel_native import native_channel_step
nx,ny,nz=4,21,4
re_tau=5.0; u_tau=0.01
H=ny-2; half=H/2; g=u_tau**2/half; nu=u_tau*half/re_tau; tau=0.5+3*nu
# initialise like the C driver: law of wall
z=np.zeros((27,nz,ny,nx),dtype=np.float32)
import math
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27
import torch
y=np.arange(ny)-0.5
yw=np.minimum(y, H-y); yp=np.maximum(yw*re_tau/half,1e-6)
up=np.where(yp<11.4, yp, np.log(yp)/0.41+5.2)
ux=u_tau*up; ux[0]=0; ux[-1]=0
f0 = entropic_equilibrium_gauss_d3q27(torch.ones(nz,ny,nx,dtype=torch.float32), torch.tensor(ux,dtype=torch.float32).view(1,ny,1).expand(nz,ny,nx).contiguous(), torch.zeros(nz,ny,nx), torch.zeros(nz,ny,nx)).numpy()
alpha=np.ones((nz,ny,nx),dtype=np.float32)
f=f0
for step in range(1,30001):
    f,alpha,st = native_channel_step(f,alpha,nx=nx,ny=ny,nz=nz,tau=tau,force=g,n_iter=4)
    if step % 10000 == 0:
        # profile
        rho = f.sum(0); uxf = (f*np.arange(27).reshape(27,1,1,1)*0).sum(0)  # dummy
        CX=np.array([0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1],dtype=np.float32)
        ux_num = (f*CX.reshape(27,1,1,1)).sum(0)/rho + 0.5*g
        prof = ux_num.mean(axis=(0,2)); ap = alpha.mean(axis=(0,2))
        print(f"step {step}: ucenter={prof[10]:.5f} u(y=1)={prof[1]:.5f} u(y=2)={prof[2]:.5f}")
        print("   alpha rows 1..19:", np.round(ap[1:20],3))
u_ana = g/(2*nu)*y*(H-y)
print("analytic center", u_ana[10], "numeric", prof[10])
print("full profile:", np.round(prof,5))
print("analytic     :", np.round(u_ana,5))
print("nu_wall_measured", u_tau**2/((prof[2]-prof[1])/1.0))
