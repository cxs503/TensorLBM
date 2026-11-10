import sys; sys.path.insert(0,'src')
import numpy as np
from tensorlbm.elbm_channel_native import native_channel_run, native_channel_step
nx,ny,nz=4,21,4
re_tau=5.0; u_tau=0.01
r = native_channel_run(nx=nx,ny=ny,nz=nz,steps=40000,warmup=20000,sample_every=100,re_tau=re_tau,u_tau=u_tau,n_iter=4,noise_amplitude=1e-8,log_every=10000)
H=ny-1; half=H/2; g=u_tau**2/half; nu=u_tau*half/re_tau; tau=0.5+3*nu
y=np.arange(ny)+0.5-0.5
u=r.profiles["u"]; ana=g/(2*nu)*y*(H-y)
m=(y>0)&(y<H)
print("tau",tau,"nu",nu)
print("center num/ana", u[m].max(), ana[m].max(), "rel", abs(u[m].max()-ana[m].max())/ana[m].max())
print("max rel err interior", float(np.abs(u[m]-ana[m]).max()/ana[m].max()))
print("alpha mean/min/max", r.scalars["alpha_mean"], r.scalars["alpha_min"], r.scalars["alpha_max"])
# alpha profile
import numpy as np
f = None
