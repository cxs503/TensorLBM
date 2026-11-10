import sys; sys.path.insert(0,'src')
import numpy as np
from tensorlbm.elbm_channel_native import native_channel_run
nx,ny,nz=4,21,4
re_tau=5.0; u_tau=0.01
r = native_channel_run(nx=nx,ny=ny,nz=nz,steps=60000,warmup=30000,sample_every=200,re_tau=re_tau,u_tau=u_tau,n_iter=4,noise_amplitude=1e-8,log_every=20000)
H=ny-2; half=H/2; g=u_tau**2/half; nu=u_tau*half/re_tau; tau=0.5+3*nu
y=np.arange(ny)-0.5; u=r.profiles["u"]; ana=g/(2*nu)*y*(H-y)
m=(y>0.5-1e-9)&(y<H-0.5+1e-9)
print("H",H,"half",half,"tau",round(tau,6),"nu",nu)
print("center num/ana", round(float(u[10]),6), round(float(ana[10]),6), "rel err", round(abs(u[10]-ana[10])/ana[10],5))
print("max rel err interior", round(float(np.abs(u[m]-ana[m]).max()/ana[m].max()),5))
print("alpha", r.scalars["alpha_mean"], r.scalars["alpha_min"], r.scalars["alpha_max"])
print("min pop", r.scalars["min_population"], "entropy drop max", r.scalars["entropy_drop_max"], "finite", r.scalars["finite"])
