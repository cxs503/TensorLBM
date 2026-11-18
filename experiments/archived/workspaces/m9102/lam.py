import sys; sys.path.insert(0,'src')
import numpy as np
from tensorlbm.elbm_channel_native import native_channel_run
nx,ny,nz=6,41,6
re_tau=20.0; u_tau=0.01
r = native_channel_run(nx=nx,ny=ny,nz=nz,steps=8000,warmup=4000,sample_every=50,re_tau=re_tau,u_tau=u_tau,n_iter=4,noise_amplitude=1e-6,log_every=2000)
H = ny-1; half=H/2; g=u_tau**2/half; nu=u_tau*half/re_tau; tau=0.5+3*nu
y = np.arange(ny)+0.5-0.5
u = r.profiles["u"]
ana = g/(2*nu)*y*(H-y)
m = (y>0)&(y<H)
print("tau", tau, "nu", nu, "g", g)
print("u_tau_balance", np.sqrt(g*half), "u_tau_input", u_tau)
print("rel err vs Poiseuille (interior):", float(np.abs(u[m]-ana[m]).max()/np.abs(ana).max()))
print("center num/ana", u[m].max(), ana[m].max())
# wall shear from numerical derivative
dy=y[1]-y[0]
slope=(u[2]-u[1])/dy   # between y=1 and y=2 (wall at 0.5)
print("nu measured from slope:", u_tau**2/slope, "vs", nu)
print("alpha mean/min/max", r.scalars["alpha_mean"], r.scalars["alpha_min"], r.scalars["alpha_max"])
print("min pop", r.scalars["min_population"], "finite", r.scalars["finite"], "H drop max", r.scalars["entropy_drop_max"], "samples", r.scalars["samples"])
