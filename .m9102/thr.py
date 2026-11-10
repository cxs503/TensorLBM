import sys, time, os; sys.path.insert(0,'src')
import numpy as np
from tensorlbm.elbm_channel_native import native_channel_step
nx,ny,nz=96,120,64
f = np.ones((27,nz,ny,nx),dtype=np.float32)*0.037; a=np.ones((nz,ny,nx),dtype=np.float32)
n_iter=int(os.environ.get("NI","5"))
r = native_channel_step(f,a,nx=nx,ny=ny,nz=nz,tau=0.5042,force=2e-6,n_iter=n_iter)
f1,a1 = r[0],r[1]
n=int(os.environ.get("NS","6")); t=time.time()
for _ in range(n): f1,a1,_ = native_channel_step(f1,a1,nx=nx,ny=ny,nz=nz,tau=0.5042,force=2e-6,n_iter=n_iter)
dt=(time.time()-t)/n; cells=nx*ny*nz
print(f"OMP={os.environ.get('OMP_NUM_THREADS','default')} n_iter={n_iter}: {dt*1e3:7.1f} ms/step -> {cells/dt/1e6:6.2f} Mcell/s")
