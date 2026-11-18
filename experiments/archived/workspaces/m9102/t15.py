import sys, time; sys.path.insert(0,'src')
import numpy as np
from tensorlbm.elbm_channel_native import native_channel_run, native_channel_step
for (nx,ny,nz) in ((64,80,64),(96,120,64)):
    for n_iter in (5,):
        t=time.time()
        step_calls = 40
        f = np.ones((27,nz,ny,nx),dtype=np.float32)*0.037; a=np.ones((nz,ny,nx),dtype=np.float32)
        r = native_channel_step(f,a,nx=nx,ny=ny,nz=nz,tau=0.5042,force=2e-6,n_iter=n_iter)
        f, a = r[0], r[1]
        t=time.time()
        for _ in range(step_calls): f,a,_ = native_channel_step(f,a,nx=nx,ny=ny,nz=nz,tau=0.5042,force=2e-6,n_iter=n_iter)
        dt=(time.time()-t)/step_calls
        cells=nx*ny*nz
        print(f"nx,ny,nz={nx},{ny},{nz} n_iter={n_iter}: {dt*1e3:.1f} ms/step -> {cells/dt/1e6:.2f} Mcell/s")
