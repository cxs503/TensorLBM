import sys, time; sys.path.insert(0,'src')
import numpy as np
from tensorlbm.elbm_channel_native import native_channel_run
for (nx,ny,nz,steps) in ((4,13,4,14000),):
    t=time.time()
    r = native_channel_run(nx=nx,ny=ny,nz=nz,steps=steps,warmup=6000,sample_every=200,re_tau=4.0,u_tau=0.01,n_iter=4,noise_amplitude=1e-8,log_every=5000)
    print(nx,ny,nz,"steps",steps,"->",round(time.time()-t,2),"s")
