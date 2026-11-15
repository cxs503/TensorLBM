import sys, time, os; sys.path.insert(0,'src')
import torch
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27, solve_entropy_condition, gauss_entropy
from tensorlbm.elbm_channel import stream_channel_d3q27
from tensorlbm.d3q27 import W_EXACT64 as W27, macroscopic27
nt = int(os.environ.get("NT","32")); torch.set_num_threads(nt)
dt = torch.float32
nz,ny,nx = 64,66,64
rho = torch.ones(nz,ny,nx,dtype=dt); ux = 0.05*torch.randn(nz,ny,nx,dtype=dt)
uy = 0.01*torch.randn(nz,ny,nx,dtype=dt); uz = 0.01*torch.randn(nz,ny,nx,dtype=dt)
feq = entropic_equilibrium_gauss_d3q27(rho,ux,uy,uz)
cells = nz*ny*nx
def bench(name, fn, n=3):
    fn(); t=time.time()
    for _ in range(n): out=fn()
    dt_s=(time.time()-t)/n
    print(f"{name:34s} {dt_s*1e3:8.1f} ms  {cells/dt_s/1e6:7.2f} Mcell/s", flush=True)
    return out
print("threads", nt, "cells", cells)
bench("macroscopic", lambda: macroscopic27(feq))
bench("equilibrium closed form", lambda: entropic_equilibrium_gauss_d3q27(rho,ux,uy,uz))
w = W27.to(dtype=dt).view(27,1,1,1)
bench("entropy H", lambda: gauss_entropy(feq, w))
for ni in (2,4,6):
    bench(f"alpha solve n_iter={ni}", lambda ni=ni: solve_entropy_condition(feq,feq,torch.tensor(0.504,dtype=dt),w=w,n_iter=ni))
bench("stream", lambda: stream_channel_d3q27(feq))
