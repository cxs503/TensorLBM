import sys, ctypes; sys.path.insert(0,'src')
import numpy as np, torch
torch.set_num_threads(4)
from tensorlbm.elbm_channel_native import build_kernel, _library
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27, gauss_entropy, solve_entropy_condition
from tensorlbm.d3q27 import W_EXACT64 as W27, macroscopic27
lib = _library(str(build_kernel(force=True)))
dp = ctypes.POINTER(ctypes.c_double); fp = ctypes.POINTER(ctypes.c_float)
lib.elbm_debug_cell.argtypes=[fp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_double,ctypes.c_double,dp]
lib.elbm_debug_cell.restype=ctypes.c_int
nx,ny,nz = 10,12,9
torch.manual_seed(3)
rho = torch.ones(nz,ny,nx); ux=0.05*torch.randn(nz,ny,nx); uy=0.02*torch.randn(nz,ny,nx); uz=0.02*torch.randn(nz,ny,nx)
f = entropic_equilibrium_gauss_d3q27(rho,ux,uy,uz)
f = torch.clamp(f*(1+0.08*torch.randn_like(f)), min=1e-6)
tau=0.504
zi,yi,xi = 4,6,3
out = np.zeros(9+3*27)
fa = np.ascontiguousarray(f.numpy(), dtype=np.float32)
lib.elbm_debug_cell(fa.ctypes.data_as(fp), nx,ny,nz, zi, yi, xi, tau, 20, 2.0, 1.0, out.ctypes.data_as(dp))
print("C: rho,u =", out[:4], "H0", out[4], "alpha_hi", out[5], "H_hi", out[6], "alpha", out[7], "H1", out[8])
fc = f[:, zi, yi, xi].view(27,1,1,1)
r,u1,u2,u3 = macroscopic27(fc)
feq = entropic_equilibrium_gauss_d3q27(r,u1,u2,u3)
w = W27.view(27,1,1,1)
H0 = gauss_entropy(fc, w)
info = solve_entropy_condition(fc, feq, tau, w=w, n_iter=20)
fst = fc + (info.alpha/tau)*(feq-fc)
print("PY: rho,u =", [float(r), float(u1), float(u2), float(u3)], "H0", float(H0))
print("PY alpha", float(info.alpha), "H1", float(gauss_entropy(fst, w)))
print("feq C vs PY max diff: (see below)")
import numpy as np
print("C feq[:6]", out[9:15])
print("PY feq[:6]", feq[:6,0,0,0].tolist())
print("C f[:6]", out[9+27:9+27+6])
print("PY f[:6]", fc[:6,0,0,0].tolist())
print("max |C feq - PY feq|", float(np.abs(out[9:9+27]-feq[:,0,0,0].numpy()).max()))
