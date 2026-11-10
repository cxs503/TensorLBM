import sys; sys.path.insert(0,'src')
import numpy as np, torch
from tensorlbm.elbm import entropic_equilibrium_gauss_d3q27, _axis_factors, gauss_entropy
from tensorlbm.d3q27 import macroscopic27, W_EXACT64 as W27, C
u=0.1
g=_axis_factors(torch.tensor(u,dtype=torch.float64),dtype=torch.float64)
print("factors", [float(x) for x in g])
feq = entropic_equilibrium_gauss_d3q27(torch.ones(1,1,1,dtype=torch.float64), torch.tensor(u,dtype=torch.float64).view(1,1,1), torch.zeros(1,1,1,dtype=torch.float64), torch.zeros(1,1,1,dtype=torch.float64))
print("feq q=0..6", [float(x) for x in feq[:7,0,0,0]])
print("sum w*g", float((W27*feq[:,0,0,0]).sum()))
r,ux,uy,uz = macroscopic27(feq)
print("moments", float(r), float(ux), float(uy))
# analytic product
import math
S=math.sqrt(1+3*u*u); rr=(S-2*u)/(1+u); P=6/(4+rr+1/rr)
print("S,r,P",S,rr,P, "g+", P/rr)
print("check f1 =", 2/27*P/rr*P*P)
print("1D check", 1/6*(P/rr), 1/6*P*rr, 2/3*P, "sum", 1/6*(P/rr)+1/6*P*rr+2/3*P, "mom", 1/6*(P/rr)-1/6*P*rr)
