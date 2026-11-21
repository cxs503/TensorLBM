import sys; sys.path.insert(0,'src')
import torch
from tensorlbm.elbm import _axis_factors
from tensorlbm.d3q27 import C, W_EXACT64 as W27
dt=torch.float64
u = torch.full((1,1,1), 0.1, dtype=dt)
g = _axis_factors(u, dtype=dt)
c = C.to(dtype=torch.int64)
def sel(factors, comp):
    cv = comp.view(27,1,1,1)
    return torch.where(cv>0, factors[0], torch.where(cv==0, factors[1], factors[2]))
fx=sel(g,c[:,0]); fy=sel(g,c[:,1]); fz=sel(g,c[:,2])
print("fx[:5]", fx[:5,0,0,0], "fy[:5]", fy[:5,0,0,0], "fz[:5]", fz[:5,0,0,0])
prod = fx*fy*fz
print("prod[:5]", prod[:5,0,0,0])
print("w*prod :", [(W27[q]*prod[q,0,0,0]).item() for q in range(5)])
print("mass", (W27.view(27,1,1,1)*prod).sum().item())
