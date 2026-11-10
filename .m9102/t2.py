import sys; sys.path.insert(0,'src')
import torch
from tensorlbm.elbm import _axis_factors
from tensorlbm.d3q27 import C
dt=torch.float64
rho=torch.ones(2,3,4,dtype=dt); u=0.1*torch.ones(2,3,4,dtype=dt)
gx=_axis_factors(u,dtype=dt); print([g.shape for g in gx])
ix=(C[:,0].view(27,1,1,1)+1).expand(27,2,3,4)
print(ix.shape, torch.stack(gx)[ix].shape)
