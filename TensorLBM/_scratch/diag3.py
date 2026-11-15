import sys; sys.path.insert(0,"src")
import torch
from tensorlbm.d2q9 import stream, C, equilibrium, macroscopic
# pulse test: one cell, direction 1 (c=(1,0))
f = torch.zeros(9,8,8, dtype=torch.float64); f[1,4,4]=1.0
g = stream(f)
idx = (g[1]>0).nonzero()[0].tolist()
print("dir1 c=(1,0) from (4,4) -> ", idx, "  expected [4,5]")
f2 = torch.zeros(9,8,8, dtype=torch.float64); f2[2,4,4]=1.0   # c=(0,1)
print("dir2 c=(0,1) from (4,4) -> ", (stream(f2)[2]>0).nonzero()[0].tolist(), " expected [5,4]")
