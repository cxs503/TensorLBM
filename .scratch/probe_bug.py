import torch

from tensorlbm.boundary.boundaries import bounce_back_cells

solid = torch.zeros(8, 8, dtype=torch.bool)
solid[:, 0] = True
g = torch.rand(5, 8, 8, dtype=torch.float64)  # D2Q5 scalar distribution
print("g shape", tuple(g.shape))
try:
    out = bounce_back_cells(g, solid)
    print("bounce_back_cells output shape:", tuple(out.shape))
except Exception as e:
    print("EXCEPTION:", type(e).__name__, e)