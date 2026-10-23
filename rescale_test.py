"""验证: rescale_nonequilibrium 是否在 SDAA 卡死 (octree _fill_ghost 卡点)。"""
import sys, time
sys.path.insert(0, 'src')
import torch
from tensorlbm.amr_population_transfer import rescale_nonequilibrium

dev = torch.device('sdaa:0')
q, n_ghost = 27, 5000
sampled = torch.randn(q, 1, 1, n_ghost, device=dev) * 0.01 + 0.03
print(f'input: {tuple(sampled.shape)}', flush=True)

t0 = time.time()
out = rescale_nonequilibrium(
    sampled, tau_source=0.500037, tau_target=0.500009,
    spatial_ratio=0.5, regularize=True,
)
torch.sdaa.synchronize()
print(f'rescale: {time.time()-t0:.3f}s, shape={tuple(out.shape)}', flush=True)

