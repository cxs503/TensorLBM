"""验证: torch.nonzero 大布尔张量在 SDAA 的速度 (ghost 分支用)。"""
import sys, time
sys.path.insert(0, 'src')
import torch

dev = torch.device('sdaa:0')
q, n = 27, 20000
mask = torch.rand(q, n, device=dev) > 0.9  # ~10% True

# nonzero
t0 = time.time()
rows = torch.nonzero(mask, as_tuple=False)
torch.sdaa.synchronize()
print(f'nonzero (27x20000, ~10%): {time.time()-t0:.3f}s, rows={tuple(rows.shape)}', flush=True)

# 对比: 无 nonzero 的批量方案 (argwhere 一样)
t0 = time.time()
idx = mask.nonzero()
torch.sdaa.synchronize()
print(f'.nonzero(): {time.time()-t0:.3f}s', flush=True)

# 更大: 真实 ghost 可能 50% 
mask2 = torch.rand(q, n, device=dev) > 0.5
t0 = time.time()
rows2 = torch.nonzero(mask2, as_tuple=False)
torch.sdaa.synchronize()
print(f'nonzero (50%): {time.time()-t0:.3f}s, rows={tuple(rows2.shape)}', flush=True)
