"""验证: 三维高级索引 (octree ghost fill 模式) 是否在 SDAA 卡死。"""
import sys, time
sys.path.insert(0, 'src')
import torch

dev = torch.device('sdaa:0')

# 模拟 _fill_ghost 的 trilinear 采样
# parent_time_state: (q, nz_p, ny_p, nx_p)
nz_p, ny_p, nx_p = 32, 32, 64
q = 27
n_ghost = 5000
parent = torch.randn(q, nz_p, ny_p, nx_p, device=dev)
z0 = torch.randint(0, nz_p, (n_ghost,), device=dev)
y0 = torch.randint(0, ny_p, (n_ghost,), device=dev)
x0 = torch.randint(0, nx_p, (n_ghost,), device=dev)

print(f'parent: {tuple(parent.shape)}, ghost: {n_ghost}', flush=True)

# 三维高级索引 (octree ghost fill 的核心)
t0 = time.time()
sampled = parent[:, z0, y0, x0]
torch.sdaa.synchronize()
print(f'3D advanced index: {time.time()-t0:.3f}s, shape={tuple(sampled.shape)}', flush=True)

# 更大规模
n2 = 50000
z1 = torch.randint(0, nz_p, (n2,), device=dev)
y1 = torch.randint(0, ny_p, (n2,), device=dev)
x1 = torch.randint(0, nx_p, (n2,), device=dev)
t0 = time.time()
sampled2 = parent[:, z1, y1, x1]
torch.sdaa.synchronize()
print(f'3D advanced index (50k): {time.time()-t0:.3f}s, shape={tuple(sampled2.shape)}', flush=True)
