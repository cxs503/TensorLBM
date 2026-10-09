"""验证: 完整 _fill_ghost trilinear (真实规模) 是否卡死。"""
import sys, time
sys.path.insert(0, 'src')
import torch

dev = torch.device('sdaa:0')

# 真实规模: L1 block ghost 数
# 96x64x64 网格, L1 block 约 1/8 体积, ghost 界面约 2万格
nz_p, ny_p, nx_p = 64, 48, 48  # coarse (父层)
q = 27
n_ghost = 20000

parent = torch.randn(q, nz_p, ny_p, nx_p, device=dev)
z0 = torch.randint(0, nz_p - 1, (n_ghost,), device=dev)
y0 = torch.randint(0, ny_p - 1, (n_ghost,), device=dev)
x0 = torch.randint(0, nx_p - 1, (n_ghost,), device=dev)
z1, y1, x1 = z0 + 1, y0 + 1, x0 + 1
wx = (torch.rand(n_ghost, device=dev) * 0.5 + 0.25).clamp(0, 1)
wy = (torch.rand(n_ghost, device=dev) * 0.5 + 0.25).clamp(0, 1)
wz = (torch.rand(n_ghost, device=dev) * 0.5 + 0.25).clamp(0, 1)

print(f'parent: {tuple(parent.shape)}, ghost: {n_ghost}', flush=True)

# 完整 trilinear (复刻 _fill_ghost)
t0 = time.time()
v00 = torch.lerp(parent[:, z0, y0, x0], parent[:, z0, y0, x1], wx)
v01 = torch.lerp(parent[:, z0, y1, x0], parent[:, z0, y1, x1], wx)
v10 = torch.lerp(parent[:, z1, y0, x0], parent[:, z1, y0, x1], wx)
v11 = torch.lerp(parent[:, z1, y1, x0], parent[:, z1, y1, x1], wx)
sampled = torch.lerp(torch.lerp(v00, v01, wy), torch.lerp(v10, v11, wy), wz)
torch.sdaa.synchronize()
print(f'trilinear: {time.time()-t0:.3f}s, shape={tuple(sampled.shape)}', flush=True)

# 写回 fine_f
fine_f = torch.randn(q, nz_p * 2, ny_p * 2, nx_p * 2, device=dev)
target_flat = torch.randint(0, fine_f.shape[1] * fine_f.shape[2] * fine_f.shape[3], (n_ghost,), device=dev)
t0 = time.time()
fine_f.reshape(q, -1)[:, target_flat] = sampled
torch.sdaa.synchronize()
print(f'writeback: {time.time()-t0:.3f}s', flush=True)
