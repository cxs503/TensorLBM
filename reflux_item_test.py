"""验证: 真实规模的 .item() 循环是否在 SDAA 卡死（模拟 reflux）。"""
import sys, time
sys.path.insert(0, 'src')
import torch

dev = torch.device('sdaa:0')

# 模拟 octree reflux 的真实规模
# L1 block 界面 links: 大约几千个
n_links = 5000
q = 19  # D3Q19 (或 27)
populations = torch.randn(q, n_links, device=dev)
mask = (torch.rand(q, n_links, device=dev) > 0.3).bool()
requested = torch.randn(q, device=dev)

print(f'populations: {tuple(populations.shape)}, n_links={n_links}', flush=True)

# 复刻 _apply_population_total 的原始循环 (每方向 .item())
def apply_orig(populations, mask, requested, max_frac=0.5):
    applied = torch.zeros_like(requested)
    for direction in range(populations.shape[0]):
        selected = mask[direction]
        count = int(selected.sum().item())
        if count == 0:
            continue
        values = populations[direction, selected]
        inventory = values.sum()
        desired = requested[direction]
        factor = desired / inventory.clamp_min(1e-30)
        limited_factor = factor.clamp(-max_frac, max_frac)
        delta = values * limited_factor
        applied[direction] = delta.sum()
    return applied

print('开始原始 .item() 循环...', flush=True)
t0 = time.time()
a1 = apply_orig(populations, mask, requested)
torch.sdaa.synchronize()
print(f'原始循环: {time.time()-t0:.3f}s', flush=True)

# 批量版本
def apply_batch(populations, mask, requested, max_frac=0.5):
    counts = mask.sum(dim=1)
    inventories = (populations * mask).sum(dim=1)
    factors = requested / inventories.clamp_min(1e-30)
    limited = factors.clamp(-max_frac, max_frac)
    # delta per direction: 需要 masked 加权和 —— 用 where 保持形状
    weight = limited.view(q, 1) * mask
    applied = (populations * weight).sum(dim=1)
    return applied

t0 = time.time()
a2 = apply_batch(populations, mask, requested)
torch.sdaa.synchronize()
print(f'批量版: {time.time()-t0:.3f}s', flush=True)
print(f'差异: {(a1-a2).abs().max().item():.2e}', flush=True)

# 更大规模测试 (真实 octree: 可能 5万+ links)
print('\n更大规模 (n_links=50000)...', flush=True)
n2 = 50000
pop2 = torch.randn(q, n2, device=dev)
mask2 = (torch.rand(q, n2, device=dev) > 0.3).bool()
t0 = time.time()
a1b = apply_orig(pop2, mask2, requested)
torch.sdaa.synchronize()
print(f'原始循环 (50k): {time.time()-t0:.3f}s', flush=True)
