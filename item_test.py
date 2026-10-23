"""验证: .item() 循环 vs 批量 GPU 操作在 SDAA 上的速度。"""
import sys, time
sys.path.insert(0, 'src')
import torch

dev = torch.device('sdaa:0')
n = 1000
populations = torch.randn(19, n, device=dev)
mask = (torch.rand(19, n, device=dev) > 0.5).bool()

# 模拟 _apply_population_total 的 .item() 循环
t0 = time.time()
total_items = 0
for direction in range(19):
    selected = mask[direction]
    count = int(selected.sum().item())
    if count == 0:
        continue
    values = populations[direction, selected]
    inventory = values.sum()
    factor = (inventory / inventory.clamp_min(1e-30))
    lim = abs(float(factor.item()))
    total_items += lim
torch.sdaa.synchronize()
t_item = time.time() - t0
print(f'.item() 循环: {t_item:.3f}s', flush=True)

# 批量版本 (无 .item())
t0 = time.time()
counts = mask.sum(dim=1)
inventories = (populations * mask).sum(dim=1)
factors = inventories / inventories.clamp_min(1e-30)
lims = factors.abs()
total_batch = float(lims.sum().item())
torch.sdaa.synchronize()
t_batch = time.time() - t0
print(f'批量 GPU: {t_batch:.3f}s', flush=True)
print(f'加速比: {t_item/max(t_batch,1e-9):.1f}x', flush=True)
