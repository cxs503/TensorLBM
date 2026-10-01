"""验证: torch.gather 大张量在 SDAA 的真实 GPU 耗时。"""
import sys, time
sys.path.insert(0, 'src')
import torch

dev = torch.device('sdaa:0')
q, n = 27, 20000
populations = torch.randn(q, n, device=dev)
idx = torch.randint(0, n, (q, n), device=dev)

# warmup
torch.gather(populations, 1, idx)
torch.sdaa.synchronize()

# 计时 (含 GPU 同步 = 真实 GPU 时间)
t0 = time.time()
for _ in range(10):
    out = torch.gather(populations, 1, idx)
torch.sdaa.synchronize()
print(f'torch.gather (27x20000): {(time.time()-t0)/10:.4f}s/次', flush=True)

# 对比: 27 次一维 gather
t0 = time.time()
for _ in range(10):
    for d in range(q):
        out_d = populations[d][idx[d]]
torch.sdaa.synchronize()
print(f'27x 一维 gather: {(time.time()-t0)/10:.4f}s/次', flush=True)

# 对比: roll (16卡 coarse 用的)
t0 = time.time()
for _ in range(10):
    out = torch.roll(populations, 1, dims=1)
torch.sdaa.synchronize()
print(f'torch.roll: {(time.time()-t0)/10:.4f}s/次', flush=True)
