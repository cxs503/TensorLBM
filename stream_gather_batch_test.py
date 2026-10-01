"""验证: stream_gather 批量化 (27 方向一次 gather) vs 循环 27 次。"""
import sys, time
sys.path.insert(0, 'src')
import torch

dev = torch.device('sdaa:0')
n_leaf = 20000
q = 27

populations = torch.randn(q, n_leaf, device=dev)
# 模拟 neighbor table: (q, n_leaf), 值 = 下游叶子编号 或 -1 (无效)
src = torch.randint(-1, n_leaf, (q, n_leaf), device=dev)
valid = src >= 0

# 原版: 27 方向循环 gather (复刻 stream_gather)
def stream_gather_loop(populations, src, valid):
    q_, n = populations.shape
    out = torch.empty_like(populations)
    for d in range(q_):
        sv = src[d][valid[d]]
        out[d][valid[d]] = populations[d][sv]
    return out

t0 = time.time()
out1 = stream_gather_loop(populations, src, valid)
torch.sdaa.synchronize()
print(f'循环 gather (27 方向): {time.time()-t0:.3f}s', flush=True)

# 批量版: 一次 gather (用 -1 哨兵 + clamp)
def stream_gather_batch(populations, src, valid):
    q_, n = populations.shape
    # 用 where 把无效索引替换为 0, gather 后 mask
    idx = src.clamp(min=0)
    gathered = torch.gather(populations, 1, idx)  # (q, n)
    return torch.where(valid, gathered, torch.zeros_like(gathered))

t0 = time.time()
out2 = stream_gather_batch(populations, src, valid)
torch.sdaa.synchronize()
print(f'批量 gather (1 次): {time.time()-t0:.3f}s', flush=True)
print(f'差异: {(out1-out2).abs().max().item():.2e}', flush=True)
