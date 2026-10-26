"""验证 SDAA 上 gather (advanced indexing) 是否卡死——octree 卡死根因。"""
import sys, time
sys.path.insert(0, 'src')
import torch

dev = torch.device('sdaa:0')

# 模拟 octree stream_gather: populations[d, src] 高级索引
n_leaf = 1000
q = 27
populations = torch.randn(q, n_leaf, device=dev)
src = torch.randint(-1, n_leaf, (n_leaf,), device=dev)
src[src < 0] = n_leaf - 1  # 有效索引

print(f'populations: {tuple(populations.shape)}, src: {tuple(src.shape)}', flush=True)

# 测试1: 基本 advanced indexing (gather)
t0 = time.time()
out1 = populations[:, src]
torch.sdaa.synchronize()
print(f'basic gather: {time.time()-t0:.3f}s OK', flush=True)

# 测试2: masked advanced indexing (octree 的实际模式: out[d, valid] = populations[d, src[valid]])
valid = src >= 0
t0 = time.time()
out2 = torch.empty_like(populations)
out2[:, valid] = populations[:, src[valid]]
torch.sdaa.synchronize()
print(f'masked gather: {time.time()-t0:.3f}s OK', flush=True)

# 测试3: 循环 gather (替代方案)
t0 = time.time()
out3 = torch.empty_like(populations)
for d in range(q):
    out3[d] = populations[d, src]
torch.sdaa.synchronize()
print(f'loop gather: {time.time()-t0:.3f}s OK', flush=True)

# 验证结果一致
print(f'max diff (gather vs loop): {(out1 - out3).abs().max().item():.2e}', flush=True)
