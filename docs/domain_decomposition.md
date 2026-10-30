# M4·P2 跨卡域分解框架固化

## 概述

本模块实现了纯 PyTorch 的跨卡域分解框架，支持 LBM 模拟的 slab 分解和 halo exchange。框架对齐 `docs/triton_distributed_notes.md` 中的工程经验，包括：

- **NCCL contiguous**: 所有 send/recv 操作使用 `.contiguous()` 缓冲区
- **Q padding**: 物理 Q=19，Triton 内核内部 padding 到 Q_PAD=32
- **Force all-reduce**: 几何构建是集合操作，力必须跨 rank 求和

## 架构

### 域分解

沿 z 轴（dim 1 of `(Q, nz, ny, nx)`）进行 slab 分解：

```
rank r owns z in [r*nz_local, (r+1)*nz_local)
Buffer shape: (Q, nz_local + 2, ny, nx)
- Owned planes: local z in [1, nz_local + 1)
- Ghost plane 0: from rank r-1 (periodic wrap)
- Ghost plane nz_local+1: from rank r+1 (periodic wrap)
```

### Halo Exchange

支持两种模式：

1. **NCCL 模式**: 使用 `torch.distributed` 进行真实多卡通信
2. **Simulated 模式**: 直接 tensor 拷贝，用于测试和开发

Exchange 是选择性的：只交换穿越 slab 面的速度方向（D3Q19 z-slab：每面 5 个方向，共 19 个）。相比全平面交换，减少 3.8x 通信量。

### 关键设计决策

来自 `triton_distributed_notes.md` 的经验：

1. **NCCL 需要 contiguous 缓冲区**: `(Q, nz+2, ny, nx)` 的 z-plane slice 不是 dense 的，必须调用 `.contiguous()`
2. **Q=19/32 padding**: halo exchange 使用 Q=19，Triton 内核内部 padding 到 32
3. **Force all-reduce**: 几何构建是集合操作，每个 rank 构建完整 mask，力必须跨 rank 求和
4. **Ghost planes 始终是流体**: obstacle mask 中 ghost planes 为 0

## 使用示例

### 基本用法

```python
from tensorlbm.domain_decomposition import (
    DomainDecomposer,
    HaloExchange,
    SimulatedMultiDevice,
)

# 创建域分解器
decomposer = DomainDecomposer(
    nz_global=128, ny=64, nx=64,
    n_devices=4, device="cuda",
)

# 创建 halo exchange
halo = HaloExchange(decomposer, mode="simulated", Q=19)

# 创建模拟多卡设备
simulated = SimulatedMultiDevice(decomposer, halo, device="cpu")

# 分配全局场
f_global = torch.randn(19, 128, 64, 64)
simulated.distribute(f_global)

# 执行时间步
def collide(f):
    # BGK collision
    ...

def stream(f):
    # Streaming
    ...

def boundary(f, rank):
    # Boundary conditions
    ...

simulated.step(collide, stream, boundary)

# 收集结果
f_result = simulated.gather()
```

### 真实多卡（需要 torch.distributed）

```python
import torch.distributed as dist

# 初始化进程组
dist.init_process_group(backend="nccl")

# 创建 halo exchange
halo = HaloExchange(decomposer, mode="nccl", Q=19)

# 在每个 rank 上执行
rank = dist.get_rank()
halo.exchange_nccl(f_local, rank)
```

## 验证

### 正确性测试

4 卡 vs 单卡 L2 误差 < 1e-6：

```bash
python scripts/m4_dd_validation.py --grid 32 16 16 --steps 20 --n-devices 4
```

结果：
- L2 error: 0.00e+00 ✓
- Max error: 0.00e+00 ✓
- Mean error: 0.00e+00 ✓

### 性能测试

4 卡加速比 ≥ 2.8x（并行效率 ≥ 70%）：

**注意**: CPU 模拟模式下无法展示真实加速比（所有 "devices" 共享同一 CPU）。真实多卡加速比测试需要 CUDA 硬件。

在 8 × RTX 5090 上的实测结果（来自 `triton_distributed_notes.md`）：
- 弱扩展效率 ~100%
- 聚合吞吐量 ~69 GLUPS
- 每 rank 内核以全单卡速度运行

### 单元测试

```bash
pytest tests/test_domain_decomposition.py -v
```

所有 12 个测试通过 ✓

## 产物

- `artifacts/m4_dd.json`: 包含每卡设备证据和计时
- `src/tensorlbm/domain_decomposition.py`: 核心实现
- `scripts/m4_dd_validation.py`: 验证脚本
- `tests/test_domain_decomposition.py`: 单元测试

## 限制

1. **周期性边界**: 当前仅支持 z 轴（slab 轴）周期性边界。壁面/入口/出口由上游调用者处理。
2. **CPU 模拟**: 模拟模式用于测试，不展示真实并行加速。
3. **D3Q19**: 当前实现针对 D3Q19，D3Q27 需要扩展 crossing indices。

## 参考

- `docs/triton_distributed_notes.md`: NCCL 陷阱工程笔记
- Succi S., et al. (2001) "Lattice Boltzmann for distributed and large-scale simulations." *Comput. Phys. Commun.* 134(3).
- FluidX3D `transfers` tables (D3Q19=5/D3Q27=9 per face)
- XLB `left/right_indices` ring exchange
