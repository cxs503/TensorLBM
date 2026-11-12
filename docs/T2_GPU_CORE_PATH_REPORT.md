# T2 GPU Core Path 统一优化报告

## 1. 问题设定

**锚点**: GAP-LBM-T2-GPU-COREPATH  
**目标**: T2 GPU 核心路径统一优化——碰撞迁移+边界条件融合/内存管理/多后端/多GPU通信/精度性能权衡  
**优化对象**: T3已验证物理算例（Poiseuille流、Taylor-Green衰减、方腔流Re=100）

### 基准出处
| 算例 | 参考 | 判据 |
|------|------|------|
| Poiseuille 2D | 解析抛物线 u(y)=4u_max·y(H-y)/H² | 相对误差 < 1% |
| Taylor-Green 2D | 解析衰减 KE(t)=KE₀·exp(-4νt) | 相对误差 < 5% |
| 方腔流 Re=100 | Ghia et al. (1982) JCP 48:387 | 中心线速度误差 < 10% |

## 2. 方法与实现

### 2.1 现有基础设施盘点

TensorLBM 已具备以下 GPU 优化模块（总计 ~5000 行）：

| 模块 | 行数 | 功能 |
|------|------|------|
| `triton_fused.py` | 962 | Triton融合碰撞+流送（周期域） |
| `triton_fused_diff.py` | 443 | 可微分版本（autograd支持） |
| `triton_fused_distributed.py` | 680 | 多GPU slab分解+NCCL halo交换 |
| `triton_fused_obstacle.py` | 2911 | 障碍物/边界条件扩展 |
| `perf_solver.py` | ~400 | PyTorch优化求解器（预分配缓冲） |
| `perf_buffers.py` | ~200 | 预分配内存池 |
| `precision.py` | ~150 | 计算/存储精度策略 |
| `multi_gpu.py` | ~300 | 多GPU域分解 |
| `backends/` | ~500 | 多后端支持(torch/paddle/mindspore) |

### 2.2 新增统一核心路径模块

**文件**: `src/tensorlbm/gpu_core_path.py` (新增)

提供统一 API 整合五个优化维度：

```python
from tensorlbm.gpu_core_path import GPUCorePathSolver, GPUCorePathConfig

cfg = GPUCorePathConfig(
    nx=128, ny=65, nz=64,
    re_tau=180, u_tau=0.05,
    precision="FP32FP32",
    backend="auto",  # auto | triton | torch_compile | eager
)
solver = GPUCorePathSolver(cfg)
info = solver.run(n_steps=10000)
```

### 2.3 后端选择策略

```
auto → Triton (CUDA) → torch.compile (CPU/CUDA) → eager PyTorch
```

### 2.4 精度策略

| 策略 | 计算精度 | 存储精度 | 适用场景 |
|------|----------|----------|----------|
| FP64FP64 | float64 | float64 | 参考/收敛审计 |
| FP64FP32 | float64 | float32 | 高精度生产 |
| FP32FP32 | float32 | float32 | 默认生产 |
| FP32FP16 | float32 | float16 | 带宽受限 (~2e-4误差) |

### 2.5 多GPU通信优化

- **Slab分解**: 沿z轴分割域，每GPU含2个ghost平面
- **选择性传输**: D3Q19仅传输5个穿越方向（非全部19个），带宽减少3.8倍
- **NCCL异步交换**: 计算与通信重叠

## 3. 参数表

### Poiseuille 2D
| 参数 | 符号 | 值 |
|------|------|-----|
| 网格 | n | 65 |
| 雷诺数 | Re | 100 |
| 最大速度 | u_max | 0.1 |
| 松弛时间 | τ | 0.8 |
| 步数 | steps | 5000 |

### Taylor-Green 2D
| 参数 | 符号 | 值 |
|------|------|-----|
| 网格 | n | 64 |
| 初始速度 | u₀ | 0.1 |
| 粘度 | ν | 0.01 |
| 步数 | steps | 2000 |

### 方腔流 Re=100
| 参数 | 符号 | 值 |
|------|------|-----|
| 网格 | n | 129 |
| 雷诺数 | Re | 100 |
| 盖速度 | u_lid | 0.1 |
| 步数 | steps | 20000 |

## 4. 实现状态

### 已完成
- [x] 统一核心路径模块 `gpu_core_path.py`
- [x] 后端自动选择逻辑
- [x] 精度策略集成
- [x] 多GPU分布式求解器框架
- [x] 预分配缓冲池集成

### 待验证（需GPU环境）
- [ ] Triton融合内核性能基准（RTX 5090: 8.6 GLUPS @ n=256）
- [ ] 多GPU弱扩展效率（8×GPU: ~69 GLUPS）
- [ ] FP32FP16精度验证（~2e-4相对误差）

### 当前环境限制
- 本机无CUDA设备，torch.compile在CPU上有已知bug（shape索引问题）
- 基准测试使用eager模式验证正确性

## 5. 失败模式与工件

### 失败原因
torch.compile 在 CPU 上对 `f.shape[0]` 等索引操作有已知限制（Dynamo bug），导致编译失败。这是 PyTorch 2.7.1 的已知问题，在 GPU 上通常不受影响。

### 工件路径
- 核心模块: `src/tensorlbm/gpu_core_path.py`
- 基准脚本: `benchmarks/bench_gpu_core_path.py`
- 本文档: `docs/T2_GPU_CORE_PATH_REPORT.md`

### 复现命令
```bash
# GPU环境（推荐）
PYTHONPATH=src python benchmarks/bench_gpu_core_path.py --backend triton

# CPU环境（eager模式，验证正确性）
PYTHONPATH=src python benchmarks/bench_gpu_core_path.py --backend eager
```

## 6. 结论

T2 GPU 核心路径统一优化已完成框架集成，整合了现有的 Triton 融合内核、预分配缓冲、精度策略和多GPU通信模块。由于当前环境无 GPU，性能基准需在 GPU 集群上完成验证。

**关键成果**:
1. 统一 API 封装五个优化维度
2. 后端自动降级策略（Triton → torch.compile → eager）
3. 多GPU选择性传输（3.8x带宽优化）
4. 精度/性能权衡策略文档化

**下一步**: 在 GPU 集群上运行完整基准测试，验证性能指标（8.6 GLUPS @ n=256）和精度指标（<1%误差）。
