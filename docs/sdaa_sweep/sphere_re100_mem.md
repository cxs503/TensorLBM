# SDAA Sweep: sphere_re100_mem

## 案例信息

- **案例名称**: sphere_re100_mem
- **物理问题**: 球体绕流 Re=100，BFL 插值边界 + 稀疏核大域
- **入口文件**: `benchmarks/verified/sphere_re100/run.py`
- **案例文档**: `benchmarks/verified/sphere_re100_mem/README.md`

## 配置参数

### T1 档 (D=40)
- 分辨率: D=40
- 域尺寸: 280×280×320 (nz×ny×nx)
- 步数: 12,000
- tau: 0.56
- 核类型: sparse (稀疏 BFL 核)

### T2 档 (D=60)
- 分辨率: D=60
- 域尺寸: 420×420×480 (nz×ny×nx)
- 步数: 24,000
- tau: 0.59
- 核类型: sparse (稀疏 BFL 核)

## 设备信息

| 项目 | 值 |
|------|-----|
| 设备类型 | SDAA |
| 设备名称 | /dev/tcaicard3 |
| 设备数量 | 1 |
| 总显存 | 14.94 GiB |
| 当前设备 | 0 |

## 判据口径

案例采用四门标准核验：

1. **library_entry**: 使用 `src/tensorlbm` 库路径驱动
2. **direct_observable**: 直接观测量 |err| ≤ 3%
3. **monotone_convergence**: ≥2 档单调收敛（|err| 随 D 严格单调降）
4. **reference_locked_first**: 参考先锁定后跑

参考值：Schiller–Naumann Cd_ref = 24/Re·(1+0.15·Re^0.687) = 1.0917310910948732

## 运行结果

### 状态：❌ FAIL（显存不足）

两档均因 SDAA 显存不足而失败：

#### T1 (D=40)
- **状态**: FAIL
- **错误**: `RuntimeError: sdaa out of memory`
- **详情**: Tried to allocate 990.00 MiB (SDAA 0; 14.94 GiB total capacity; 2.23 GiB already allocated; 1.38 GiB free; 2.29 GiB reserved in total by Torch-SDAA)
- **失败位置**: `compute_q_sphere` 分配 q_field (19, nz, ny, nx) 时

#### T2 (D=60)
- **状态**: FAIL
- **错误**: `RuntimeError: sdaa out of memory`
- **详情**: Tried to allocate 1.72 GiB (SDAA 0; 14.94 GiB total capacity; 2.56 GiB already allocated; 287.97 MiB free; 3.39 GiB reserved in total by Torch-SDAA)
- **失败位置**: `compute_q_sphere` 分配 q_field (19, nz, ny, nx) 时

## 判据逐条比对

| 判据 | 要求 | 实际 | 结果 |
|------|------|------|------|
| library_entry | 使用 tensorlbm 库 | 使用 tensorlbm.bfl_d3q19, tensorlbm.solver3d 等 | ✅ PASS |
| direct_observable | \|err\| ≤ 3% | 未能运行，无结果 | ❌ FAIL (OOM) |
| monotone_convergence | \|err\| 单调下降 | 未能运行，无结果 | ❌ FAIL (OOM) |
| reference_locked_first | 参考先锁定 | Schiller-Naumann 公式已锁定 | ✅ PASS |

**总体判定**: ❌ FAIL

## 失败原因分析

### 显存需求估算

T1 (D=40) 域尺寸 280×280×320 = 25,088,000 网格点
- q_field: 19 × 280 × 280 × 320 × 4 bytes = 1.72 GiB
- bfl_mask: 19 × 280 × 280 × 320 × 1 byte = 456 MiB
- 分布函数 f: 19 × 280 × 280 × 320 × 4 bytes = 1.72 GiB
- 总计约 4-5 GiB（仅基础数据结构）

T2 (D=60) 域尺寸 420×420×480 = 84,672,000 网格点
- q_field: 19 × 420 × 420 × 480 × 4 bytes = 5.78 GiB
- 总计约 12-15 GiB

### SDAA 设备限制

- 总显存: 14.94 GiB
- 可用显存: ~1.38 GiB (T1) / ~287 MiB (T2)
- PyTorch 保留: ~2.29 GiB / ~3.39 GiB

### 对比 NVIDIA GPU

案例文档标注运行时长参考：
- T1 约 180 秒（GPU5，RTX 5090，稀疏核）
- T2 约 600 秒（GPU5，RTX 5090，稀疏核）

RTX 5090 具有 32 GiB 显存，足以容纳该案例。

## 绕行记录

### 尝试的绕行方案

1. **torch.cuda → torch.sdaa 猴子补丁**
   - 将 `torch.cuda.empty_cache` 等函数映射到 `torch.sdaa` 对应函数
   - 成功解决 API 兼容性问题
   - 但无法解决显存不足问题

2. **降低网格分辨率**
   - 未尝试（会改变判据口径，违反任务要求）

3. **使用 CPU 运行**
   - 未尝试（违反任务要求：禁止用 CPU 结果冒充 sdaa）

### 未尝试的绕行方案

- 多卡并行（需要修改业务代码）
- 域分解（需要修改业务代码）
- 降低采样频率（不影响显存峰值）

## 结论

该案例在当前 SDAA 设备（14.94 GiB 显存）上**无法运行**，原因是显存不足。

### 建议

1. **硬件升级**: 使用 32 GiB 或更大显存的 SDAA 设备
2. **算法优化**: 
   - 实现更激进的显存节省策略（如分块计算 q_field）
   - 优化稀疏核的显存使用
3. **案例调整**: 
   - 为 SDAA 设备创建更小分辨率的验证案例（如 D=20）
   - 但需重新验证判据口径

### 诚实声明

本次测试如实记录了 FAIL 状态与原因，未降低判据标准，未使用 CPU 结果冒充 SDAA。

## 产物清单

- `artifacts/sdaa_sphere_re100_mem.json`: 完整运行记录与判据结果
- `scripts/sdaa_sweep_sphere_re100_mem.py`: 独立运行脚本
- `docs/sdaa_sweep/sphere_re100_mem.md`: 本文档

## 时间戳

- 运行时间: 2026-10-29
- 墙钟时间: < 1 分钟（因 OOM 快速失败）
