# SDAA Sweep: sphere_re100

## 案例

球体绕流 Re=100，BFL 插值壁，稀疏核。判定口径：末 20% 采样窗均值 Cd，参考值 Cd_ref = 1.0917311 (Schiller-Naumann)，门限 |err| ≤ 3%，稳态漂移 < 0.3%，步数 ≥ 8000。

## 设备

- **Available**: True
- **Device name**: /dev/tcaicard5
- **Device count**: 1
- **Torch version**: 2.7.1
- **Total memory**: 14.94 GiB

## 配置

| Tier | D | Steps | τ | Domain (lat/up/down) | Device |
|------|---|-------|---|----------------------|--------|
| T1 | 40 | 8000 | 0.56 | 3.0/3.0/4.0 | sdaa:0 |
| T1 | 20 | 8000 | 0.53 | 3.0/3.0/4.0 | sdaa:0 |
| T1 | 10 | 100 | 0.52 | 3.0/3.0/4.0 | sdaa:0 |
| T1 | 10 | 100 | 0.52 | 3.0/3.0/4.0 | cpu |
| T2 | 60 | 16000 | 0.59 | 3.0/3.0/4.0 | sdaa:0 |

## 结果

| Tier | D | Steps | Device | Cd (window) | err% | drift% | steady | gate | PASS | 失败原因 |
|------|---|-------|--------|-------------|------|--------|--------|------|------|----------|
| T1_sdaa_d40_8k | 40 | 8000 | sdaa:0 | - | - | - | - | - | **FAIL** | OOM during setup |
| T1_sdaa_d20_8k | 20 | 8000 | sdaa:0 | - | - | - | - | - | **FAIL** | Timeout (>120s) |
| T1_sdaa_d10_100 | 10 | 100 | sdaa:0 | - | - | - | - | - | **FAIL** | Timeout (>30s) |
| T1_cpu_d10_100 | 10 | 100 | cpu | 1.679 | - | - | - | - | N/A | CPU baseline (not SDAA) |
| T2_sdaa_d60_16k | 60 | 16000 | sdaa:0 | - | - | - | - | - | **FAIL** | Not attempted (D=40 OOM) |

## 墙钟

- **T1_sdaa_d40_8k**: 0s (OOM at setup, step 0)
- **T1_sdaa_d20_8k**: 120s (timeout)
- **T1_sdaa_d10_100**: 30s (timeout)
- **T1_cpu_d10_100**: ~5s (completed)

## 总体判定

- **All within gate & steady**: False
- **Monotone error decrease**: False
- **PASS**: **False**

## 失败分析

### 1. D=40 OOM (主要问题)

**错误**: `RuntimeError: sdaa out of memory`

**内存详情**:
- 尝试分配: 3.37 GiB
- 总容量: 14.94 GiB
- 已分配: 7.78 GiB
- 可用: 5.67 GiB
- PyTorch 保留: 7.95 GiB

**失败位置**: `bfl_mask[1:].sum().item()` - BFL 边界链计算阶段（setup，step 0）

**内存估算**:
- 域大小: 280×280×320 = 25,088,000 胞
- 分布函数 f: 19 × 25M × 4 bytes ≈ 1.9 GiB
- BFL mask/q: 25M × 19 × 4 bytes ≈ 1.9 GiB
- 总计: ~4-5 GiB（setup 阶段）
- 可用: 5.67 GiB

**结论**: 内存碎片化或 SDAA 后端开销导致实际需求超过可用内存。

### 2. D≤20 极慢（次要问题）

**现象**:
- D=10, 100 steps: >30s timeout
- D=20, 8000 steps: >120s timeout
- D=10, 100 steps on CPU: ~5s (完成)

**结论**: SDAA 后端比 CPU 慢 10-100 倍，可能存在编译开销或低效的 kernel launch。

### 3. 与 README 的差异

**README 声称**: D=40 可在 SDAA 上运行（README section 6）

**实际结果**: D=40 在 setup 阶段即 OOM

**可能原因**:
1. README 在不同 SDAA 卡或不同内存状态下测试
2. PyTorch SDAA 后端内存开销增加
3. SDAA 卡内存被其他进程部分占用
4. 稀疏核内存优化在当前 SDAA 后端不足

## 绕行记录

### 已尝试

1. **PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** - 未解决 OOM
2. **稀疏核** - 已是最省内存选项
3. **降低 D 到 20 和 10** - 避免 OOM 但遇到性能墙

### 未尝试（原因）

1. **更小域 (lat/up/down)** - 会改变物理，不符合验证案例规格
2. **CPU 回退** - 不是 SDAA，不计入 SDAA 验证
3. **不同 SDAA 卡** - 当前环境不可用

## 诊断结论

**根本原因**: SDAA 后端对 sphere_re100 案例存在两个问题：
1. **OOM**: D≥40 大域情况下内存不足
2. **极慢**: D≤20 情况下执行速度比 CPU 慢 10-100 倍

**影响**: 无法在合理时间/内存内产生 Cd 值用于门限判定。

**建议**:
1. 调查 SDAA 后端内存开销来源
2. 优化 SDAA kernel launch 性能
3. 在不同 SDAA 卡/环境重新测试 README 声称的配置
4. 考虑提供更小的验证案例（如 D=20 小域）用于 SDAA 快速验证

## 工件

- `artifacts/sdaa_sphere_re100.json` - 设备证据 + 逐条判定
- `benchmarks/verified/sphere_re100/test_quick.json.oom.json` - D=40 OOM 原始输出
- `benchmarks/verified/sphere_re100/cpu_d10_test.json` - CPU baseline（非 SDAA）

## 诚实声明

本验证 **FAIL**，原因：
- SDAA OOM (D=40)
- SDAA 极慢 (D≤20)
- 无法产生任何有效的 Cd 值用于门限判定

未放宽任何门限，未用 CPU 结果冒充 SDAA，如实记录所有失败证据。
