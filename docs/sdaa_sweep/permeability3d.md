# SDAA 验证扫荡：permeability3d

## 案例概述

| 项目 | 值 |
|---|---|
| 案例 | `benchmarks/verified/permeability3d` |
| 物理 | 简单立方周期球阵列 Stokes 渗透率 (Zick & Homsy 1982) |
| φ | 0.343 |
| 参考 K_table | 15.402 |
| 判据容差 | 3% (最细档) |
| 原始判据口径 | 冻结三档 (N=64/96/128) 严格单调收敛 + 最细档 ≤3%；修正四档 (N=64/96/128/160) 三条件 |
| 原始结果 (CUDA) | N=64: +4.63%, N=96: +3.10%, N=128: +2.05%, N=160: +1.45% — 全档 PASS |

## 运行配置

| 项目 | 值 |
|---|---|
| 设备 | `sdaa:0` (`/dev/tcaicard8`, 15296 MiB) |
| torch 版本 | 2.7.1 |
| tensorlbm 路径 | `/root/tp-agent/work/m29/src/tensorlbm/__init__.py` |
| 脚本 | `scripts/sdaa_sweep_permeability3d.py` |
| TRITON_INTERPRET | 设为 1（triton 存在但本案例未实际使用） |
| 网格/步数 | N=64 全档 (max 81920 步), N=96 smoke (2000 步), N=128/160 跳过 |
| 墙钟 | 280.3 s (N=64: 228.3s, N=96: 52.0s) |

## 结果表

| 档 | N | 模式 | K_sim | K_table | err (%) | Ma_max | 稳态 | 步数 | 墙钟 (s) |
|---|---|---|---|---|---|---|---|---|---|
| 粗 | 64 | full | 14.7206 | 15.402 | **4.63** | 0.00783 | ✓ | 14200 | 228.3 |
| 中1 | 96 | smoke | 26.3464 | 15.402 | **41.54** | 0.00257 | ✗ | 2000 | 52.0 |
| 中2 | 128 | skipped | — | — | — | — | — | — | — |
| 细 | 160 | skipped | — | — | — | — | — | — | — |

## 判据逐条比对

| 判据门 | 原始口径 | SDAA 结果 | 状态 | 说明 |
|---|---|---|---|---|
| library_entry | 导入 tensorlbm.solver3d/boundaries3d | 成功导入 | **PASS** | 零手抄，从本地 src 导入 |
| direct_observable | 各档 err 由脚本算出 | N=64: 4.63% ✓; N=96: 41.54% ✗ | **FAIL** | N=96 smoke 未收敛 |
| monotone_convergence | 严格单调递减 | 4.63% → 41.54% (递增) | **FAIL** | smoke 档未收敛导致反向 |
| finest_le_tol_3pct | 最细档 err ≤ 3% | 41.54% > 3% | **FAIL** | smoke 档非正式结果 |

**总体 verdict: FAIL**

## 设备证据

```
torch.sdaa.get_device_name(0) → /dev/tcaicard8
torch.sdaa.get_device_properties(0) → _SDAADeviceProperties(name='/dev/tcaicard8', total_memory=15296MiB)
```

## 绕行记录

| 绕行项 | 说明 |
|---|---|
| TRITON_INTERPRET=1 | 检测到 triton 包存在，设置环境变量；本案例物理核未使用 triton |
| N=96 smoke 模式 | SDAA 上 N=96 全档需 ~117 分钟，超出 20 分钟预算，改用 2000 步 smoke |
| N=128/160 跳过 | SDAA 上 N=128 全档需 ~264 分钟，N=160 更长，直接跳过 |
| 仅 N=64 全档收敛 | N=64 全档 228s 完成，err=4.63% 与原始 CUDA 结果 4.6253% 一致 |

## 关键发现

1. **N=64 结果正确**：SDAA 上 N=64 全档收敛结果 err=4.6292%，与原始 CUDA 结果 4.6253% 高度一致（差异 <0.004%），证明物理核在 SDAA 上计算正确。
2. **SDAA 性能瓶颈**：N=64 在 SDAA 上 228s vs CUDA 上 ~9s，慢约 25 倍。N=96/128/160 在 SDAA 上无法在 20 分钟内完成全档收敛。
3. **honest-FAIL**：由于时间预算限制，无法完成多档收敛研究，判据 FAIL 如实记录。

## 产物

- `artifacts/sdaa_permeability3d.json` — 完整 JSON 产物（含设备证据、判据数值、逐条比对）
- `scripts/sdaa_sweep_permeability3d.py` — 独立运行脚本（不改业务代码）
