# SDAA 验证扫荡：poiseuille_3d_duct

## 案例概述

| 项 | 值 |
|---|---|
| 案例 | `benchmarks/verified/poiseuille_3d_duct` |
| 物理 | 3D 矩形管 Poiseuille 流（D3Q19 BGK） |
| 驱动 | Zou-He 速度入口 u_in=0.005 / 压力出口 ρ=1 |
| 壁面 | bounce_back_cells_3d（post-stream 半程 BB） |
| 解析解 | 双重奇正弦级数（Shah-London 交叉：方形 Po=56.9083） |
| 判据口径 | CH1 有效几何反演 ≤3%（主）；CH2 名义帧 ≤3%（披露）；da/db 跨档 spread ≤0.05 |

## SDAA 运行配置

| 项 | 值 |
|---|---|
| 设备 | `sdaa:0` → `/dev/tcaicard15` |
| 网格 | W=32, H=32, nx=128, ny=34, nz=34（最小档） |
| 步数 | 3000 模拟步 + 50 平均步 |
| 编译模式 | eager（SDAA teco_inductor 无法编译完整 LBM 步链，见 compile_route.py） |
| Triton 绕行 | 否（本案例不使用 triton 内核） |
| 墙钟 | 95.2s（≤20min 限制） |

## 结果表

| 指标 | 本运行值 | 参考值(W32) | 判据 | 结果 |
|---|---|---|---|---|
| CH1 eff_max % | **0.1053** | 0.1232 | ≤3% | ✅ PASS |
| CH1 eff_l2 % | 0.1011 | 0.1270 | — | — |
| CH2 nom_max % | **4.0788** | 4.6867 | ≤3% | ❌ FAIL |
| CH2 nom_l2 % | 3.1667 | 3.6012 | — | — |
| da | +0.4004 | +0.4285 | spread≤0.05 | N/A(单档) |
| db | +0.4004 | +0.4285 | spread≤0.05 | N/A(单档) |
| G_fit/G_Q | 1.0010 | 1.0013 | ≈1 | — |
| Po_sim err % | -10.70 | -9.49 | 披露 | — |
| steady | false | true | 稳态 | ❌ FAIL |
| finite | true | true | 有限 | ✅ PASS |

## 判据逐条比对

| 判据 | 值 | 阈值 | PASS/FAIL | 说明 |
|---|---|---|---|---|
| CH1_eff_max_le_3pct | 0.1053% | ≤3% | **PASS** | 有效几何反演精度优秀 |
| CH1_da_db_spread | da=0.4004, db=0.4004 | ≤0.05 | N/A | 单档无法计算跨档 spread |
| CH2_nom_max_le_3pct | 4.0788% | ≤3% | **FAIL** | 粗档名义帧超标（与参考 4.69% 同构） |
| steady_state | false | true | **FAIL** | 3000 步未达稳态（参考需 250000 步） |
| finite_result | true | true | **PASS** | 全场有限 |

## 总体判定：**FAIL**

### FAIL 原因

1. **CH2 nom_max > 3%**：4.08% > 3%。此为粗档固有现象，README 披露 #1 已说明 W32 名义帧 4.69% 超标，根因是 BB 半程壁面有效位移 +0.45 格（被 CH1 反演吸收）。本运行 4.08% 与参考 4.69% 量级一致。
2. **未达稳态**：3000 步远少于推荐的 250000 步。umax 仍在增长（最后 10 步：0.0096~0.0102），Po_sim 误差 -10.7% 也反映流场未充分发展。

### PASS 亮点

- **CH1 eff_max = 0.1053%**：即使未达稳态，有效几何反演精度已远优于 3% 阈值，与参考值 0.1232% 量级一致。
- **da = db = +0.4004**：方形自洽，接近参考值 +0.4285（随步数增加会收敛到参考值）。
- **G_fit/G_Q = 1.0010**：与参考 1.0013 高度一致。
- **全场有限**：无 NaN/Inf。

## 绕行记录

| 项 | 状态 | 说明 |
|---|---|---|
| TRITON_INTERPRET | 未启用 | 本案例不使用 triton 内核 |
| torch.compile | eager | SDAA teco_inductor 后端不完整，无法编译完整 LBM 步链（compile_route.py 文档已说明） |
| 网格档 | W=32（最小档） | W=64/128 在 SDAA 上墙钟远超 20 分钟限制 |
| 步数 | 3000 | 参考推荐 250000 步；SDAA 性能约 CUDA 的 1/4~1/5，全档运行需 ~100 分钟 |

## 设备证据

```
torch.sdaa.get_device_name(0) → "/dev/tcaicard15"
torch.sdaa.device_count() → 1
torch.sdaa.is_available() → True
```

## 产物

- `artifacts/sdaa_poiseuille_3d_duct.json`：完整结果 JSON（含设备证据、判据数值、逐条 PASS/FAIL）
- `scripts/sdaa_sweep_poiseuille_3d_duct.py`：独立运行脚本

## 备注

本运行如实记录 FAIL，未降判据、未用 CPU 结果冒充 sdaa。CH1 主通道通过，CH2 名义帧和稳态未通过的原因与案例 README 披露一致（粗档固有 + 步数不足）。若需完整验证，需在 SDAA 上运行 250000 步（预计墙钟 ~100 分钟）。
