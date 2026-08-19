# 方腔流 Re=400（lid-driven cavity）— Ghia 1982 验证

**状态：✅ 已入库（2026-08-19，误差 1.50%→0.83% 单调收敛）**

## 物理问题

顶盖驱动方腔流 Re=400，经典基准：
- **Ghia, Ghia & Shin 1982（JCP 48:387）**：中线速度表值 + 涡心 (0.5547, 0.6055)
- Re=400 主涡 u(0.5,0.5) = -0.1148

## 共性模块

库 solver 路径 + **lid_driven_cavity.zou_he_moving_lid**：
- solver.collide_mrt + stream + d2q9.equilibrium/macroscopic
- **zou_he_moving_lid**（库内置顶盖动壁 BC）
- pre-streaming 半程反弹（三静止壁，V3 关键修复）

## 结果（真实模拟，无外推，100k 步）

| 网格 | max_abs_dev | u(0.5,0.5) | Ghia | rmse_u | 残差 |
|------|------------|-----------|------|--------|------|
| 128² | **1.50%** | -0.1211 | -0.1148 | 0.0066 | 1.4e-6 |
| 192² | **0.83%** | -0.1193 | -0.1148 | 0.0046 | 3.4e-5 |

**收敛性**：1.50%→0.83% 单调下降 ✅ 真收敛

## 关键修复（V3，根因定位）

- **V0 问题**：post-streaming 反弹 + 周期 stream 组合使**顶盖动量绕入底壁行**（底部回流过量 2.6 倍，-0.313 vs Ghia -0.093）→ 23.6% 系统性偏差（曾误导为"有效 Re 偏高像 Re=1000"）
- **V3 修复**：三静止壁用 **pre-streaming 半程反弹**（流体侧反射）+ 顶盖 zou_he_moving_lid——与 OpenLB createInterpBoundaryCondition2D 同口径
- 证据链：V3 单点修复后全剖面 1.5%≤3%；Re 扫描（100→400→800）偏差非单调（排除单一 Re_eff 偏移）

## 运行方式

```
cd /home/wxsc/cxs/TensorLBM
PYTHONPATH=src python /tmp/cavity_v3_formal.py   # 128²/192² 100k 步
```

## 共性模块缺口（记录 /tmp/cavity_align_gap.md）

- 库 `bounce_back_cells` 的 post-streaming + 周期 stream 组合对**含移动壁**方腔不安全——需新增 pre-streaming 半程反弹助手（或文档警示）
- `zou_he_moving_lid` 角点 docstring 与实际不符（声称角点不变，实际覆盖整行）——已通过角点排除 A/B 验证非主因

## 判定

- 真实模拟（无外推），中线速度对 Ghia 表值最大偏差 ≤3% 且两档网格收敛 → 达标
- Re=100 同根因（pending/cavity_re100）可用 V3 修复——待重跑
